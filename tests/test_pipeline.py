"""
tests/test_pipeline.py
─────────────────────────
Fast sanity tests that don't require DeBERTa weights, GROQ_API_KEY, or
TAVILY_API_KEY. Heavy stages (claim_detection's models, evidence's Tavily
client, verdict's Groq client) are monkeypatched with deterministic
fakes so the *assembly* logic (pipeline.py, analysis/*) can be checked
in CI or before you have API keys / model weights set up.

Run with:  python -m tests.test_pipeline
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from unittest.mock import patch


def _fake_sentences(article_text, claim_threshold=0.65):
    data = [
        ("In my opinion the weather has been strange lately.", False, 0.1, 0.3),
        ("According to NASA, 2024 was the hottest year on record.", True, 0.9, 0.4),
        ("The Great Wall of China was built in 1952 by the Republic of China.", True, 0.85, 0.6),
    ]
    out = []
    pos = 0
    for i, (text, is_claim, p_claim, p_ai) in enumerate(data):
        start = article_text.find(text, pos) if text in article_text else pos
        end = start + len(text)
        pos = end
        out.append({
            "index": i, "text": text, "char_start": start, "char_end": end,
            "p_claim": p_claim, "p_ai_generated": p_ai, "is_claim": is_claim,
            "ai_label": "ai" if p_ai >= 0.7 else "human" if p_ai < 0.4 else "uncertain",
            "ai_confidence": round(abs(p_ai - 0.5) * 2, 4),
        })
    return out


def _fake_evidence(claims, max_results=8, top_k=3):
    for c in claims:
        c["evidence"] = [{"title": "Fake Source", "url": "https://example.com/fake", "content": "Fake content.", "relevance_score": 0.9}]
    return claims


def _fake_score_claims(claims):
    for i, c in enumerate(claims):
        supported = i % 2 == 0
        c["verdict"] = "supported" if supported else "contradicted"
        c["contradiction_score"] = 0.0 if supported else 0.9
        c["justification"] = "fake justification"
        c["evidence_spans_used"] = ["https://example.com/fake"]
        from core.verdict import assign_risk_tier, compute_risk_score
        c["risk_score"] = compute_risk_score(c["p_ai_generated"], c["p_claim"], c["contradiction_score"])
        c["risk_tier"] = assign_risk_tier(c["risk_score"], verdict=c["verdict"])
    return claims


def _fake_article_metrics(article_text, sentences=None):
    # Avoids loading the sentence-transformer model (needs a network call
    # to Hugging Face, which isn't available in every environment/sandbox).
    return {
        "readability": {"flesch_reading_ease": 55.0, "flesch_kincaid_grade": 9.1,
                         "gunning_fog": 10.0, "smog_index": 9.0, "coleman_liau_index": 10.0,
                         "automated_readability_index": 9.5},
        "coherence": {"avg_coherence": 0.32, "min_coherence": 0.1, "max_coherence": 0.5, "per_pair": [0.2, 0.44]},
    }


def _fake_judge_article(article_text, title=None, **kwargs):
    return {
        "ratings": {"credibility": 4, "clarity": 4, "neutrality": 3, "complexity": 3, "coherence": 4, "quality": 4},
        "reasons": {k: "fake reason" for k in ["credibility", "clarity", "neutrality", "complexity", "coherence", "quality"]},
        "overall_reasoning": "fake overall reasoning",
        "model": "fake-model",
    }


def test_pipeline_assembles_with_claims():
    with patch("pipeline.analyze_article_sentences", side_effect=_fake_sentences), \
         patch("pipeline.attach_evidence_to_claims", side_effect=_fake_evidence), \
         patch("pipeline.score_claims", side_effect=_fake_score_claims), \
         patch("pipeline.judge_article", side_effect=_fake_judge_article), \
         patch("pipeline.compute_article_metrics", side_effect=_fake_article_metrics):
        from pipeline import analyze_article

        result = analyze_article("some article text", claim_threshold=0.65, title="Test headline", include_llm_judge=True)

        assert result["num_claims_found"] == 2
        assert result["credibility"]["score"] is not None
        assert "highlights" in result["explainability"]
        assert result["ai_detection"]["sentence_counts"]["ai"] + \
               result["ai_detection"]["sentence_counts"]["human"] + \
               result["ai_detection"]["sentence_counts"]["uncertain"] == 3
        print("test_pipeline_assembles_with_claims: OK")
        print(f"  credibility={result['credibility']['score']} tier={result['credibility']['tier']}")
        print(f"  overall_risk={result['overall_risk_score']} tier={result['overall_risk_tier']}")


def test_pipeline_handles_opinion_only_article():
    def _no_claims(article_text, claim_threshold=0.65):
        return [{"index": 0, "text": "I really don't like Mondays.", "char_start": 0, "char_end": 29,
                 "p_claim": 0.05, "p_ai_generated": 0.2, "is_claim": False, "ai_label": "human", "ai_confidence": 0.6}]

    with patch("pipeline.analyze_article_sentences", side_effect=_no_claims), \
         patch("pipeline.judge_article", side_effect=_fake_judge_article), \
         patch("pipeline.compute_article_metrics", side_effect=_fake_article_metrics):
        from pipeline import analyze_article

        result = analyze_article("I really don't like Mondays.", include_llm_judge=True)
        assert result["num_claims_found"] == 0
        assert result["overall_risk_score"] == 0.0
        assert result["credibility"]["score"] is not None  # still scored from remaining components
        print("test_pipeline_handles_opinion_only_article: OK")
        print(f"  credibility={result['credibility']['score']} missing={result['credibility']['missing_components']}")


def _fake_many_sentences(article_text, claim_threshold=0.65):
    """10 check-worthy claims with distinct p_claim values (0.91 down to 0.73,
    not in sorted order) so the test can check the CAP picks the highest ones,
    not just the first N in document order."""
    p_claims = [0.80, 0.91, 0.75, 0.88, 0.73, 0.85, 0.79, 0.90, 0.77, 0.83]
    out = []
    for i, p in enumerate(p_claims):
        out.append({
            "index": i, "text": f"Claim number {i} with p_claim {p}.", "char_start": i * 10, "char_end": i * 10 + 9,
            "p_claim": p, "p_ai_generated": 0.5, "is_claim": True,
            "ai_label": "uncertain", "ai_confidence": 0.0,
        })
    return out


def test_pipeline_caps_claims_verified_by_checkworthiness():
    seen_lengths = []

    def _counting_evidence(claims, max_results=8, top_k=3):
        seen_lengths.append(len(claims))
        return _fake_evidence(claims, max_results, top_k)

    with patch("pipeline.analyze_article_sentences", side_effect=_fake_many_sentences), \
         patch("pipeline.attach_evidence_to_claims", side_effect=_counting_evidence), \
         patch("pipeline.score_claims", side_effect=_fake_score_claims), \
         patch("pipeline.judge_article", side_effect=_fake_judge_article), \
         patch("pipeline.compute_article_metrics", side_effect=_fake_article_metrics):
        from pipeline import analyze_article

        result = analyze_article("ten-claim article", include_llm_judge=False, max_claims_to_verify=4)

        assert result["num_claims_found"] == 10
        assert result["num_claims_verified"] == 4
        assert result["num_claims_skipped"] == 6
        assert seen_lengths == [4], f"evidence stage should only ever see the capped 4, got {seen_lengths}"

        verified = [c for c in result["claims"] if "risk_score" in c]
        skipped = [c for c in result["claims"] if c.get("verification_skipped")]
        assert len(verified) == 4 and len(skipped) == 6

        # the 4 verified must be exactly the 4 highest p_claim values: 0.91, 0.90, 0.88, 0.85
        verified_p = sorted(c["p_claim"] for c in verified)
        assert verified_p == [0.85, 0.88, 0.90, 0.91]

        for c in skipped:
            assert "skip_reason" in c and "risk_score" not in c

        print("test_pipeline_caps_claims_verified_by_checkworthiness: OK")
        print(f"  verified p_claim values: {verified_p}")


if __name__ == "__main__":
    test_pipeline_assembles_with_claims()
    test_pipeline_handles_opinion_only_article()
    test_pipeline_caps_claims_verified_by_checkworthiness()
    print("\nALL TESTS PASSED")
