"""
pipeline.py
────────────
End-to-end orchestrator for "Machine Credibility: How News Readers
Evaluate AI-Generated Content".

    Stage 0  Article metrics       core.metrics                (readability, coherence)
    Stage 1  Sentence scoring      core.claim_detection         (check-worthiness + AI probability, every sentence)
    Stage 2  Evidence retrieval    core.evidence                (web search + relevance ranking, claims only)
    Stage 3  NLI verdict + risk    core.verdict                 (claims only)
    Stage 4  Engagement/sentiment  analysis.engagement
    Stage 5  LLM reader-simulation analysis.llm_judge            (optional, one Groq call for the whole article)
    Stage 6  Composite credibility analysis.credibility
    Stage 7  Explainability layer  analysis.explainability

Stages 2 and 3 are skipped entirely when there are no check-worthy claims
(pure-opinion text) - no wasted search/LLM calls. Stage 6 still runs so an
opinion-only article gets a credibility score built from whatever
components remain available.

Stages 2-3 also do each other one claim at a time (one Tavily search +
one Groq call per claim, with a short sleep between calls to respect
rate limits), so a long article with many check-worthy claims can take
several minutes. `max_claims_to_verify` bounds this: when more claims
are found than the cap, only the top N by check-worthiness (p_claim)
go through evidence retrieval + verdict scoring; the rest are returned
unverified with a `verification_skipped` flag and a visible reason,
rather than silently dropped or silently all processed no matter how
long it takes. analysis/credibility.py's factual_reliability component
already only counts claims that actually have a risk_score, so skipped
claims correctly don't influence the credibility score either way.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional

from analysis.credibility import compute_credibility
from analysis.engagement import compute_engagement_features
from analysis.explainability import build_explainability
from analysis.llm_judge import judge_article
from core.claim_detection import analyze_article_sentences, claims_from_sentences, summarize_ai_signal
from core.evidence import attach_evidence_to_claims
from core.metrics import compute_article_metrics
from core.verdict import W_AI, W_CLAIM, W_CONTRADICTION, assign_risk_tier, score_claims


def analyze_article(
    article_text: str,
    claim_threshold: float = 0.65,
    title: Optional[str] = None,
    include_llm_judge: bool = True,
    max_claims_to_verify: Optional[int] = 15,
) -> Dict[str, Any]:
    if not article_text or not article_text.strip():
        raise ValueError("article_text is empty or whitespace-only.")

    t0 = time.perf_counter()
    timings: Dict[str, float] = {}
    article_len = len(article_text)
    print(f"\n{'=' * 70}\nMACHINE CREDIBILITY PIPELINE ({article_len} chars)\n{'=' * 70}")

    # Stage 1: sentence scoring (segmentation happens once here)
    t = time.perf_counter()
    sentences = analyze_article_sentences(article_text, claim_threshold)
    claims: List[Dict[str, Any]] = claims_from_sentences(sentences)
    timings["sentence_scoring"] = round(time.perf_counter() - t, 3)
    print(f"[stage 1] {len(sentences)} sentences, {len(claims)} check-worthy claim(s)")

    # Stage 0: article metrics - reuse the same sentence segmentation for coherence
    t = time.perf_counter()
    sentence_texts = [s["text"] for s in sentences]
    article_metrics = compute_article_metrics(article_text, sentences=sentence_texts)
    timings["article_metrics"] = round(time.perf_counter() - t, 3)

    # Stages 2-3: evidence + verdicts (skipped if no claims; capped by
    # max_claims_to_verify so a long article with many claims doesn't take
    # several minutes - see module docstring)
    num_claims_skipped = 0
    if claims:
        claims_to_verify = claims
        if max_claims_to_verify is not None and len(claims) > max_claims_to_verify:
            ranked = sorted(claims, key=lambda c: c.get("p_claim", 0.0), reverse=True)
            claims_to_verify = ranked[:max_claims_to_verify]
            verify_ids = {id(c) for c in claims_to_verify}
            skipped = [c for c in claims if id(c) not in verify_ids]
            num_claims_skipped = len(skipped)
            for c in skipped:
                c["verification_skipped"] = True
                c["skip_reason"] = (
                    f"Not checked against evidence - only the top {max_claims_to_verify} most "
                    f"check-worthy claim(s) (by p_claim) are verified per run, out of {len(claims)} "
                    f"found, to keep analysis time reasonable."
                )
            print(f"[stage 2-3] {len(claims)} claims found - verifying top {max_claims_to_verify} "
                  f"by check-worthiness, skipping {num_claims_skipped}")

        t = time.perf_counter()
        claims_to_verify = attach_evidence_to_claims(claims_to_verify, max_results=8, top_k=3)
        timings["evidence_retrieval"] = round(time.perf_counter() - t, 3)

        t = time.perf_counter()
        claims_to_verify = score_claims(claims_to_verify)
        timings["verdict_scoring"] = round(time.perf_counter() - t, 3)

        verified_claims = [c for c in claims if "risk_score" in c]
        if verified_claims:
            risk_scores = [c["risk_score"] for c in verified_claims]
            overall_risk_score = round(sum(risk_scores) / len(risk_scores), 2)
            highest_risk_claim = max(verified_claims, key=lambda c: c["risk_score"])
        else:
            overall_risk_score = 0.0
            highest_risk_claim = None
    else:
        print("[stage 2-3] no check-worthy claims - skipping evidence + verdict stages")
        overall_risk_score = 0.0
        highest_risk_claim = None
    overall_risk_tier = assign_risk_tier(overall_risk_score)

    # Stage 4: engagement / sentiment
    t = time.perf_counter()
    engagement = compute_engagement_features(article_text, headline=title, sentences=sentence_texts)
    timings["engagement"] = round(time.perf_counter() - t, 3)

    # Stage 5: LLM reader-simulation (optional)
    llm = None
    if include_llm_judge:
        t = time.perf_counter()
        llm = judge_article(article_text, title=title)
        timings["llm_judge"] = round(time.perf_counter() - t, 3)

    # Stage 6: composite credibility
    credibility = compute_credibility(claims, article_metrics, engagement, sentence_texts, llm_judge=llm)

    # Stage 7: explainability
    explainability = build_explainability(
        sentences, claims, credibility, engagement["sentiment_arc"],
        risk_weights={"ai": W_AI, "claim": W_CLAIM, "contradiction": W_CONTRADICTION},
    )

    timings["total"] = round(time.perf_counter() - t0, 3)
    print(f"credibility={credibility['score']} ({credibility['tier']}), "
          f"risk={overall_risk_score} ({overall_risk_tier}), total {timings['total']}s\n")

    return {
        "article_length": article_len,
        "num_claims_found": len(claims),
        "num_claims_verified": len(claims) - num_claims_skipped,
        "num_claims_skipped": num_claims_skipped,
        "max_claims_to_verify": max_claims_to_verify,
        "claims": claims,
        "overall_risk_score": overall_risk_score,
        "overall_risk_tier": overall_risk_tier,
        "highest_risk_claim": highest_risk_claim,
        "article_metrics": article_metrics,
        "title": title,
        "ai_detection": summarize_ai_signal(sentences),
        "engagement": engagement,
        "llm_judge": llm,
        "credibility": credibility,
        "explainability": explainability,
        "timings": timings,
    }


if __name__ == "__main__":
    SAMPLE = (
        "In my opinion, world leaders are completely detached from everyday realities. "
        "According to NASA, the global average surface temperature in 2024 was 1.29 degrees C "
        "above the pre-industrial baseline, confirming it as the hottest year on record. "
        "However, conspiracy theories allege the Great Wall of China was built in 1952 by "
        "the Republic of China government. We clearly need stronger institutional integrity."
    )
    result = analyze_article(SAMPLE, claim_threshold=0.60, title="World leaders and the facts")
    print(json.dumps(result, indent=2, ensure_ascii=False))
