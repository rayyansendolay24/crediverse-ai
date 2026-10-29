"""
analysis/explainability.py
─────────────────────────────
Turns raw pipeline output into a UI-ready explainability payload:

1. highlights          per-sentence AI label/confidence, claim status, risk
2. claim_explanations  per-claim verdict, confidence, justification, sources,
                        exact risk-score arithmetic
3. sources             de-duplicated source attribution across all claims
4. credibility         plain-language reasons behind the credibility score
5. methods             what each number means and its known limits

Confidence conventions:
    AI detection confidence  = |p_ai - 0.5| * 2   (0 = on the fence, 1 = certain)
    Verdict confidence       = |contradiction - 0.5| * 2 for supported/
                                contradicted, 0 for insufficient_evidence
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


def _find_sentence_index(claim: Dict[str, Any], sentences: List[Dict[str, Any]]) -> Optional[int]:
    if claim.get("sentence_index") is not None:
        return claim["sentence_index"]
    for s in sentences:
        if s["char_start"] == claim.get("char_start") and s["char_end"] == claim.get("char_end"):
            return s["index"]
    return None


def build_highlights(sentences: List[Dict[str, Any]], claims: List[Dict[str, Any]], sentiment_arc: List[float]) -> List[Dict[str, Any]]:
    by_sentence = {}
    for c in claims:
        idx = _find_sentence_index(c, sentences)
        if idx is not None:
            by_sentence[idx] = c

    out = []
    for s in sentences:
        c = by_sentence.get(s["index"])
        sentiment = sentiment_arc[s["index"]] if s["index"] < len(sentiment_arc) else None
        out.append({
            "index": s["index"], "text": s["text"], "char_start": s["char_start"], "char_end": s["char_end"],
            "ai_label": s["ai_label"], "p_ai": s["p_ai_generated"], "ai_confidence": s["ai_confidence"],
            "is_claim": s["is_claim"], "p_claim": s["p_claim"],
            "verdict": c.get("verdict") if c else None,
            "risk_tier": c.get("risk_tier") if c else None,
            "risk_score": c.get("risk_score") if c else None,
            "sentiment": sentiment,
        })
    return out


def _verdict_confidence(claim: Dict[str, Any]) -> float:
    if claim.get("verdict") == "insufficient_evidence":
        return 0.0
    return round(abs(claim.get("contradiction_score", 0.5) - 0.5) * 2, 3)


def build_claim_explanations(claims: List[Dict[str, Any]], sentences: List[Dict[str, Any]], risk_weights: Dict[str, float]) -> List[Dict[str, Any]]:
    out = []
    for i, c in enumerate(claims):
        p_ai, p_cl, contra = c.get("p_ai_generated", 0.0), c.get("p_claim", 0.0), c.get("contradiction_score", 0.5)
        out.append({
            "claim_index": i,
            "sentence_index": _find_sentence_index(c, sentences),
            "text": c.get("text"),
            "verdict": c.get("verdict"),
            "verdict_confidence": _verdict_confidence(c),
            "justification": c.get("justification"),
            "cited_sources": c.get("evidence_spans_used", []),
            "risk_score": c.get("risk_score"),
            "risk_tier": c.get("risk_tier"),
            "risk_breakdown": {
                "ai_generated_points": round(risk_weights["ai"] * p_ai * 100, 2),
                "check_worthiness_points": round(risk_weights["claim"] * p_cl * 100, 2),
                "contradiction_points": round(risk_weights["contradiction"] * contra * 100, 2),
            },
        })
    return out


def build_source_attribution(claims: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sources: Dict[str, Dict[str, Any]] = {}
    for ci, c in enumerate(claims):
        used = set(c.get("evidence_spans_used", []))
        for ev in c.get("evidence", []) or []:
            url = ev.get("url")
            if not url:
                continue
            s = sources.setdefault(url, {"url": url, "domain": urlparse(url).netloc.replace("www.", ""),
                                          "title": ev.get("title"), "max_relevance": 0.0,
                                          "retrieved_for_claims": [], "used_in_verdict_for_claims": []})
            s["max_relevance"] = round(max(s["max_relevance"], float(ev.get("relevance_score") or 0.0)), 3)
            s["retrieved_for_claims"].append(ci)
            if url in used:
                s["used_in_verdict_for_claims"].append(ci)
    return sorted(sources.values(), key=lambda s: (-len(s["used_in_verdict_for_claims"]), -s["max_relevance"]))


_LABELS = {
    "factual_reliability": "Factual reliability", "source_attribution": "Source attribution",
    "coherence": "Coherence", "neutrality": "Neutrality", "readability": "Readability",
    "llm_quality": "LLM reader-simulation",
}


def _describe(name: str, d: Dict[str, Any]) -> str:
    if name == "factual_reliability":
        return (f"{d['supported']} of {d['claims_checked']} verified claims were supported by web evidence, "
                f"{d['contradicted']} contradicted, {d['insufficient_evidence']} inconclusive.")
    if name == "source_attribution":
        return f"{d['attributed_sentences']} of {d['total_sentences']} sentences attribute information to a source or quote."
    if name == "coherence":
        return f"Consecutive sentences have average semantic similarity {d['avg_coherence']:.2f}."
    if name == "neutrality":
        base = f"Emotional intensity of the language is {d['emotional_intensity']:.2f} (lower = more neutral)"
        if "llm_neutrality" in d:
            base += f"; the LLM judge's neutrality rating maps to {d['llm_neutrality']:.2f}"
        return base + "."
    if name == "readability":
        return f"Flesch Reading Ease {d['flesch_reading_ease']} (grade {d['flesch_kincaid_grade']}); target band is ~50-70 for general news."
    if name == "llm_quality":
        return f"The LLM judge rated credibility {d['llm_credibility']}/5 and quality {d['llm_quality']}/5."
    return ""


def explain_credibility(cred: Dict[str, Any]) -> Dict[str, Any]:
    comps = cred["components"]
    available = {k: v for k, v in comps.items() if v["available"]}

    strengths, concerns = [], []
    for name, v in sorted(available.items(), key=lambda kv: kv[1]["value"], reverse=True):
        line = f"{_LABELS[name]} ({v['value']:.2f}): {_describe(name, v['detail'])}"
        if v["value"] >= 0.70:
            strengths.append(line)
        elif v["value"] < 0.50:
            concerns.append(line)

    notes = []
    for name in cred["missing_components"]:
        notes.append(f"{_LABELS[name]} not scored ({comps[name]['detail'].get('reason', 'unavailable')}); remaining weights renormalised.")
    if cred["evidence_coverage"] < 0.75:
        notes.append(f"Score is based on {int(cred['evidence_coverage'] * 100)}% of the scoring weight - treat with extra caution.")

    summary = ("Not enough information to compute a credibility score." if cred["score"] is None else
               f"Credibility {cred['score']:.0f}/100 ({cred['tier']}). {len(strengths)} strong signal(s), {len(concerns)} concern(s).")
    return {"summary": summary, "strengths": strengths, "concerns": concerns, "notes": notes}


def build_explainability(
    sentences: List[Dict[str, Any]],
    claims: List[Dict[str, Any]],
    credibility: Dict[str, Any],
    sentiment_arc: List[float],
    risk_weights: Dict[str, float],
) -> Dict[str, Any]:
    return {
        "highlights": build_highlights(sentences, claims, sentiment_arc),
        "claim_explanations": build_claim_explanations(claims, sentences, risk_weights),
        "sources": build_source_attribution(claims),
        "credibility": explain_credibility(credibility),
        "methods": {
            "ai_detection": "RoBERTa detector (HC3-trained). Label: ai>=0.70, human<0.40, else uncertain. "
                            "Confidence = |p-0.5|*2. Caveat: correlates with formal writing register, not just authorship.",
            "claim_verification": "Tavily web search -> MiniLM relevance ranking -> Groq LLM NLI restricted to retrieved evidence.",
            "risk_score": "0.05*P(AI) + 0.05*P(claim) + 0.90*contradiction, scaled 0-100 (calibrated on a 24-claim benchmark, 83.33% tier accuracy).",
            "credibility_score": "Weighted mean of factual reliability, source attribution, coherence, neutrality, "
                                  "readability, LLM judgement; missing parts renormalised.",
        },
    }
