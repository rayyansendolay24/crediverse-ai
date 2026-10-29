"""
analysis/credibility.py
──────────────────────────
Composite, explainable credibility score (0-100, higher = more credible).

Note the direction flips from core.verdict.risk_score (0-100, higher =
riskier) - readers think in terms of "how much can I trust this", so this
score points the other way on purpose.

Six weighted components, each normalised to 0-1:
    factual_reliability  1 - mean claim risk (web-evidence NLI verdicts)
    source_attribution    share of sentences that cite a source or quote
    coherence             consecutive-sentence embedding similarity
    neutrality            low emotional intensity, blended with LLM rating
    readability           Flesch Reading Ease, target band ~50-70
    llm_quality           LLM reader-simulation: mean(credibility, quality)

Any component that can't be computed (no claims verified, Groq offline,
etc) is dropped and the rest are renormalised. `evidence_coverage` reports
how much of the total weight was actually available, so a score built
from thin evidence is visibly flagged.

DEFAULT_WEIGHTS below are hand-set starting values. For Objective 5, fit
these against real reader credibility ratings from the user study
(e.g. linear regression of human rating on the six component values) and
report the fitted weights alongside these defaults.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_WEIGHTS: Dict[str, float] = {
    "factual_reliability": 0.35,
    "source_attribution": 0.15,
    "coherence": 0.15,
    "neutrality": 0.15,
    "readability": 0.10,
    "llm_quality": 0.10,
}

_ATTRIB_RE = re.compile(
    r"\b(according to|said|says|told|reported|reports|announced|confirmed|stated|"
    r"noted|argued|claimed|cited|as per|spokes(?:man|woman|person)|"
    r"(?:study|report|survey|data|analysis|statement|filing) (?:by|from))\b"
    r"|[\"\u201c\u201d]",
    re.I,
)


def _clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _factual(claims: List[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    scored = [c for c in claims if "risk_score" in c]
    if not scored:
        return None, {"reason": "no check-worthy claims were verified"}
    mean_risk = sum(c["risk_score"] for c in scored) / len(scored)
    counts = Counter(c.get("verdict") for c in scored)
    return _clamp(1.0 - mean_risk / 100.0), {
        "claims_checked": len(scored),
        "supported": counts.get("supported", 0),
        "contradicted": counts.get("contradicted", 0),
        "insufficient_evidence": counts.get("insufficient_evidence", 0),
        "mean_claim_risk": round(mean_risk, 2),
    }


def _attribution(sentences: List[str]) -> Tuple[Optional[float], Dict[str, Any]]:
    if not sentences:
        return None, {"reason": "no sentences"}
    hits = sum(1 for s in sentences if _ATTRIB_RE.search(s))
    ratio = hits / len(sentences)
    return _clamp(ratio / 0.35), {"attributed_sentences": hits, "total_sentences": len(sentences), "ratio": round(ratio, 3)}


def _coherence(article_metrics: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    coh = (article_metrics.get("coherence") or {}).get("avg_coherence")
    if coh is None:
        return None, {"reason": "fewer than 2 sentences"}
    return _clamp((coh - 0.10) / 0.40), {"avg_coherence": coh}


def _neutrality(engagement: Dict[str, Any], llm: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    intensity = engagement.get("sentiment_intensity")
    if intensity is None:
        return None, {"reason": "no sentiment data"}
    heuristic = 1.0 - _clamp(intensity / 0.5)
    detail: Dict[str, Any] = {"emotional_intensity": intensity, "heuristic": round(heuristic, 3)}
    value = heuristic
    if llm and llm.get("ratings", {}).get("neutrality") is not None:
        llm_n = (llm["ratings"]["neutrality"] - 1) / 4.0
        detail["llm_neutrality"] = round(llm_n, 3)
        value = 0.5 * heuristic + 0.5 * llm_n
    return _clamp(value), detail


def _readability(article_metrics: Dict[str, Any]) -> Tuple[Optional[float], Dict[str, Any]]:
    r = article_metrics.get("readability") or {}
    fre = r.get("flesch_reading_ease")
    if fre is None:
        return None, {"reason": "text too short for readability formulas"}
    return _clamp(1.0 - abs(fre - 60.0) / 40.0), {"flesch_reading_ease": fre, "flesch_kincaid_grade": r.get("flesch_kincaid_grade")}


def _llm_quality(llm: Optional[Dict[str, Any]]) -> Tuple[Optional[float], Dict[str, Any]]:
    if not llm or not llm.get("ratings"):
        return None, {"reason": "LLM judge unavailable"}
    rt = llm["ratings"]
    m = (rt["credibility"] + rt["quality"]) / 2.0
    return _clamp((m - 1) / 4.0), {"llm_credibility": rt["credibility"], "llm_quality": rt["quality"]}


def compute_credibility(
    claims: List[Dict[str, Any]],
    article_metrics: Dict[str, Any],
    engagement: Dict[str, Any],
    sentences: List[str],
    llm_judge: Optional[Dict[str, Any]] = None,
    weights: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    w = dict(DEFAULT_WEIGHTS)
    if weights:
        w.update(weights)

    raw = {
        "factual_reliability": _factual(claims),
        "source_attribution": _attribution(sentences),
        "coherence": _coherence(article_metrics),
        "neutrality": _neutrality(engagement, llm_judge),
        "readability": _readability(article_metrics),
        "llm_quality": _llm_quality(llm_judge),
    }

    avail_w = sum(w[k] for k, (v, _) in raw.items() if v is not None)
    total_w = sum(w.values())

    components: Dict[str, Any] = {}
    score: Optional[float] = 0.0 if avail_w > 0 else None
    for name, (value, detail) in raw.items():
        entry: Dict[str, Any] = {"available": value is not None, "value": round(value, 4) if value is not None else None,
                                  "base_weight": w[name], "detail": detail}
        if value is not None and avail_w > 0:
            w_used = w[name] / avail_w
            contribution = value * w_used * 100.0
            entry["weight_used"] = round(w_used, 4)
            entry["contribution"] = round(contribution, 2)
            score += contribution
        components[name] = entry

    tier = None if score is None else "High" if score >= 70 else "Moderate" if score >= 45 else "Low"

    return {
        "score": round(score, 2) if score is not None else None,
        "tier": tier,
        "evidence_coverage": round(avail_w / total_w, 2) if total_w else 0.0,
        "components": components,
        "missing_components": [k for k, (v, _) in raw.items() if v is None],
        "weights": w,
    }
