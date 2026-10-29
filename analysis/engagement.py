"""
analysis/engagement.py
────────────────────────
Text-level engagement proxy: sentiment arc, lexical diversity, sentence
rhythm, and headline heuristics. No heavy models - VADER + regex only.

IMPORTANT for the report: this is a *predicted* engagement proxy computed
purely from the text. Measured engagement (dwell time, scroll depth,
clicks) comes from the reader study. Objective 5 should correlate this
score against that measured data, not treat it as ground truth on its own.
"""

from __future__ import annotations

import re
import statistics
from typing import Any, Dict, List, Optional

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

_vader = SentimentIntensityAnalyzer()
_TOKEN_RE = re.compile(r"[A-Za-z']+")

SENSATIONAL_WORDS = {
    "shocking", "unbelievable", "breaking", "secret", "exposed", "destroyed",
    "outrage", "slams", "bombshell", "miracle", "terrifying", "horrifying",
    "insane", "crazy", "stunning", "revealed", "epic", "disaster", "scandal",
}


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def sentence_sentiments(sentences: List[str]) -> List[float]:
    return [round(_vader.polarity_scores(s)["compound"], 4) for s in sentences]


def _mattr(tokens: List[str], window: int = 50) -> float:
    """Moving-average type-token ratio - more length-robust than plain TTR."""
    if not tokens:
        return 0.0
    if len(tokens) <= window:
        return len(set(tokens)) / len(tokens)
    ratios = [len(set(tokens[i:i + window])) / window for i in range(0, len(tokens) - window + 1)]
    return sum(ratios) / len(ratios)


def analyze_headline(headline: Optional[str]) -> Optional[Dict[str, Any]]:
    if not headline or not headline.strip():
        return None
    h = headline.strip()
    words = h.split()
    lower = h.lower()
    sens_hits = sum(1 for w in SENSATIONAL_WORDS if w in lower)
    caps_words = sum(1 for w in words if len(w) > 2 and w.isupper())
    n = len(words)

    length_fit = n / 6 if n < 6 else 1.0 if n <= 14 else max(0.3, 1 - (n - 14) * 0.08)
    penalty = 0.15 * sens_hits + 0.20 * caps_words + 0.10 * h.count("!")
    score = _clamp01(length_fit * (1 - penalty)) * 100

    return {
        "word_count": n,
        "has_number": bool(re.search(r"\d", h)),
        "sensational_word_hits": sens_hits,
        "all_caps_words": caps_words,
        "exclamation_count": h.count("!"),
        "sentiment": round(_vader.polarity_scores(h)["compound"], 4),
        "headline_score": round(score, 1),
    }


def compute_engagement_features(
    text: str,
    headline: Optional[str] = None,
    sentences: Optional[List[str]] = None,
) -> Dict[str, Any]:
    sentences = sentences or [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    n_words = len(tokens)

    sent_lens = [len(_TOKEN_RE.findall(s)) for s in sentences] or [0]
    avg_len = statistics.mean(sent_lens)
    std_len = statistics.pstdev(sent_lens) if len(sent_lens) > 1 else 0.0
    cv = std_len / avg_len if avg_len else 0.0

    arc = sentence_sentiments(sentences)
    intensity = statistics.mean(abs(x) for x in arc) if arc else 0.0
    mattr = _mattr(tokens)
    headline_feats = analyze_headline(headline)

    length_fit = n_words / 300 if n_words < 300 else 1.0 if n_words <= 800 else max(0.5, 1 - (n_words - 800) / 2400)
    rhythm_fit = 1 - _clamp01(abs(cv - 0.55) / 0.55)
    lexical_fit = _clamp01(mattr / 0.75)
    affect_fit = 1 - _clamp01(abs(intensity - 0.25) / 0.25)

    parts = {"length": (length_fit, 0.25), "rhythm": (rhythm_fit, 0.20),
             "lexical": (lexical_fit, 0.20), "affect": (affect_fit, 0.15)}
    if headline_feats:
        parts["headline"] = (headline_feats["headline_score"] / 100, 0.20)

    total_w = sum(w for _, w in parts.values())
    engagement_score = 100 * sum(v * w for v, w in parts.values()) / total_w

    return {
        "word_count": n_words,
        "sentence_count": len(sentences),
        "avg_sentence_length": round(avg_len, 2),
        "sentence_length_cv": round(cv, 3),
        "lexical_diversity_mattr": round(mattr, 3),
        "sentiment_intensity": round(intensity, 3),
        "sentiment_arc": arc,
        "headline": headline_feats,
        "engagement_components": {k: round(v * 100, 1) for k, (v, _) in parts.items()},
        "engagement_score": round(engagement_score, 1),
        "note": "Predicted proxy from text features - validate against measured reader behaviour.",
    }
