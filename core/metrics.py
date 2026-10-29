"""
core/metrics.py
────────────────
Article-level readability and coherence metrics. Reconstructed for this
project (the original article_metrics.py wasn't available to copy from
verbatim) but keeps the same output shape the rest of the codebase expects:

    {"readability": {...}, "coherence": {"avg_coherence": float, ...}}

Readability uses textstat (Flesch Reading Ease + Flesch-Kincaid Grade,
plus a few extra indices that are useful for the report but optional).
Coherence uses the shared embedding model (core/embeddings.py) - cosine
similarity between consecutive sentence embeddings, averaged across the
article. Higher = ideas flow together more smoothly; too high can also
mean repetitive / padded text, worth noting in the report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import textstat

from core.embeddings import encode


def compute_readability(text: str, min_words: int = 30) -> Dict[str, Optional[float]]:
    keys = [
        "flesch_reading_ease", "flesch_kincaid_grade", "gunning_fog",
        "smog_index", "coleman_liau_index", "automated_readability_index",
    ]
    if not text or len(text.split()) < min_words:
        return {k: None for k in keys}

    fns = {
        "flesch_reading_ease": textstat.flesch_reading_ease,
        "flesch_kincaid_grade": textstat.flesch_kincaid_grade,
        "gunning_fog": textstat.gunning_fog,
        "smog_index": textstat.smog_index,
        "coleman_liau_index": textstat.coleman_liau_index,
        "automated_readability_index": textstat.automated_readability_index,
    }
    return {k: round(float(fn(text)), 2) for k, fn in fns.items()}


def compute_coherence(sentences: List[str]) -> Dict[str, Any]:
    if len(sentences) < 2:
        return {"avg_coherence": None, "per_pair": [], "reason": "fewer than 2 sentences"}

    embeddings = np.asarray(encode(sentences, normalize=True))
    sims = [
        float(np.dot(embeddings[i], embeddings[i + 1]))
        for i in range(len(embeddings) - 1)
    ]
    return {
        "avg_coherence": round(sum(sims) / len(sims), 4),
        "min_coherence": round(min(sims), 4),
        "max_coherence": round(max(sims), 4),
        "per_pair": [round(s, 4) for s in sims],
    }


def compute_article_metrics(text: str, sentences: Optional[List[str]] = None) -> Dict[str, Any]:
    """sentences: pass the spaCy-segmented sentences from claim_detection to
    keep segmentation consistent across the pipeline; falls back to a naive
    split on '. ' if not given (only used when calling this module standalone).
    """
    if sentences is None:
        sentences = [s.strip() for s in text.replace("\n", " ").split(". ") if s.strip()]

    return {
        "readability": compute_readability(text),
        "coherence": compute_coherence(sentences),
    }


if __name__ == "__main__":
    import json

    SAMPLE = (
        "The sun rose over the quiet village. Birds sang in the trees as "
        "farmers began their morning work. Meanwhile, in the capital, "
        "officials debated a new economic policy."
    )
    print(json.dumps(compute_article_metrics(SAMPLE), indent=2))
