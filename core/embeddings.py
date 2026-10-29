"""
core/embeddings.py
───────────────────
One shared sentence-embedding model for the whole project.

The original repo loaded a SentenceTransformer inside evidence_retrieval.py
only. This project also needs sentence embeddings for coherence scoring
(core/metrics.py), so the model is centralised here and lazily loaded once,
the first time either module actually needs it. That avoids pulling in
~120MB of weights just to import a module that doesn't use them yet (useful
when unit-testing core/metrics.py without touching the network stage).
"""

from __future__ import annotations

from functools import lru_cache

MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=1)
def get_embedder():
    """Return the shared SentenceTransformer instance, loading it on first use."""
    from sentence_transformers import SentenceTransformer

    print(f"[embeddings] loading '{MODEL_NAME}' ...")
    model = SentenceTransformer(MODEL_NAME)
    print("[embeddings] ready")
    return model


def encode(texts: list[str], normalize: bool = True):
    """Encode a batch of texts in one forward pass."""
    if not texts:
        return []
    model = get_embedder()
    return model.encode(texts, normalize_embeddings=normalize, show_progress_bar=False)
