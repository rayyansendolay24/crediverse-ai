"""
core/evidence.py
──────────────────
Web evidence retrieval + semantic relevance ranking, for use before NLI
verdict scoring (core/verdict.py).

Rewritten from the original evidence_retrieval.py:
  * Tavily client is lazily created (module import no longer requires
    TAVILY_API_KEY or a network check just to load this file).
  * Uses the shared embedder in core/embeddings.py instead of loading its
    own SentenceTransformer instance, so evidence ranking and article
    coherence (core/metrics.py) share one model in memory.
  * Same overall design as the original: Tavily for page-level content
    (better context than a plain search snippet), cosine similarity for
    local relevance ranking (Tavily's own ranking optimises query-document
    relevance, not claim-evidence alignment).
"""

from __future__ import annotations

import os
import time
import warnings
from functools import lru_cache
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

from core.embeddings import encode

load_dotenv()


@lru_cache(maxsize=1)
def _get_tavily_client():
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        warnings.warn(
            "TAVILY_API_KEY not set - evidence search will return empty results. "
            "Add it to your .env file."
        )
        return None
    from tavily import TavilyClient

    return TavilyClient(api_key=api_key)


def search_claim(claim_text: str, max_results: int = 8) -> List[Dict]:
    """Query Tavily for pages related to claim_text. Never raises - returns
    [] on any failure (missing key, network error, zero results)."""
    client = _get_tavily_client()
    if client is None or not claim_text or not claim_text.strip():
        return []

    try:
        response = client.search(
            query=claim_text,
            max_results=max_results,
            search_depth="basic",
            include_answer=False,
        )
    except Exception as exc:  # noqa: BLE001 - network/auth/rate-limit, all non-fatal
        print(f"  [evidence] Tavily search failed: {type(exc).__name__}: {exc}")
        return []

    snippets = []
    for r in response.get("results", []):
        content = (r.get("content") or "").strip()
        if content:
            snippets.append({"title": r.get("title", ""), "url": r.get("url", ""), "content": content})
    return snippets


def rank_by_relevance(claim_text: str, snippets: List[Dict], top_k: int = 3) -> List[Dict]:
    """Return the top_k snippets most semantically similar to claim_text."""
    if not snippets:
        return []
    top_k = min(top_k, len(snippets))

    texts = [claim_text] + [s["content"] for s in snippets]
    embeddings = np.asarray(encode(texts, normalize=True))
    claim_emb, snippet_embs = embeddings[0], embeddings[1:]

    similarities = snippet_embs @ claim_emb
    top_indices = np.argsort(similarities)[::-1][:top_k]

    ranked = []
    for idx in top_indices:
        entry = dict(snippets[idx])
        entry["relevance_score"] = round(float(similarities[idx]), 4)
        ranked.append(entry)
    return ranked


def get_evidence_for_claim(claim_text: str, max_results: int = 8, top_k: int = 3) -> List[Dict]:
    snippets = search_claim(claim_text, max_results=max_results)
    return rank_by_relevance(claim_text, snippets, top_k=top_k) if snippets else []


def attach_evidence_to_claims(claims: List[Dict], max_results: int = 8, top_k: int = 3) -> List[Dict]:
    """Mutates each claim dict in place, adding an 'evidence' key."""
    if not claims:
        return []

    total = len(claims)
    for i, claim in enumerate(claims, 1):
        text = claim.get("text", "")
        print(f"  [evidence] ({i}/{total}) searching: \"{text[:80]}\"")
        claim["evidence"] = get_evidence_for_claim(text, max_results=max_results, top_k=top_k)
        if i < total:
            time.sleep(0.5)  # stay under Tavily's rate limit
    return claims


if __name__ == "__main__":
    import json

    result = get_evidence_for_claim(
        "The Eiffel Tower was completed in 1889 and stands 330 metres tall.",
        max_results=6,
        top_k=3,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))
