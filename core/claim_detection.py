"""
core/claim_detection.py
────────────────────────
Sentence-level scoring: splits an article into sentences and scores every
one of them for (a) check-worthiness and (b) probability of being
AI-generated text.

Change vs. the original claim_pipeline.py / sentence_analysis.py split:
  * Models are lazy-loaded (functools.lru_cache) instead of loaded at
    import time. Importing this module no longer pulls DeBERTa + RoBERTa +
    spaCy into memory unless a scoring function is actually called - useful
    for tests and for any script that only needs the dataclass shapes.
  * The two original files (extract_and_score_claims + analyze_sentences)
    are merged into one: every sentence is always scored and returned,
    with `is_claim` as a boolean flag, rather than silently discarding
    sub-threshold sentences. XAI highlighting needs the full sentence list.

Model choices (unchanged from the original):
  * Check-worthiness -> fine-tuned DeBERTa-v3-small, local folder.
    NOT committed to git (see .gitignore) - copy it in manually.
  * AI-text detector -> Hello-SimpleAI/chatgpt-detector-roberta (HF hub).
    Trained on ChatGPT-vs-human short-form QA (HC3), not news. Treat its
    output as a noisy signal, not ground truth - see AI_DETECTOR_CAVEAT.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, List

DEBERTA_MODEL_PATH = "deberta_claim_classifier_fp32_v2"
AI_DETECTOR_MODEL_ID = "Hello-SimpleAI/chatgpt-detector-roberta"

# Thresholds for turning p_ai_generated into a 3-way label.
AI_HIGH_THRESHOLD = 0.70   # >= this -> "ai"
AI_LOW_THRESHOLD = 0.40    # <  this -> "human"; in between -> "uncertain"

AI_DETECTOR_CAVEAT = (
    "The detector was trained on ChatGPT-vs-human Q&A text (HC3), not news "
    "articles. In our own calibration, plain factual/encyclopedic sentences "
    "scored consistently high p_ai_generated (~0.97 mean) regardless of who "
    "wrote them - it appears to respond to formal writing register, not "
    "authorship. Report this as a measured limitation rather than treating "
    "the score as proof of AI authorship."
)


@lru_cache(maxsize=1)
def _load_spacy():
    import spacy

    print("[claim_detection] loading spaCy 'en_core_web_sm' ...")
    return spacy.load("en_core_web_sm")


@lru_cache(maxsize=1)
def _load_checkworthiness_model():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[claim_detection] loading check-worthiness model from '{DEBERTA_MODEL_PATH}' ...")
    tokenizer = AutoTokenizer.from_pretrained(DEBERTA_MODEL_PATH)
    model = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL_PATH).to(device)
    model.eval()
    return tokenizer, model, device


@lru_cache(maxsize=1)
def _load_ai_detector():
    import torch
    from transformers import pipeline

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[claim_detection] loading AI-text detector '{AI_DETECTOR_MODEL_ID}' ...")
    return pipeline("text-classification", model=AI_DETECTOR_MODEL_ID, top_k=None, device=device)


def score_checkworthiness_batch(texts: List[str]) -> List[float]:
    """P(check-worthy claim) for each text, one batched forward pass."""
    if not texts:
        return []
    import torch

    tokenizer, model, device = _load_checkworthiness_model()
    encodings = tokenizer(texts, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
    with torch.no_grad():
        probs = torch.softmax(model(**encodings).logits, dim=-1)[:, 1]
    return probs.cpu().tolist()


def score_ai_generated_batch(texts: List[str]) -> List[float]:
    """P(AI-generated) for each text using the HF pipeline's batching."""
    if not texts:
        return []
    ai_pipe = _load_ai_detector()
    raw = ai_pipe(texts, batch_size=len(texts))
    scores = []
    for label_scores in raw:
        score = next((d["score"] for d in label_scores if d["label"] == "ChatGPT"), 0.0)
        scores.append(score)
    return scores


def _ai_label(p: float) -> str:
    if p >= AI_HIGH_THRESHOLD:
        return "ai"
    if p < AI_LOW_THRESHOLD:
        return "human"
    return "uncertain"


def analyze_article_sentences(article_text: str, claim_threshold: float = 0.65) -> List[Dict[str, Any]]:
    """Split into sentences and score every one for check-worthiness + AI probability.

    Returns one dict per non-blank sentence, in document order, with keys:
    index, text, char_start, char_end, p_claim, p_ai_generated, is_claim,
    ai_label, ai_confidence.
    """
    if not article_text or not article_text.strip():
        raise ValueError("article_text is empty or whitespace-only.")

    nlp = _load_spacy()
    doc = nlp(article_text)
    raw = [(s.text, s.start_char, s.end_char) for s in doc.sents if s.text.strip()]
    if not raw:
        return []

    texts = [t for t, _, _ in raw]
    p_claims = score_checkworthiness_batch(texts)
    p_ais = score_ai_generated_batch(texts)

    out: List[Dict[str, Any]] = []
    for i, ((text, start, end), pc, pa) in enumerate(zip(raw, p_claims, p_ais)):
        out.append(
            {
                "index": i,
                "text": text,
                "char_start": start,
                "char_end": end,
                "p_claim": round(pc, 4),
                "p_ai_generated": round(pa, 4),
                "is_claim": pc >= claim_threshold,
                "ai_label": _ai_label(pa),
                "ai_confidence": round(abs(pa - 0.5) * 2, 4),
            }
        )
    return out


def summarize_ai_signal(sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Article-level AI-likelihood, weighted by sentence character length."""
    if not sentences:
        return {
            "article_p_ai": None,
            "label": None,
            "confidence": None,
            "ai_sentence_ratio": None,
            "sentence_counts": {"ai": 0, "uncertain": 0, "human": 0},
            "caveat": AI_DETECTOR_CAVEAT,
        }

    weights = [max(1, s["char_end"] - s["char_start"]) for s in sentences]
    total = sum(weights)
    p = sum(s["p_ai_generated"] * w for s, w in zip(sentences, weights)) / total
    ai_chars = sum(w for s, w in zip(sentences, weights) if s["p_ai_generated"] >= 0.5)

    counts = {"ai": 0, "uncertain": 0, "human": 0}
    for s in sentences:
        counts[s["ai_label"]] += 1

    label_text = {"ai": "Likely AI-generated", "uncertain": "Mixed / uncertain", "human": "Likely human-written"}
    return {
        "article_p_ai": round(p, 4),
        "label": label_text[_ai_label(p)],
        "confidence": round(abs(p - 0.5) * 2, 4),
        "ai_sentence_ratio": round(ai_chars / total, 4),
        "sentence_counts": counts,
        "caveat": AI_DETECTOR_CAVEAT,
    }


def claims_from_sentences(sentences: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract just the check-worthy sentences, reshaped for evidence.py / verdict.py."""
    return [
        {
            "text": s["text"],
            "p_claim": s["p_claim"],
            "p_ai_generated": s["p_ai_generated"],
            "char_start": s["char_start"],
            "char_end": s["char_end"],
            "sentence_index": s["index"],
        }
        for s in sentences
        if s["is_claim"]
    ]


if __name__ == "__main__":
    import json

    SAMPLE = (
        "In my opinion, world leaders are completely detached from reality. "
        "According to NASA, the global average surface temperature in 2024 was "
        "1.29 degrees C above the pre-industrial baseline."
    )
    sentences = analyze_article_sentences(SAMPLE, claim_threshold=0.6)
    print(json.dumps(sentences, indent=2))
    print(json.dumps(summarize_ai_signal(sentences), indent=2))
