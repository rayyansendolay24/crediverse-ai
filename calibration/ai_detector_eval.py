"""
calibration/ai_detector_eval.py
─────────────────────────────────────
Real AI-vs-human detector accuracy, now that a paired dataset exists
(study_articles/human/*.txt + study_articles/ai/*.txt, loaded into
study.db's `articles` table via `study.seed_articles`, each row
carrying its real, known `actual_origin`).

This was explicitly refused in calibration/evaluate.py before a paired
dataset existed - reporting a number against the 24-claim benchmark
would have measured the wrong thing (those are true/false/unverifiable
claims, none of them actually AI-written). This module measures the
right thing: Hello-SimpleAI/chatgpt-detector-roberta's actual ability
to tell a human-written news article from an AI-rewritten one, on
article-level text, with a known ground-truth label for every row.

Needs the real models (spaCy + the AI detector), so - unlike the
claim-tier section - this does NOT run from a cache; it loads whatever
articles are currently in study.db's curated set and scores them live.
Run this on the machine with the model folder + dependencies installed,
not in a CI/test environment (see tests/test_ai_detector_eval.py for
the mocked version that checks only the metric math).

Usage:
    python -m calibration.ai_detector_eval
    python -m calibration.ai_detector_eval --threshold 0.6
"""

from __future__ import annotations

import argparse
import json
from typing import Any, Dict, List

LABELS = ["human", "ai"]


def _confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def _precision_recall_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        out[label] = {"precision": round(precision, 3), "recall": round(recall, 3),
                       "f1": round(f1, 3), "support": tp + fn}
    out["macro_avg"] = {
        "precision": round(sum(v["precision"] for v in out.values()) / len(labels), 3),
        "recall": round(sum(v["recall"] for v in out.values()) / len(labels), 3),
        "f1": round(sum(v["f1"] for v in out.values()) / len(labels), 3),
        "support": sum(out[l]["support"] for l in labels),
    }
    return out


def _mean(values: List[float]) -> float:
    return round(sum(values) / len(values), 4) if values else 0.0


def evaluate_ai_detector(threshold: float = 0.5, articles: List[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Scores every curated-study article's article-level p_ai_generated
    against its real actual_origin. `articles` is injectable for tests
    (mocked scorer); omit it to read from study.db and run the live model.

    Predicts "ai" when the article-level p_ai_generated >= threshold,
    else "human" - this is a binary read of the same 3-way signal
    `core.claim_detection.summarize_ai_signal` already computes.
    """
    if articles is None:
        from study.database import init_db, list_articles

        init_db()
        articles = list_articles()

    human_count = sum(1 for a in articles if a["actual_origin"] == "human")
    ai_count = sum(1 for a in articles if a["actual_origin"] == "ai")
    if human_count < 2 or ai_count < 2:
        return {
            "available": False,
            "reason": (
                f"Need at least 2 human-written and 2 AI-written articles in the curated "
                f"study set to report a meaningful accuracy figure (have {human_count} human, "
                f"{ai_count} ai). Run `python -m study.seed_articles <folder>` to load more - "
                f"see study_articles.zip for a starter set of same-story pairs."
            ),
        }

    from core.claim_detection import analyze_article_sentences, summarize_ai_signal

    y_true: List[str] = []
    y_pred: List[str] = []
    per_article: List[Dict[str, Any]] = []
    p_ai_by_origin: Dict[str, List[float]] = {"human": [], "ai": []}

    for article in articles:
        sentences = analyze_article_sentences(article["body_text"])
        signal = summarize_ai_signal(sentences)
        p_ai = signal["article_p_ai"] if signal["article_p_ai"] is not None else 0.0
        predicted = "ai" if p_ai >= threshold else "human"

        y_true.append(article["actual_origin"])
        y_pred.append(predicted)
        p_ai_by_origin[article["actual_origin"]].append(p_ai)
        per_article.append({
            "article_id": article["id"],
            "title": article.get("title"),
            "topic": article.get("topic"),
            "actual_origin": article["actual_origin"],
            "predicted": predicted,
            "article_p_ai": p_ai,
            "correct": predicted == article["actual_origin"],
        })

    accuracy = round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true) * 100, 2)

    return {
        "available": True,
        "n": len(articles),
        "threshold": threshold,
        "accuracy_pct": accuracy,
        "confusion_matrix": _confusion_matrix(y_true, y_pred, LABELS),
        "per_class": _precision_recall_f1(y_true, y_pred, LABELS),
        "mean_p_ai_generated_by_actual_origin": {
            "human": _mean(p_ai_by_origin["human"]),
            "ai": _mean(p_ai_by_origin["ai"]),
        },
        "per_article": per_article,
        "caveat": (
            "N is small (one data point per article, not per sentence) - treat this as a "
            "pilot measurement, not a statistically robust accuracy figure. Expand "
            "study_articles/ with more same-story human/AI pairs across topics for a "
            "sturdier result."
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the AI-text detector against the curated paired dataset.")
    parser.add_argument("--threshold", type=float, default=0.5, help="p_ai_generated cutoff for predicting 'ai' (default 0.5)")
    parser.add_argument("--out", default=None, help="Optional path to also write the report as JSON")
    args = parser.parse_args()

    report = evaluate_ai_detector(threshold=args.threshold)

    if not report["available"]:
        print(f"\nNot available: {report['reason']}")
    else:
        print("\n" + "=" * 70)
        print("AI-TEXT DETECTOR ACCURACY (article-level, paired human/AI dataset)")
        print("=" * 70)
        print(f"n={report['n']}  threshold={report['threshold']}  accuracy={report['accuracy_pct']}%")
        print("confusion matrix (rows=true, cols=predicted):")
        for label in LABELS:
            print(f"  {label:8s} {report['confusion_matrix'][label]}")
        print("per-class precision/recall/F1:")
        for label, m in report["per_class"].items():
            print(f"  {label:10s} {m}")
        print(f"\nmean p_ai_generated by actual origin: {report['mean_p_ai_generated_by_actual_origin']}")
        print(f"\n{report['caveat']}")
        print("\nper-article detail:")
        for a in report["per_article"]:
            mark = "OK " if a["correct"] else "ERR"
            print(f"  [{mark}] {a['actual_origin']:6s} -> predicted {a['predicted']:6s} "
                  f"(p_ai={a['article_p_ai']:.3f})  topic={a['topic']}  {a['title']}")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full report to {args.out}")
