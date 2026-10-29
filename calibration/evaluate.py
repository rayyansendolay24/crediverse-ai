"""
calibration/evaluate.py
───────────────────────────
Reproducible evaluation/benchmarking report (Objective 5 / Priority 7).

Three independent sections, each only reporting what can actually be
measured from data that exists:

1. Claim risk-tier classification - precision/recall/F1/confusion
   matrix for the calibrated weights against the 24-claim benchmark
   (calibration/benchmark_claims.py), using the cached signals in
   calibration/cache/raw_signals.json so it runs with no API keys.

2. Reader trust/engagement from the study database (study/database.py)
   - mean ratings, mean model credibility vs mean reader credibility,
   and behavior-event summaries. Reports "no data yet" rather than
   fabricating numbers when study.db has no rows (i.e. before the
   reader study in Priority 5/8 has actually been run).

3. AI-vs-human comparison - explicitly documents that this requires a
   paired human/AI dataset that does not exist in this repo yet (see
   README's "known limitations"), and refuses to report an accuracy
   number for it. Running this section against the 24-claim benchmark
   would be measuring the wrong thing: none of those claims are
   actually AI-written, they're just test claims for the NLI verdict.

Usage:
    python -m calibration.evaluate
    python -m calibration.evaluate --refresh   # re-collect claim signals from live APIs
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

from calibration.benchmark_claims import EXPECTED_TIER, LABELED_CLAIMS
from calibration.weights import collect_raw_signals
from core.verdict import W_AI, W_CLAIM, W_CONTRADICTION, assign_risk_tier, compute_risk_score

TIERS = ["Low", "Medium", "High"]


def _confusion_matrix(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, int]]:
    matrix = {t: {p: 0 for p in labels} for t in labels}
    for t, p in zip(y_true, y_pred):
        matrix[t][p] += 1
    return matrix


def _precision_recall_f1(y_true: List[str], y_pred: List[str], labels: List[str]) -> Dict[str, Dict[str, float]]:
    """Per-class precision/recall/F1 computed directly (no sklearn
    dependency needed for just this), plus macro averages."""
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


def evaluate_claim_tier_classification(force_refresh: bool = False) -> Dict[str, Any]:
    records = collect_raw_signals(LABELED_CLAIMS, force_refresh=force_refresh)

    y_true = [EXPECTED_TIER[r["ground_truth"]] for r in records]
    y_pred = [
        assign_risk_tier(
            compute_risk_score(r["p_ai_generated"], r["p_claim"], r["contradiction_score"],
                                W_AI, W_CLAIM, W_CONTRADICTION),
            verdict=r.get("verdict"),
        )
        for r in records
    ]

    accuracy = round(sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(records) * 100, 2)
    return {
        "n": len(records),
        "weights": {"w_ai": W_AI, "w_claim": W_CLAIM, "w_contradiction": W_CONTRADICTION},
        "accuracy_pct": accuracy,
        "confusion_matrix": _confusion_matrix(y_true, y_pred, TIERS),
        "per_class": _precision_recall_f1(y_true, y_pred, TIERS),
    }


def evaluate_reader_study() -> Dict[str, Any]:
    """Reads study.db directly. Returns an explicit 'no data yet' state
    instead of fabricating any statistic when the study hasn't run."""
    try:
        from study.database import get_all_ratings, init_db
    except ImportError:
        return {"available": False, "reason": "study.database not importable."}

    init_db()
    ratings = get_all_ratings()
    if not ratings:
        return {"available": False, "reason": "No reader-study ratings recorded in study.db yet - "
                                                "run the survey (Priority 5/8) before trusting this section."}

    dims = ["credibility", "clarity", "neutrality", "complexity", "coherence", "quality"]
    by_label: Dict[str, List[Dict[str, Any]]] = {}
    for r in ratings:
        by_label.setdefault(r["shown_label"], []).append(r)

    summary = {}
    for label, rows in by_label.items():
        summary[label] = {
            "n": len(rows),
            "mean_ratings": {d: round(sum(r[d] for r in rows) / len(rows), 2) for d in dims},
        }
    return {"available": True, "n_ratings": len(ratings), "by_shown_label": summary}


def evaluate_ai_human_comparison() -> Dict[str, Any]:
    """Delegates to calibration.ai_detector_eval, which scores the
    RoBERTa AI-text detector against whatever paired human/AI articles
    are currently loaded into study.db's curated set (see
    study.seed_articles / study_articles.zip for a starter set of
    same-story pairs). Returns an explicit 'not available' reason
    instead of a number if too few articles of each origin are loaded -
    this never falls back to the 24-claim benchmark, which is NOT an
    AI-vs-human dataset (none of those claims are actually AI-written,
    they're true/false/unverifiable claims used for the NLI risk
    formula - a different axis entirely)."""
    from calibration.ai_detector_eval import evaluate_ai_detector

    return evaluate_ai_detector()


def run_full_evaluation(force_refresh: bool = False) -> Dict[str, Any]:
    return {
        "claim_tier_classification": evaluate_claim_tier_classification(force_refresh=force_refresh),
        "reader_study": evaluate_reader_study(),
        "ai_vs_human_detection": evaluate_ai_human_comparison(),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the full evaluation/benchmarking report.")
    parser.add_argument("--refresh", action="store_true", help="Re-run claim-signal collection against live APIs.")
    parser.add_argument("--out", default=None, help="Optional path to also write the report as JSON.")
    args = parser.parse_args()

    report = run_full_evaluation(force_refresh=args.refresh)

    print("\n" + "=" * 70)
    print("CLAIM RISK-TIER CLASSIFICATION")
    print("=" * 70)
    ctc = report["claim_tier_classification"]
    print(f"n={ctc['n']}  weights={ctc['weights']}  accuracy={ctc['accuracy_pct']}%")
    print("confusion matrix (rows=true, cols=predicted):")
    for t in TIERS:
        print(f"  {t:8s} {ctc['confusion_matrix'][t]}")
    print("per-class precision/recall/F1:")
    for label, m in ctc["per_class"].items():
        print(f"  {label:10s} {m}")

    print("\n" + "=" * 70)
    print("READER STUDY")
    print("=" * 70)
    rs = report["reader_study"]
    print(json.dumps(rs, indent=2))

    print("\n" + "=" * 70)
    print("AI vs HUMAN DETECTION")
    print("=" * 70)
    print(json.dumps(report["ai_vs_human_detection"], indent=2))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\nWrote full report to {args.out}")
