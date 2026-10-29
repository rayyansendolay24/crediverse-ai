"""
calibration/weights.py
─────────────────────────
Grid search over the risk-score weights (w_ai, w_claim, w_contradiction)
in core/verdict.py, evaluated against calibration/benchmark_claims.py.

Signal collection (calling DeBERTa/RoBERTa/Tavily/Groq for all 24 claims)
is separated from the grid search itself: raw signals are cached to
calibration/cache/raw_signals.json so hundreds of weight combinations can
be scored in milliseconds without re-paying API cost. A cached run from
a previous session is already included - run with --refresh to redo it
from scratch against your own API keys.

Usage:
    python -m calibration.weights                 # use cached signals
    python -m calibration.weights --refresh        # re-run all API calls
    python -m calibration.weights --step 0.02      # finer grid search
"""

from __future__ import annotations

import argparse
import json
import os
import time
from typing import Any, Dict, List, Tuple

from calibration.benchmark_claims import EXPECTED_TIER, LABELED_CLAIMS
from core.verdict import assign_risk_tier, compute_risk_score

CACHE_PATH = os.path.join(os.path.dirname(__file__), "cache", "raw_signals.json")


def collect_raw_signals(claims: List[Dict[str, str]], cache_path: str = CACHE_PATH, force_refresh: bool = False) -> List[Dict[str, Any]]:
    if os.path.exists(cache_path) and not force_refresh:
        print(f"[calibration] loading cached signals from '{cache_path}'")
        with open(cache_path, "r", encoding="utf-8") as f:
            return json.load(f)

    print(f"[calibration] no cache / refresh requested - running models + APIs on {len(claims)} claims")
    from core.claim_detection import score_ai_generated_batch, score_checkworthiness_batch
    from core.evidence import get_evidence_for_claim
    from core.verdict import _FALLBACK_VERDICT, _call_groq_nli

    texts = [c["text"] for c in claims]
    p_claims = score_checkworthiness_batch(texts)
    p_ais = score_ai_generated_batch(texts)

    records: List[Dict[str, Any]] = []
    for i, (item, p_claim, p_ai) in enumerate(zip(claims, p_claims, p_ais), 1):
        print(f"  ({i}/{len(claims)}) {item['text'][:70]}")
        snippets = get_evidence_for_claim(item["text"], max_results=6, top_k=3)
        verdict_data = _call_groq_nli(item["text"], snippets) or dict(_FALLBACK_VERDICT)

        records.append({
            "text": item["text"], "ground_truth": item["ground_truth"], "category": item["category"],
            "p_claim": round(p_claim, 4), "p_ai_generated": round(p_ai, 4),
            "verdict": verdict_data["verdict"], "contradiction_score": round(float(verdict_data["contradiction_score"]), 4),
            "justification": verdict_data.get("justification", ""), "evidence_count": len(snippets),
        })
        if i < len(claims):
            time.sleep(0.5)

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    return records


def evaluate_weights(records: List[Dict[str, Any]], w1: float, w2: float, w3: float) -> Tuple[float, int, Dict[str, Dict[str, int]]]:
    correct = 0
    breakdown = {gt: {"Low": 0, "Medium": 0, "High": 0} for gt in EXPECTED_TIER}
    for r in records:
        score = compute_risk_score(r["p_ai_generated"], r["p_claim"], r["contradiction_score"], w1, w2, w3)
        tier = assign_risk_tier(score, verdict=r.get("verdict"))
        breakdown[r["ground_truth"]][tier] += 1
        if tier == EXPECTED_TIER[r["ground_truth"]]:
            correct += 1
    return round(correct / len(records) * 100, 2), correct, breakdown


def run_grid_search(records: List[Dict[str, Any]], step: float = 0.05) -> List[Dict[str, Any]]:
    results = []
    n = int(round(1.0 / step))
    for i in range(n + 1):
        w1 = round(i * step, 3)
        for j in range(n - i + 1):
            w2 = round(j * step, 3)
            w3 = max(0.0, round(1.0 - w1 - w2, 3))
            acc, correct, breakdown = evaluate_weights(records, w1, w2, w3)
            results.append({"w1_ai": w1, "w2_claim": w2, "w3_contradiction": w3, "accuracy": acc,
                             "correct": correct, "total": len(records), "breakdown": breakdown})
    results.sort(key=lambda x: (x["accuracy"], x["w3_contradiction"]), reverse=True)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate the risk-score weights.")
    parser.add_argument("--refresh", action="store_true", help="Re-run all model/API calls instead of using the cache.")
    parser.add_argument("--step", type=float, default=0.05, help="Grid search step size.")
    args = parser.parse_args()

    records = collect_raw_signals(LABELED_CLAIMS, force_refresh=args.refresh)

    baseline_acc, baseline_correct, baseline_bd = evaluate_weights(records, 0.30, 0.30, 0.40)
    print(f"\nBaseline (0.30, 0.30, 0.40): {baseline_acc}% ({baseline_correct}/{len(records)})")

    results = run_grid_search(records, step=args.step)
    print(f"Explored {len(results)} weight combinations.\n")
    print("Top 5:")
    for rank, res in enumerate(results[:5], 1):
        print(f"  #{rank} w1={res['w1_ai']:.2f} w2={res['w2_claim']:.2f} w3={res['w3_contradiction']:.2f} "
              f"-> {res['accuracy']}% ({res['correct']}/{res['total']})")

    best = results[0]
    print(f"\nBest: w1={best['w1_ai']}, w2={best['w2_claim']}, w3={best['w3_contradiction']}, "
          f"accuracy={best['accuracy']}% (vs baseline {baseline_acc}%)")
    for gt in EXPECTED_TIER:
        print(f"  {gt:12s} -> target {EXPECTED_TIER[gt]:6s} : {best['breakdown'][gt]}")
