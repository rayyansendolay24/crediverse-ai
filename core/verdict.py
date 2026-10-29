"""
core/verdict.py
─────────────────
NLI verdict (does the evidence support/contradict the claim?) + the
deterministic risk-score formula that turns three signals into one number.

Rewritten from the original verdict_scoring.py. Behaviour is the same;
the Groq client is now lazily created (lru_cache) rather than built at
import time, so this module can be imported for its pure functions
(compute_risk_score, assign_risk_tier) without needing GROQ_API_KEY set.

Weights: calibrated via calibration/weights.py on a 24-claim benchmark.
Re-running that grid search against the actual cached signals gives
83.33% tier accuracy for (0.05, 0.05, 0.90) vs 66.67% for the untuned
(0.30, 0.30, 0.40) baseline - note this corrects the 75% figure that was
in the original code's comment, which didn't match its own cached data.
"""

from __future__ import annotations

import json
import os
import time
import warnings
from functools import lru_cache
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from core.rate_limit import (
    RATE_LIMIT_MAX_RETRIES,
    is_rate_limit_error,
    wait_for_rate_limit,
)

load_dotenv()

GROQ_MODEL = "openai/gpt-oss-120b"
MAX_RETRIES = 2

# Calibrated risk weights - see calibration/weights.py for the grid search.
W_AI = 0.05
W_CLAIM = 0.05
W_CONTRADICTION = 0.90

_SYSTEM_PROMPT = (
    "You are a precise fact-checking NLI system. Respond with a single valid "
    "JSON object and nothing else - no markdown, no commentary."
)

_FALLBACK_VERDICT: Dict[str, Any] = {
    "verdict": "insufficient_evidence",
    "contradiction_score": 0.5,
    "justification": "Automated NLI scoring failed after retries - manual review recommended.",
    "evidence_spans_used": [],
}


@lru_cache(maxsize=1)
def _get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        warnings.warn("GROQ_API_KEY not set - verdicts will fall back to 'insufficient_evidence'.")
        return None
    from groq import Groq

    return Groq(api_key=api_key)


def _build_nli_prompt(claim_text: str, evidence: List[Dict]) -> str:
    if not evidence:
        evidence_block = "\n(No evidence snippets were retrieved.)\n"
    else:
        evidence_block = "\n".join(
            f"\n--- Evidence {i} ---\nTitle: {ev.get('title', 'N/A')}\nURL: {ev.get('url', 'N/A')}\n"
            f"Relevance: {ev.get('relevance_score', 'N/A')}\nContent:\n{ev.get('content', '')}\n"
            for i, ev in enumerate(evidence, 1)
        )

    return f"""Determine whether the CLAIM is supported or contradicted by the EVIDENCE below.
Reason ONLY from the evidence provided - do not use outside knowledge.
If the evidence is sparse or irrelevant, return "insufficient_evidence" rather than guessing.

CLAIM:
"{claim_text}"

EVIDENCE:
{evidence_block}

Respond with ONLY this JSON schema:
{{
  "verdict": "supported" | "contradicted" | "insufficient_evidence",
  "contradiction_score": <float 0.0-1.0, 0=fully supported, 1=fully contradicted>,
  "justification": "<1-2 sentences citing which evidence>",
  "evidence_spans_used": ["<URL1>", ...]
}}"""


def _call_groq_nli(claim_text: str, evidence: List[Dict]) -> Optional[Dict]:
    client = _get_groq_client()
    if client is None:
        return None

    prompt = _build_nli_prompt(claim_text, evidence)
    rate_limit_retries = 0
    attempt = 1
    while attempt <= MAX_RETRIES + 1:
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": _SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            data = json.loads(resp.choices[0].message.content.strip())
            required = {"verdict", "contradiction_score", "justification", "evidence_spans_used"}
            if not required.issubset(data.keys()):
                print(f"  [verdict attempt {attempt}] missing keys, retrying")
                time.sleep(0.5)
                attempt += 1
                continue
            if data["verdict"] not in {"supported", "contradicted", "insufficient_evidence"}:
                print(f"  [verdict attempt {attempt}] bad verdict value, retrying")
                time.sleep(0.5)
                attempt += 1
                continue
            data["contradiction_score"] = max(0.0, min(1.0, float(data["contradiction_score"])))
            return data
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_error(exc) and rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                rate_limit_retries += 1
                waited = wait_for_rate_limit(exc, rate_limit_retries)
                print(f"  [verdict attempt {attempt}] rate limited - waiting {waited:.1f}s "
                      f"(rate-limit retry {rate_limit_retries}/{RATE_LIMIT_MAX_RETRIES})")
                continue  # rate-limit waits don't consume a MAX_RETRIES attempt
            print(f"  [verdict attempt {attempt}] {type(exc).__name__}: {exc}")
            time.sleep(0.5 if attempt <= MAX_RETRIES else 0)
            attempt += 1

    return None


def compute_risk_score(
    p_ai_generated: float,
    p_claim: float,
    contradiction_score: float,
    w_ai: float = W_AI,
    w_claim: float = W_CLAIM,
    w_contradiction: float = W_CONTRADICTION,
) -> float:
    raw = w_ai * p_ai_generated + w_claim * p_claim + w_contradiction * contradiction_score
    return round(max(0.0, min(1.0, raw)) * 100, 2)


def assign_risk_tier(risk_score: float, verdict: Optional[str] = None) -> str:
    tier = "Low" if risk_score <= 33 else "Medium" if risk_score <= 66 else "High"
    if verdict == "insufficient_evidence" and tier == "High":
        tier = "Medium"  # don't conflate "unknown" with "confirmed false"
    return tier


def score_claim(claim: Dict) -> Dict:
    """Mutates and returns claim with verdict, contradiction_score,
    justification, evidence_spans_used, risk_score, risk_tier."""
    verdict_data = _call_groq_nli(claim.get("text", ""), claim.get("evidence", [])) or dict(_FALLBACK_VERDICT)

    claim["verdict"] = verdict_data["verdict"]
    claim["contradiction_score"] = verdict_data["contradiction_score"]
    claim["justification"] = verdict_data["justification"]
    claim["evidence_spans_used"] = verdict_data["evidence_spans_used"]
    claim["risk_score"] = compute_risk_score(
        p_ai_generated=claim.get("p_ai_generated", 0.0),
        p_claim=claim.get("p_claim", 0.0),
        contradiction_score=claim["contradiction_score"],
    )
    claim["risk_tier"] = assign_risk_tier(claim["risk_score"], verdict=claim["verdict"])
    return claim


def score_claims(claims: List[Dict]) -> List[Dict]:
    total = len(claims)
    for i, claim in enumerate(claims, 1):
        print(f"  [verdict] ({i}/{total}) \"{claim.get('text', '')[:80]}\"")
        score_claim(claim)
        if i < total:
            time.sleep(0.5)
    return claims


if __name__ == "__main__":
    print(compute_risk_score(p_ai_generated=0.9, p_claim=0.8, contradiction_score=0.0), "-> should be Low")
    print(compute_risk_score(p_ai_generated=0.9, p_claim=0.8, contradiction_score=1.0), "-> should be High")
