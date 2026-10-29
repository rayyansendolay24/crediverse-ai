"""
analysis/llm_judge.py
────────────────────────
LLM-as-judge rating an article on the same six dimensions the reader
survey uses, so the machine's perception can be compared against real
readers' later:

    credibility, clarity, neutrality, complexity, coherence, quality  (1-5)

`complexity` is descriptive (1=simple, 5=complex), not good/bad. Factual
accuracy is judged separately, in core/verdict.py, using web evidence -
this module only judges how the text *reads*.
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, Optional

from core.rate_limit import (
    RATE_LIMIT_MAX_RETRIES,
    is_rate_limit_error,
    wait_for_rate_limit,
)
from core.verdict import GROQ_MODEL, _get_groq_client

DIMENSIONS = ["credibility", "clarity", "neutrality", "complexity", "coherence", "quality"]
MAX_RETRIES = 2

_SYSTEM = (
    "You are an experienced news editor simulating how an ordinary adult reader "
    "perceives a news article. Respond with a single valid JSON object only."
)


def _build_prompt(article_text: str, title: Optional[str]) -> str:
    head = f'HEADLINE: "{title}"\n\n' if title else ""
    return f"""Rate the article below as an ordinary reader would perceive it.
Judge presentation and writing style, NOT factual accuracy (verified separately).

Scale 1-5 for every dimension:
- credibility: how trustworthy it feels (sourcing, tone, specificity)
- clarity: how easy it is to understand
- neutrality: 1=strongly slanted/emotive, 5=balanced and impartial
- complexity: 1=very simple, 5=very complex (descriptive, not good/bad)
- coherence: how logically ideas flow sentence to sentence
- quality: overall writing quality

{head}ARTICLE:
\"\"\"
{article_text}
\"\"\"

Return ONLY this JSON:
{{
  "ratings": {{"credibility": int, "clarity": int, "neutrality": int, "complexity": int, "coherence": int, "quality": int}},
  "reasons": {{"credibility": "<one sentence>", "clarity": "<one sentence>", "neutrality": "<one sentence>", "complexity": "<one sentence>", "coherence": "<one sentence>", "quality": "<one sentence>"}},
  "overall_reasoning": "<2-3 sentences>"
}}"""


def _validate(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    ratings = data.get("ratings")
    if not isinstance(ratings, dict):
        return None
    clean: Dict[str, int] = {}
    for d in DIMENSIONS:
        try:
            clean[d] = int(max(1, min(5, round(float(ratings[d])))))
        except (KeyError, TypeError, ValueError):
            return None
    reasons = data.get("reasons") if isinstance(data.get("reasons"), dict) else {}
    return {
        "ratings": clean,
        "reasons": {d: str(reasons.get(d, "")) for d in DIMENSIONS},
        "overall_reasoning": str(data.get("overall_reasoning", "")),
        "model": GROQ_MODEL,
    }


def judge_article(article_text: str, title: Optional[str] = None, max_chars: int = 6000) -> Optional[Dict[str, Any]]:
    client = _get_groq_client()
    if client is None:
        print("  [llm_judge] no Groq client - skipping.")
        return None

    prompt = _build_prompt(article_text[:max_chars], title)
    rate_limit_retries = 0
    attempt = 1
    while attempt <= MAX_RETRIES + 1:
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "system", "content": _SYSTEM}, {"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"},
            )
            result = _validate(json.loads(resp.choices[0].message.content.strip()))
            if result is not None:
                return result
            print(f"  [llm_judge attempt {attempt}] invalid structure, retrying")
        except Exception as exc:  # noqa: BLE001
            if is_rate_limit_error(exc) and rate_limit_retries < RATE_LIMIT_MAX_RETRIES:
                rate_limit_retries += 1
                waited = wait_for_rate_limit(exc, rate_limit_retries)
                print(f"  [llm_judge attempt {attempt}] rate limited - waiting {waited:.1f}s "
                      f"(rate-limit retry {rate_limit_retries}/{RATE_LIMIT_MAX_RETRIES})")
                continue  # rate-limit waits don't consume a MAX_RETRIES attempt
            print(f"  [llm_judge attempt {attempt}] {type(exc).__name__}: {exc}")
        time.sleep(0.5)
        attempt += 1
    return None
