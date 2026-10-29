"""
core/rate_limit.py
────────────────────
Shared rate-limit detection + backoff for Groq API calls, used by
core/verdict.py and analysis/llm_judge.py.

This module is consulted only inside those modules' existing
except-clauses, and only to decide whether a raised exception is a
rate limit (429) response. When it is, the caller waits longer
(respecting the server's Retry-After header when present, otherwise
exponential backoff with jitter) before trying again, and that wait
does NOT consume one of the caller's existing MAX_RETRIES attempts.
Any other exception falls through to the caller's original
non-rate-limit handling, unchanged.
"""

from __future__ import annotations

import random
import time
from typing import Optional

# Rate-limit waits are bounded separately from MAX_RETRIES in the callers -
# this is "how many extra times will we wait out a 429" before giving up
# and letting the caller's normal fallback behavior take over.
RATE_LIMIT_MAX_RETRIES = 5
RATE_LIMIT_BASE_SECONDS = 1.0
RATE_LIMIT_MAX_WAIT_SECONDS = 30.0


def is_rate_limit_error(exc: BaseException) -> bool:
    """True if `exc` looks like a 429 / rate-limit response from Groq
    (or any OpenAI-compatible client raising the same exception shape)."""
    status = getattr(exc, "status_code", None)
    if status is None:
        response = getattr(exc, "response", None)
        status = getattr(response, "status_code", None)
    if status == 429:
        return True
    if "RateLimit" in type(exc).__name__:
        return True
    msg = str(exc).lower()
    return "rate limit" in msg or "rate_limit" in msg or "429" in msg


def _retry_after_seconds(exc: BaseException) -> Optional[float]:
    """Reads a Retry-After header off the exception's HTTP response, if
    the client library attached one (groq's APIStatusError does)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    value = None
    try:
        value = headers.get("Retry-After") or headers.get("retry-after")
    except AttributeError:
        return None
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def compute_backoff_seconds(exc: BaseException, attempt: int) -> float:
    """Seconds to wait before the next attempt (`attempt` is 1-indexed,
    counting rate-limit waits only). Honors the server's Retry-After
    header when present; otherwise exponential backoff with jitter,
    capped at RATE_LIMIT_MAX_WAIT_SECONDS so a single call can't stall
    a request indefinitely."""
    retry_after = _retry_after_seconds(exc)
    if retry_after is not None:
        return min(retry_after, RATE_LIMIT_MAX_WAIT_SECONDS)
    backoff = RATE_LIMIT_BASE_SECONDS * (2 ** (attempt - 1))
    jitter = random.uniform(0, RATE_LIMIT_BASE_SECONDS)
    return min(backoff + jitter, RATE_LIMIT_MAX_WAIT_SECONDS)


def wait_for_rate_limit(exc: BaseException, attempt: int, sleep=time.sleep) -> float:
    """Sleeps the computed backoff and returns how many seconds it slept.
    Tests pass a fake `sleep` (e.g. a no-op) so they don't actually wait."""
    seconds = compute_backoff_seconds(exc, attempt)
    sleep(seconds)
    return seconds
