"""
tests/test_rate_limit.py
────────────────────────────
Mocked tests for Priority 2 (Groq rate-limit handling). Verifies:

  - is_rate_limit_error correctly detects 429s in a few shapes
  - Retry-After header is honored when present
  - exponential backoff is used otherwise, capped at the max wait
  - core.verdict._call_groq_nli retries through rate limits and still
    succeeds once the client stops raising
  - core.verdict._call_groq_nli gives up after RATE_LIMIT_MAX_RETRIES
    rate-limit errors and falls back (same fallback behavior as any
    other exhausted-retries case)
  - a NON-rate-limit exception still only gets MAX_RETRIES+1 attempts
    (i.e. rate-limit handling didn't change existing retry counts)

Run with:  python -m tests.test_rate_limit
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_is_rate_limit_error_detects_shapes():
    from core.rate_limit import is_rate_limit_error

    class FakeRateLimitError(Exception):
        pass

    assert is_rate_limit_error(FakeRateLimitError("boom")) is True  # name-based
    assert is_rate_limit_error(Exception("429 Too Many Requests")) is True  # message-based

    class StatusExc(Exception):
        status_code = 429

    assert is_rate_limit_error(StatusExc()) is True

    class ResponseExc(Exception):
        def __init__(self):
            self.response = SimpleNamespace(status_code=429, headers={})

    assert is_rate_limit_error(ResponseExc()) is True
    assert is_rate_limit_error(ValueError("totally unrelated")) is False
    print("test_is_rate_limit_error_detects_shapes: OK")


def test_retry_after_header_is_honored():
    from core.rate_limit import compute_backoff_seconds

    class ExcWithRetryAfter(Exception):
        def __init__(self, seconds):
            self.response = SimpleNamespace(status_code=429, headers={"Retry-After": str(seconds)})

    seconds = compute_backoff_seconds(ExcWithRetryAfter(7), attempt=1)
    assert seconds == 7.0, seconds
    print("test_retry_after_header_is_honored: OK")


def test_backoff_without_header_is_bounded():
    from core.rate_limit import RATE_LIMIT_MAX_WAIT_SECONDS, compute_backoff_seconds

    for attempt in range(1, 10):
        seconds = compute_backoff_seconds(Exception("429"), attempt=attempt)
        assert 0 <= seconds <= RATE_LIMIT_MAX_WAIT_SECONDS, seconds
    print("test_backoff_without_header_is_bounded: OK")


def test_verdict_retries_through_rate_limit_then_succeeds():
    import core.verdict as verdict_mod

    class FakeRateLimitError(Exception):
        status_code = 429

    call_count = {"n": 0}

    def fake_create(**kwargs):
        call_count["n"] += 1
        if call_count["n"] < 3:
            raise FakeRateLimitError("slow down")
        msg = SimpleNamespace(content='{"verdict": "supported", "contradiction_score": 0.1, '
                                       '"justification": "ok", "evidence_spans_used": []}')
        choice = SimpleNamespace(message=msg)
        return SimpleNamespace(choices=[choice])

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=fake_create)))

    with patch.object(verdict_mod, "_get_groq_client", return_value=fake_client), \
         patch.object(verdict_mod, "wait_for_rate_limit", return_value=0.0), \
         patch.object(verdict_mod.time, "sleep", return_value=None):
        result = verdict_mod._call_groq_nli("some claim", [])

    assert result is not None
    assert result["verdict"] == "supported"
    assert call_count["n"] == 3, call_count  # 2 rate-limited calls + 1 success, none counted against MAX_RETRIES
    print("test_verdict_retries_through_rate_limit_then_succeeds: OK")


def test_verdict_falls_back_after_rate_limit_exhausted():
    import core.verdict as verdict_mod
    from core.rate_limit import RATE_LIMIT_MAX_RETRIES

    class FakeRateLimitError(Exception):
        status_code = 429

    call_count = {"n": 0}

    def always_rate_limited(**kwargs):
        call_count["n"] += 1
        raise FakeRateLimitError("still slow down")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=always_rate_limited)))

    with patch.object(verdict_mod, "_get_groq_client", return_value=fake_client), \
         patch.object(verdict_mod, "wait_for_rate_limit", return_value=0.0), \
         patch.object(verdict_mod.time, "sleep", return_value=None):
        result = verdict_mod._call_groq_nli("some claim", [])

    # RATE_LIMIT_MAX_RETRIES rate-limited attempts, then falls through to the
    # normal MAX_RETRIES+1 non-rate-limit attempt budget (which also raises
    # the same error, now past the rate-limit retry budget) -> None.
    assert result is None
    assert call_count["n"] == RATE_LIMIT_MAX_RETRIES + (verdict_mod.MAX_RETRIES + 1), call_count
    print("test_verdict_falls_back_after_rate_limit_exhausted: OK")


def test_non_rate_limit_error_keeps_original_retry_count():
    import core.verdict as verdict_mod

    call_count = {"n": 0}

    def always_broken(**kwargs):
        call_count["n"] += 1
        raise ValueError("totally unrelated failure")

    fake_client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=always_broken)))

    with patch.object(verdict_mod, "_get_groq_client", return_value=fake_client), \
         patch.object(verdict_mod.time, "sleep", return_value=None):
        result = verdict_mod._call_groq_nli("some claim", [])

    assert result is None
    assert call_count["n"] == verdict_mod.MAX_RETRIES + 1, call_count  # unchanged from original behavior
    print("test_non_rate_limit_error_keeps_original_retry_count: OK")


if __name__ == "__main__":
    test_is_rate_limit_error_detects_shapes()
    test_retry_after_header_is_honored()
    test_backoff_without_header_is_bounded()
    test_verdict_retries_through_rate_limit_then_succeeds()
    test_verdict_falls_back_after_rate_limit_exhausted()
    test_non_rate_limit_error_keeps_original_retry_count()
    print("\nALL RATE-LIMIT TESTS PASSED")
