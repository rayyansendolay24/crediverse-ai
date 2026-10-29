"""
tests/test_ai_detector_eval.py
───────────────────────────────────
Mocked tests for calibration/ai_detector_eval.py - checks the metric
math (confusion matrix, precision/recall/F1, accuracy, threshold
behavior, the "not enough data" guard) without loading the real spaCy
or RoBERTa models. core.claim_detection.analyze_article_sentences /
summarize_ai_signal are patched to return fixed p_ai values per article
text, keyed by a marker embedded in the (fake) article text itself.

Run with:  python tests/test_ai_detector_eval.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from calibration.ai_detector_eval import evaluate_ai_detector

# Four articles: a correctly-predicted human, a correctly-predicted ai,
# and one of each that the (fake) detector gets WRONG - so accuracy
# should land at exactly 50% (2/4) and every per-class metric is
# checkable by hand.
_ARTICLES = [
    {"id": "h-correct", "title": "H1", "topic": "t1", "body_text": "HUMAN_LOW", "actual_origin": "human"},
    {"id": "a-correct", "title": "A1", "topic": "t1", "body_text": "AI_HIGH", "actual_origin": "ai"},
    {"id": "h-wrong", "title": "H2", "topic": "t2", "body_text": "HUMAN_HIGH", "actual_origin": "human"},
    {"id": "a-wrong", "title": "A2", "topic": "t2", "body_text": "AI_LOW", "actual_origin": "ai"},
]

_P_AI_BY_TEXT = {
    "HUMAN_LOW": 0.10,
    "AI_HIGH": 0.90,
    "HUMAN_HIGH": 0.80,   # human article, but detector scores it high -> misclassified as ai
    "AI_LOW": 0.20,        # ai article, but detector scores it low -> misclassified as human
}


def _fake_analyze(article_text):
    return [{"text": article_text}]


def _fake_summarize(sentences):
    text = sentences[0]["text"]
    return {"article_p_ai": _P_AI_BY_TEXT[text]}


def test_not_enough_data_returns_unavailable():
    result = evaluate_ai_detector(articles=[
        {"id": "h1", "title": "H", "topic": None, "body_text": "x", "actual_origin": "human"},
        {"id": "a1", "title": "A", "topic": None, "body_text": "y", "actual_origin": "ai"},
    ])
    assert result["available"] is False
    assert "Need at least 2" in result["reason"]
    print("test_not_enough_data_returns_unavailable: OK")


@patch("core.claim_detection.summarize_ai_signal", side_effect=_fake_summarize)
@patch("core.claim_detection.analyze_article_sentences", side_effect=_fake_analyze)
def test_accuracy_and_confusion_matrix_at_default_threshold(mock_analyze, mock_summarize):
    result = evaluate_ai_detector(threshold=0.5, articles=_ARTICLES)

    assert result["available"] is True
    assert result["n"] == 4
    assert result["accuracy_pct"] == 50.0

    cm = result["confusion_matrix"]
    assert cm["human"]["human"] == 1 and cm["human"]["ai"] == 1
    assert cm["ai"]["ai"] == 1 and cm["ai"]["human"] == 1

    per_class = result["per_class"]
    assert per_class["human"]["precision"] == 0.5
    assert per_class["human"]["recall"] == 0.5
    assert per_class["ai"]["precision"] == 0.5
    assert per_class["ai"]["recall"] == 0.5
    assert per_class["macro_avg"]["support"] == 4

    means = result["mean_p_ai_generated_by_actual_origin"]
    assert means["human"] == round((0.10 + 0.80) / 2, 4)
    assert means["ai"] == round((0.90 + 0.20) / 2, 4)

    assert {a["article_id"] for a in result["per_article"] if a["correct"]} == {"h-correct", "a-correct"}
    print("test_accuracy_and_confusion_matrix_at_default_threshold: OK")


@patch("core.claim_detection.summarize_ai_signal", side_effect=_fake_summarize)
@patch("core.claim_detection.analyze_article_sentences", side_effect=_fake_analyze)
def test_threshold_changes_predictions(mock_analyze, mock_summarize):
    # At a very low threshold (0.05), everything with any p_ai at all gets
    # predicted "ai" - so both actual humans are misclassified, both actual
    # ai articles are correctly classified -> accuracy = 50% again, but via
    # a totally different confusion pattern (sanity check threshold is used).
    result = evaluate_ai_detector(threshold=0.05, articles=_ARTICLES)
    cm = result["confusion_matrix"]
    assert cm["human"]["ai"] == 2 and cm["human"]["human"] == 0
    assert cm["ai"]["ai"] == 2 and cm["ai"]["human"] == 0
    print("test_threshold_changes_predictions: OK")


if __name__ == "__main__":
    test_not_enough_data_returns_unavailable()
    test_accuracy_and_confusion_matrix_at_default_threshold()
    test_threshold_changes_predictions()
    print("\nALL AI-DETECTOR-EVAL TESTS PASSED")
