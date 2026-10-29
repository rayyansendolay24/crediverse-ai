"""
tests/test_study_survey.py
───────────────────────────────
Tests for the survey-delivery logic added to study/database.py:
assign_shown_label, get_next_article_for_session, get_study_progress.
Uses a temp SQLite file, no network or API keys needed.

Run with:  python tests/test_study_survey.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import study.database as db


def _fresh_db():
    tmp_dir = tempfile.mkdtemp()
    db.DB_PATH = os.path.join(tmp_dir, "test_survey.db")
    db.init_db()


def test_shown_label_is_deterministic_per_pair():
    _fresh_db()
    a = db.assign_shown_label("session-1", "article-1")
    b = db.assign_shown_label("session-1", "article-1")
    assert a == b, "same (session, article) must always get the same shown_label"
    assert a in db.SHOWN_LABEL_CHOICES
    print("test_shown_label_is_deterministic_per_pair: OK")


def test_next_article_balances_exposure_and_excludes_rated():
    _fresh_db()
    a1 = db.add_article(title="Story A", body_text="x " * 50, actual_origin="human")
    a2 = db.add_article(title="Story B", body_text="y " * 50, actual_origin="ai")
    sid = db.create_session()

    first = db.get_next_article_for_session(sid)
    assert first is not None and first["id"] in (a1, a2)
    assert "shown_label" in first and first["shown_label"] in db.SHOWN_LABEL_CHOICES

    db.add_rating(sid, first["id"], shown_label=first["shown_label"],
                  credibility=3, clarity=3, neutrality=3, complexity=3, coherence=3, quality=3)

    second = db.get_next_article_for_session(sid)
    assert second is not None
    assert second["id"] != first["id"], "already-rated article must not be served again to the same session"

    db.add_rating(sid, second["id"], shown_label=second["shown_label"],
                  credibility=3, clarity=3, neutrality=3, complexity=3, coherence=3, quality=3)

    done = db.get_next_article_for_session(sid)
    assert done is None, "session has rated every article - should get None, not repeat one"
    print("test_next_article_balances_exposure_and_excludes_rated: OK")


def test_next_article_prefers_least_rated_across_sessions():
    _fresh_db()
    a1 = db.add_article(title="Story A", body_text="x " * 50, actual_origin="human")
    a2 = db.add_article(title="Story B", body_text="y " * 50, actual_origin="ai")

    # Two different sessions both rate article a1 first, leaving a2 under-rated.
    s1 = db.create_session()
    db.add_rating(s1, a1, shown_label="human", credibility=3, clarity=3, neutrality=3,
                  complexity=3, coherence=3, quality=3)
    s2 = db.create_session()
    db.add_rating(s2, a1, shown_label="ai", credibility=3, clarity=3, neutrality=3,
                  complexity=3, coherence=3, quality=3)

    s3 = db.create_session()
    served = db.get_next_article_for_session(s3)
    assert served["id"] == a2, "under-rated article (0 ratings) should be served before the one already rated twice"
    print("test_next_article_prefers_least_rated_across_sessions: OK")


def test_progress_tracks_rated_count():
    _fresh_db()
    a1 = db.add_article(title="Story A", body_text="x " * 50, actual_origin="human")
    a2 = db.add_article(title="Story B", body_text="y " * 50, actual_origin="ai")
    sid = db.create_session()

    p0 = db.get_study_progress(sid)
    assert p0 == {"total_articles": 2, "rated_by_session": 0, "remaining": 2}

    db.add_rating(sid, a1, shown_label="human", credibility=3, clarity=3, neutrality=3,
                  complexity=3, coherence=3, quality=3)
    p1 = db.get_study_progress(sid)
    assert p1 == {"total_articles": 2, "rated_by_session": 1, "remaining": 1}
    print("test_progress_tracks_rated_count: OK")


if __name__ == "__main__":
    test_shown_label_is_deterministic_per_pair()
    test_next_article_balances_exposure_and_excludes_rated()
    test_next_article_prefers_least_rated_across_sessions()
    test_progress_tracks_rated_count()
    print("\nALL STUDY-SURVEY TESTS PASSED")
