"""
study/database.py
────────────────────
SQLite layer for the reader-perception study (Objective 2) and its
evaluation (Objective 5). Standard-library sqlite3 only - no new
dependency to install.

Four tables:

    articles          one row per article used in the study, with its
                      real origin (human/ai) as ground truth - never
                      shown to the reader directly
    sessions          one row per reader session (anonymous, no login)
    ratings           one row per reader's 1-5 ratings on a single
                      article, plus which label they were SHOWN
                      ("ai" / "human" / "none") - which may differ from
                      the article's real origin, by design, so you can
                      test whether the label itself changes perception
    behavior_events   one row per tracked behavior event (time on page,
                      scroll depth, clicks, ...) tied to a session+article

Usage:
    from study.database import init_db, add_article, create_session, add_rating

    init_db()  # call once at startup (api.py does this)
    article_id = add_article(title="...", body_text="...", actual_origin="human")
    session_id = create_session()
    add_rating(session_id, article_id, shown_label="ai",
               credibility=4, clarity=3, neutrality=2, complexity=4,
               coherence=3, quality=3)
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "study.db")

RATING_DIMENSIONS = ["credibility", "clarity", "neutrality", "complexity", "coherence", "quality"]
VALID_ORIGINS = {"human", "ai"}
VALID_SHOWN_LABELS = {"human", "ai", "none"}
VALID_EVENT_TYPES = {"time_on_page", "scroll_depth", "click", "page_view"}

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id              TEXT PRIMARY KEY,
    title           TEXT,
    body_text       TEXT NOT NULL,
    actual_origin   TEXT NOT NULL CHECK (actual_origin IN ('human', 'ai')),
    topic           TEXT,
    source_url      TEXT,
    word_count      INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS sessions (
    id              TEXT PRIMARY KEY,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ratings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    article_id      TEXT NOT NULL REFERENCES articles(id),
    shown_label     TEXT NOT NULL CHECK (shown_label IN ('human', 'ai', 'none')),
    credibility     INTEGER NOT NULL CHECK (credibility BETWEEN 1 AND 5),
    clarity         INTEGER NOT NULL CHECK (clarity BETWEEN 1 AND 5),
    neutrality      INTEGER NOT NULL CHECK (neutrality BETWEEN 1 AND 5),
    complexity      INTEGER NOT NULL CHECK (complexity BETWEEN 1 AND 5),
    coherence       INTEGER NOT NULL CHECK (coherence BETWEEN 1 AND 5),
    quality         INTEGER NOT NULL CHECK (quality BETWEEN 1 AND 5),
    submitted_at    TEXT NOT NULL,
    UNIQUE (session_id, article_id)
);

CREATE TABLE IF NOT EXISTS behavior_events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(id),
    article_id      TEXT NOT NULL REFERENCES articles(id),
    event_type      TEXT NOT NULL CHECK (event_type IN ('time_on_page', 'scroll_depth', 'click', 'page_view')),
    value            REAL,
    recorded_at     TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_ratings_article ON ratings(article_id);
CREATE INDEX IF NOT EXISTS idx_ratings_session ON ratings(session_id);
CREATE INDEX IF NOT EXISTS idx_behavior_article ON behavior_events(article_id);
CREATE INDEX IF NOT EXISTS idx_behavior_session ON behavior_events(session_id);

-- ──────────────────────────────────────────────────────────────
-- General analysis history + feedback (api.py's /analyze, /feedback,
-- /history, /analytics). Separate from articles/ratings above, which
-- are reserved for the curated reader-study set with known ground
-- truth (actual_origin). Anything analyzed through the API - any
-- article a user pastes or links - lands here instead, with no claim
-- about whether it is actually AI-written or human-written.
-- ──────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS analyses (
    id                  TEXT PRIMARY KEY,
    title               TEXT,
    source_url          TEXT,
    article_text        TEXT NOT NULL,
    result_json         TEXT NOT NULL,
    credibility_score   REAL,
    credibility_tier    TEXT,
    overall_risk_score  REAL,
    overall_risk_tier   TEXT,
    num_claims_found    INTEGER,
    created_at          TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS feedback (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    analysis_id     TEXT NOT NULL REFERENCES analyses(id),
    session_id      TEXT,
    credibility     INTEGER NOT NULL CHECK (credibility BETWEEN 1 AND 5),
    clarity         INTEGER NOT NULL CHECK (clarity BETWEEN 1 AND 5),
    neutrality      INTEGER NOT NULL CHECK (neutrality BETWEEN 1 AND 5),
    complexity      INTEGER NOT NULL CHECK (complexity BETWEEN 1 AND 5),
    coherence       INTEGER NOT NULL CHECK (coherence BETWEEN 1 AND 5),
    quality         INTEGER NOT NULL CHECK (quality BETWEEN 1 AND 5),
    comment         TEXT,
    submitted_at    TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analyses_created ON analyses(created_at);
CREATE INDEX IF NOT EXISTS idx_feedback_analysis ON feedback(analysis_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


@contextmanager
def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        conn.executescript(SCHEMA)


# ──────────────────────────────────────────────────────────────
# Articles
# ──────────────────────────────────────────────────────────────
def add_article(
    title: str,
    body_text: str,
    actual_origin: str,
    topic: Optional[str] = None,
    source_url: Optional[str] = None,
    article_id: Optional[str] = None,
) -> str:
    if actual_origin not in VALID_ORIGINS:
        raise ValueError(f"actual_origin must be one of {VALID_ORIGINS}, got '{actual_origin}'")
    article_id = article_id or str(uuid.uuid4())
    word_count = len(body_text.split())
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO articles (id, title, body_text, actual_origin, topic, source_url, word_count, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (article_id, title, body_text, actual_origin, topic, source_url, word_count, _now()),
        )
    return article_id


def get_article(article_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM articles WHERE id = ?", (article_id,)).fetchone()
        return dict(row) if row else None


def list_articles(topic: Optional[str] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        if topic:
            rows = conn.execute("SELECT * FROM articles WHERE topic = ? ORDER BY created_at DESC", (topic,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM articles ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Sessions
# ──────────────────────────────────────────────────────────────
def create_session(session_id: Optional[str] = None) -> str:
    session_id = session_id or str(uuid.uuid4())
    with get_connection() as conn:
        conn.execute("INSERT OR IGNORE INTO sessions (id, created_at) VALUES (?, ?)", (session_id, _now()))
    return session_id


# ──────────────────────────────────────────────────────────────
# Ratings
# ──────────────────────────────────────────────────────────────
def add_rating(
    session_id: str,
    article_id: str,
    shown_label: str,
    credibility: int,
    clarity: int,
    neutrality: int,
    complexity: int,
    coherence: int,
    quality: int,
) -> int:
    if shown_label not in VALID_SHOWN_LABELS:
        raise ValueError(f"shown_label must be one of {VALID_SHOWN_LABELS}, got '{shown_label}'")
    values = {"credibility": credibility, "clarity": clarity, "neutrality": neutrality,
              "complexity": complexity, "coherence": coherence, "quality": quality}
    for dim, v in values.items():
        if not (1 <= int(v) <= 5):
            raise ValueError(f"{dim} must be an integer 1-5, got {v}")

    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO ratings (session_id, article_id, shown_label, credibility, clarity, "
            "neutrality, complexity, coherence, quality, submitted_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(session_id, article_id) DO UPDATE SET "
            "shown_label=excluded.shown_label, credibility=excluded.credibility, clarity=excluded.clarity, "
            "neutrality=excluded.neutrality, complexity=excluded.complexity, coherence=excluded.coherence, "
            "quality=excluded.quality, submitted_at=excluded.submitted_at",
            (session_id, article_id, shown_label, credibility, clarity, neutrality, complexity, coherence, quality, _now()),
        )
        return cur.lastrowid


def get_ratings_for_article(article_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM ratings WHERE article_id = ?", (article_id,)).fetchall()
        return [dict(r) for r in rows]


def get_all_ratings() -> List[Dict[str, Any]]:
    """Every rating joined with its article's real origin - the table
    Objective 5's analysis scripts will read from."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT r.*, a.actual_origin, a.title, a.topic "
            "FROM ratings r JOIN articles a ON r.article_id = a.id "
            "ORDER BY r.submitted_at"
        ).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Behavior events
# ──────────────────────────────────────────────────────────────
def add_behavior_event(session_id: str, article_id: str, event_type: str, value: Optional[float] = None) -> int:
    if event_type not in VALID_EVENT_TYPES:
        raise ValueError(f"event_type must be one of {VALID_EVENT_TYPES}, got '{event_type}'")
    with get_connection() as conn:
        cur = conn.execute(
            "INSERT INTO behavior_events (session_id, article_id, event_type, value, recorded_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, article_id, event_type, value, _now()),
        )
        return cur.lastrowid


def get_behavior_for_article(article_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM behavior_events WHERE article_id = ?", (article_id,)).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Survey delivery: which article to show next, and under which label
# ──────────────────────────────────────────────────────────────
SHOWN_LABEL_CHOICES = ["human", "ai", "none"]


def assign_shown_label(session_id: str, article_id: str) -> str:
    """Deterministic (not random-per-reload) assignment of which label
    a reader is told for an article - independent of actual_origin, by
    design, so the study can measure whether the LABEL alone shifts
    perception. Deterministic on (session_id, article_id) so refreshing
    the page doesn't re-roll the condition for the same reader+article."""
    digest = hashlib.sha256(f"{session_id}:{article_id}".encode("utf-8")).hexdigest()
    idx = int(digest, 16) % len(SHOWN_LABEL_CHOICES)
    return SHOWN_LABEL_CHOICES[idx]


def get_next_article_for_session(session_id: str) -> Optional[Dict[str, Any]]:
    """Picks an article this session hasn't rated yet, preferring
    whichever article currently has the FEWEST total ratings overall
    (balances exposure across the curated set instead of always
    serving articles in the same order). Returns None once the
    session has rated every article in the study set."""
    with get_connection() as conn:
        rows = conn.execute(
            """
            SELECT a.id, a.title, a.body_text, a.topic, a.source_url,
                   COUNT(r.id) AS rating_count
            FROM articles a
            LEFT JOIN ratings r ON r.article_id = a.id
            WHERE a.id NOT IN (
                SELECT article_id FROM ratings WHERE session_id = ?
            )
            GROUP BY a.id
            ORDER BY rating_count ASC, a.created_at ASC
            LIMIT 1
            """,
            (session_id,),
        ).fetchall()
    if not rows:
        return None
    row = dict(rows[0])
    row.pop("rating_count", None)
    row["shown_label"] = assign_shown_label(session_id, row["id"])
    return row


def get_study_progress(session_id: str) -> Dict[str, Any]:
    with get_connection() as conn:
        total = conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()["n"]
        rated = conn.execute(
            "SELECT COUNT(*) AS n FROM ratings WHERE session_id = ?", (session_id,)
        ).fetchone()["n"]
    return {"total_articles": total, "rated_by_session": rated, "remaining": max(0, total - rated)}


# ──────────────────────────────────────────────────────────────
# Aggregates (used later by the dashboard, step 6)
# ──────────────────────────────────────────────────────────────
def get_article_summary(article_id: str) -> Dict[str, Any]:
    """Mean of each rating dimension + behavior stats for one article."""
    ratings = get_ratings_for_article(article_id)
    events = get_behavior_for_article(article_id)

    means = {}
    if ratings:
        for dim in RATING_DIMENSIONS:
            means[dim] = round(sum(r[dim] for r in ratings) / len(ratings), 2)

    time_values = [e["value"] for e in events if e["event_type"] == "time_on_page" and e["value"] is not None]
    scroll_values = [e["value"] for e in events if e["event_type"] == "scroll_depth" and e["value"] is not None]

    return {
        "article_id": article_id,
        "num_ratings": len(ratings),
        "mean_ratings": means,
        "mean_time_on_page_seconds": round(sum(time_values) / len(time_values), 1) if time_values else None,
        "mean_scroll_depth_pct": round(sum(scroll_values) / len(scroll_values), 1) if scroll_values else None,
    }


# ──────────────────────────────────────────────────────────────
# Analysis history (api.py's /analyze persists here, /history reads it)
# ──────────────────────────────────────────────────────────────
def save_analysis(
    title: Optional[str],
    source_url: Optional[str],
    article_text: str,
    result: Dict[str, Any],
    analysis_id: Optional[str] = None,
) -> str:
    analysis_id = analysis_id or str(uuid.uuid4())
    cred = result.get("credibility") or {}
    with get_connection() as conn:
        conn.execute(
            "INSERT INTO analyses (id, title, source_url, article_text, result_json, credibility_score, "
            "credibility_tier, overall_risk_score, overall_risk_tier, num_claims_found, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                analysis_id, title, source_url, article_text, json.dumps(result),
                cred.get("score"), cred.get("tier"),
                result.get("overall_risk_score"), result.get("overall_risk_tier"),
                result.get("num_claims_found"), _now(),
            ),
        )
    return analysis_id


def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    with get_connection() as conn:
        row = conn.execute("SELECT * FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not row:
            return None
        d = dict(row)
        d["result"] = json.loads(d.pop("result_json"))
        return d


def list_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT id, title, source_url, credibility_score, credibility_tier, overall_risk_score, "
            "overall_risk_tier, num_claims_found, created_at FROM analyses "
            "ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Reader feedback on a specific analysis (Objective 2)
# ──────────────────────────────────────────────────────────────
def add_feedback(
    analysis_id: str,
    credibility: int,
    clarity: int,
    neutrality: int,
    complexity: int,
    coherence: int,
    quality: int,
    session_id: Optional[str] = None,
    comment: Optional[str] = None,
) -> int:
    values = {"credibility": credibility, "clarity": clarity, "neutrality": neutrality,
              "complexity": complexity, "coherence": coherence, "quality": quality}
    for dim, v in values.items():
        if not (1 <= int(v) <= 5):
            raise ValueError(f"{dim} must be an integer 1-5, got {v}")

    with get_connection() as conn:
        exists = conn.execute("SELECT 1 FROM analyses WHERE id = ?", (analysis_id,)).fetchone()
        if not exists:
            raise ValueError(f"No analysis with id '{analysis_id}' - feedback must reference a real analysis.")
        cur = conn.execute(
            "INSERT INTO feedback (analysis_id, session_id, credibility, clarity, neutrality, "
            "complexity, coherence, quality, comment, submitted_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, session_id, credibility, clarity, neutrality, complexity, coherence,
             quality, comment, _now()),
        )
        return cur.lastrowid


def get_feedback_for_analysis(analysis_id: str) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("SELECT * FROM feedback WHERE analysis_id = ?", (analysis_id,)).fetchall()
        return [dict(r) for r in rows]


# ──────────────────────────────────────────────────────────────
# Analytics (api.py's /analytics, dashboard data source)
# ──────────────────────────────────────────────────────────────
def get_analytics_summary() -> Dict[str, Any]:
    """Aggregates computed only from real stored rows - fields come
    back None/empty rather than invented when there isn't data yet."""
    with get_connection() as conn:
        analyses = [dict(r) for r in conn.execute(
            "SELECT id, credibility_score, credibility_tier, overall_risk_score, overall_risk_tier, "
            "num_claims_found, created_at FROM analyses ORDER BY created_at"
        ).fetchall()]
        feedback_rows = [dict(r) for r in conn.execute("SELECT * FROM feedback").fetchall()]

    tier_counts: Dict[str, int] = {}
    for a in analyses:
        t = a.get("credibility_tier") or "Unknown"
        tier_counts[t] = tier_counts.get(t, 0) + 1

    cred_scores = [a["credibility_score"] for a in analyses if a["credibility_score"] is not None]
    risk_scores = [a["overall_risk_score"] for a in analyses if a["overall_risk_score"] is not None]

    mean_feedback: Dict[str, float] = {}
    if feedback_rows:
        for dim in RATING_DIMENSIONS:
            mean_feedback[dim] = round(sum(f[dim] for f in feedback_rows) / len(feedback_rows), 2)

    return {
        "num_analyses": len(analyses),
        "num_feedback_submissions": len(feedback_rows),
        "mean_credibility_score": round(sum(cred_scores) / len(cred_scores), 2) if cred_scores else None,
        "mean_overall_risk_score": round(sum(risk_scores) / len(risk_scores), 2) if risk_scores else None,
        "credibility_tier_distribution": tier_counts,
        "mean_reader_feedback": mean_feedback,
        "credibility_score_trend": [
            {"created_at": a["created_at"], "credibility_score": a["credibility_score"]}
            for a in analyses if a["credibility_score"] is not None
        ],
    }


if __name__ == "__main__":
    # Self-contained smoke test - run with: python -m study.database
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        DB_PATH = os.path.join(tmp, "test_study.db")
        init_db()
        print(f"[test] schema created at {DB_PATH}")

        aid = add_article(
            title="Test headline",
            body_text="This is a test article body with several words in it for word counting.",
            actual_origin="human",
            topic="politics",
        )
        print(f"[test] article added: {aid}")

        sid = create_session()
        print(f"[test] session created: {sid}")

        rid = add_rating(sid, aid, shown_label="ai", credibility=4, clarity=3,
                          neutrality=2, complexity=4, coherence=3, quality=3)
        print(f"[test] rating added: id={rid}")

        beid = add_behavior_event(sid, aid, "time_on_page", value=42.5)
        add_behavior_event(sid, aid, "scroll_depth", value=87.0)
        print(f"[test] behavior events added")

        summary = get_article_summary(aid)
        print(f"[test] article summary: {summary}")

        all_ratings = get_all_ratings()
        print(f"[test] get_all_ratings returned {len(all_ratings)} row(s): {all_ratings}")

        assert summary["num_ratings"] == 1
        assert summary["mean_ratings"]["credibility"] == 4.0
        assert summary["mean_time_on_page_seconds"] == 42.5
        assert summary["mean_scroll_depth_pct"] == 87.0
        assert len(all_ratings) == 1
        assert all_ratings[0]["actual_origin"] == "human"
        assert all_ratings[0]["shown_label"] == "ai"

        analysis_id = save_analysis(
            title="Test headline", source_url=None, article_text="Some article text.",
            result={"credibility": {"score": 61.1, "tier": "Moderate"}, "overall_risk_score": 41.0,
                    "overall_risk_tier": "Medium", "num_claims_found": 2},
        )
        print(f"[test] analysis saved: {analysis_id}")
        fetched = get_analysis(analysis_id)
        assert fetched["result"]["credibility"]["score"] == 61.1
        assert len(list_analyses()) == 1

        fid = add_feedback(analysis_id, credibility=4, clarity=4, neutrality=3,
                            complexity=3, coherence=4, quality=4, comment="clear enough")
        print(f"[test] feedback added: id={fid}")
        assert len(get_feedback_for_analysis(analysis_id)) == 1

        summary = get_analytics_summary()
        assert summary["num_analyses"] == 1
        assert summary["num_feedback_submissions"] == 1
        assert summary["mean_credibility_score"] == 61.1
        assert summary["mean_reader_feedback"]["credibility"] == 4.0
        print(f"[test] analytics summary: {summary}")

        print("\nALL DATABASE TESTS PASSED")
