"""
study/api_routes.py
────────────────────────
FastAPI router for the curated reader-perception study (Objective 2's
"articles" / "ratings" / "behavior_events" tables - distinct from the
general /analyze -> /feedback path already in api.py).

    POST /study/articles      -> add an article to the curated set
                                  (admin/seeding use - see study/seed_articles.py
                                  for loading many at once from files)
    GET  /study/articles      -> list the curated set (admin/debug)
    POST /study/session        -> create/register a reader session
    GET  /study/next           -> next not-yet-rated article for a session,
                                  with its assigned shown_label
    POST /study/rating         -> submit a reader's 1-5 ratings for one article
    POST /study/behavior        -> log a behavior event (time/scroll/click)
    GET  /study/progress        -> how many of the curated articles this
                                  session has rated

Mounted into api.py with `app.include_router(study_router)`.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from study import database as db

study_router = APIRouter(prefix="/study", tags=["reader-study"])


class AddArticleRequest(BaseModel):
    title: str
    body_text: str
    actual_origin: str = Field(..., description="'human' or 'ai' - the REAL, known origin. Never shown to the reader.")
    topic: Optional[str] = None
    source_url: Optional[str] = None


class SessionResponse(BaseModel):
    session_id: str


class RatingRequest(BaseModel):
    session_id: str
    article_id: str
    shown_label: str = Field(..., description="The label the reader was actually shown - echo back what /study/next returned.")
    credibility: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    neutrality: int = Field(..., ge=1, le=5)
    complexity: int = Field(..., ge=1, le=5)
    coherence: int = Field(..., ge=1, le=5)
    quality: int = Field(..., ge=1, le=5)


class BehaviorRequest(BaseModel):
    session_id: str
    article_id: str
    event_type: str = Field(..., description="'time_on_page' | 'scroll_depth' | 'click' | 'page_view'")
    value: Optional[float] = None


@study_router.post("/articles", summary="Add one article to the curated study set")
def add_article_endpoint(payload: AddArticleRequest) -> Dict[str, Any]:
    try:
        article_id = db.add_article(
            title=payload.title,
            body_text=payload.body_text,
            actual_origin=payload.actual_origin,
            topic=payload.topic,
            source_url=payload.source_url,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"article_id": article_id, "status": "saved"}


@study_router.get("/articles", summary="List the curated study set (reveals actual_origin - admin/debug only)")
def list_articles_endpoint(topic: Optional[str] = None) -> List[Dict[str, Any]]:
    return db.list_articles(topic=topic)


@study_router.post("/session", summary="Create or re-register a reader session")
def create_session_endpoint(session_id: Optional[str] = None) -> SessionResponse:
    sid = db.create_session(session_id=session_id)
    return SessionResponse(session_id=sid)


@study_router.get("/next", summary="Get the next article this session hasn't rated yet")
def next_article_endpoint(session_id: str) -> Dict[str, Any]:
    db.create_session(session_id=session_id)  # idempotent - ensures the FK target exists
    article = db.get_next_article_for_session(session_id)
    if article is None:
        return {"done": True, "article": None}
    return {"done": False, "article": article}


@study_router.post("/rating", summary="Submit a reader's 1-5 ratings for one article")
def submit_rating_endpoint(payload: RatingRequest) -> Dict[str, Any]:
    try:
        rating_id = db.add_rating(
            session_id=payload.session_id,
            article_id=payload.article_id,
            shown_label=payload.shown_label,
            credibility=payload.credibility,
            clarity=payload.clarity,
            neutrality=payload.neutrality,
            complexity=payload.complexity,
            coherence=payload.coherence,
            quality=payload.quality,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"rating_id": rating_id, "status": "saved"}


@study_router.post("/behavior", summary="Log a behavior event (time on page, scroll depth, click)")
def log_behavior_endpoint(payload: BehaviorRequest) -> Dict[str, Any]:
    try:
        event_id = db.add_behavior_event(
            session_id=payload.session_id,
            article_id=payload.article_id,
            event_type=payload.event_type,
            value=payload.value,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"event_id": event_id, "status": "saved"}


@study_router.get("/progress", summary="How many curated articles this session has rated")
def progress_endpoint(session_id: str) -> Dict[str, Any]:
    return db.get_study_progress(session_id)
