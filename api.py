"""
api.py
───────
FastAPI backend.

    GET  /              -> static/index.html test frontend
    GET  /health         -> liveness check
    POST /analyze        -> full analysis (see pipeline.analyze_article);
                             accepts article_text OR url, persists the
                             result and returns its analysis_id
    GET  /history         -> recent analyses (summary rows)
    GET  /history/{id}    -> one full stored analysis
    POST /feedback        -> reader feedback tied to an analysis_id
    GET  /analytics        -> aggregate trends for the dashboard
    GET  /survey           -> static/study.html reader-study survey page
    /study/*               -> curated reader-study endpoints (see
                              study/api_routes.py): add articles,
                              sessions, next-article delivery, ratings,
                              behavior events, progress
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

from core.article_fetch import ArticleFetchError, fetch_article_from_url
from pipeline import analyze_article
from study import database as db
from study.api_routes import study_router

app = FastAPI(
    title="Machine Credibility API",
    description="Credibility, readability, engagement and explainability analysis of news articles.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local development only - restrict before any real deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _startup() -> None:
    db.init_db()


class AnalyzeRequest(BaseModel):
    article_text: Optional[str] = Field(default=None, description="Raw article text.")
    url: Optional[str] = Field(default=None, description="News article URL; fetched and extracted if article_text is omitted.")
    claim_threshold: float = Field(default=0.65, ge=0.0, le=1.0)
    title: Optional[str] = Field(default=None, description="Optional headline (overrides any fetched title).")
    include_llm_judge: bool = Field(default=True)
    max_claims_to_verify: int = Field(
        default=15, ge=1, le=100,
        description="Cap on how many check-worthy claims get evidence retrieval + verdict scoring "
                     "per run (the slow, per-claim stage). Articles with more claims than this still "
                     "get them listed, just unverified, rather than taking several minutes."
    )

    @model_validator(mode="after")
    def _require_text_or_url(self) -> "AnalyzeRequest":
        text_present = bool(self.article_text and self.article_text.strip())
        url_present = bool(self.url and self.url.strip())
        if not text_present and not url_present:
            raise ValueError("Provide either 'article_text' or 'url'.")
        return self


class FeedbackRequest(BaseModel):
    analysis_id: str
    credibility: int = Field(..., ge=1, le=5)
    clarity: int = Field(..., ge=1, le=5)
    neutrality: int = Field(..., ge=1, le=5)
    complexity: int = Field(..., ge=1, le=5)
    coherence: int = Field(..., ge=1, le=5)
    quality: int = Field(..., ge=1, le=5)
    session_id: Optional[str] = None
    comment: Optional[str] = None


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/", summary="Serve test interface")
def serve_index() -> FileResponse:
    index_path = os.path.join(os.path.dirname(__file__), "static", "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=404, detail="static/index.html not found.")
    return FileResponse(index_path)


@app.get("/survey", summary="Serve the reader-study survey page")
def serve_survey() -> FileResponse:
    survey_path = os.path.join(os.path.dirname(__file__), "static", "study.html")
    if not os.path.exists(survey_path):
        raise HTTPException(status_code=404, detail="static/study.html not found.")
    return FileResponse(survey_path)


app.include_router(study_router)


@app.post("/analyze", summary="Analyze article text or a URL")
def analyze_endpoint(payload: AnalyzeRequest) -> Dict[str, Any]:
    source_url: Optional[str] = None
    title = payload.title

    if payload.article_text and payload.article_text.strip():
        article_text = payload.article_text.strip()
    else:
        try:
            fetched = fetch_article_from_url(payload.url)
        except ArticleFetchError as e:
            raise HTTPException(status_code=400, detail=str(e))
        article_text = fetched.text
        source_url = fetched.url
        if not title:
            title = fetched.title

    try:
        result = analyze_article(
            article_text=article_text,
            claim_threshold=payload.claim_threshold,
            title=title,
            include_llm_judge=payload.include_llm_judge,
            max_claims_to_verify=payload.max_claims_to_verify,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Pipeline error: {type(e).__name__}: {e}")

    try:
        analysis_id = db.save_analysis(
            title=title, source_url=source_url, article_text=article_text, result=result
        )
        result["analysis_id"] = analysis_id
        result["source_url"] = source_url
    except Exception as e:  # noqa: BLE001
        # Persistence failing shouldn't lose the analysis the person is waiting on.
        print(f"  [api] warning: could not persist analysis: {type(e).__name__}: {e}")
        result["analysis_id"] = None
        result["source_url"] = source_url

    return result


@app.get("/history", summary="List recent analyses")
def history_endpoint(limit: int = 50) -> List[Dict[str, Any]]:
    limit = max(1, min(limit, 200))
    return db.list_analyses(limit=limit)


@app.get("/history/{analysis_id}", summary="Get one full stored analysis")
def history_detail_endpoint(analysis_id: str) -> Dict[str, Any]:
    record = db.get_analysis(analysis_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No analysis found with id '{analysis_id}'.")
    record["feedback"] = db.get_feedback_for_analysis(analysis_id)
    return record


@app.post("/feedback", summary="Submit reader feedback on an analysis")
def feedback_endpoint(payload: FeedbackRequest) -> Dict[str, Any]:
    try:
        feedback_id = db.add_feedback(
            analysis_id=payload.analysis_id,
            credibility=payload.credibility,
            clarity=payload.clarity,
            neutrality=payload.neutrality,
            complexity=payload.complexity,
            coherence=payload.coherence,
            quality=payload.quality,
            session_id=payload.session_id,
            comment=payload.comment,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"feedback_id": feedback_id, "status": "saved"}


@app.get("/analytics", summary="Aggregate trends across stored analyses + feedback")
def analytics_endpoint() -> Dict[str, Any]:
    return db.get_analytics_summary()


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=False)
