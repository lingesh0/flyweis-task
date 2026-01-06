"""FastAPI entrypoint for the AI-Powered Text Intelligence API."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException

from app.embeddings import SemanticSearchEngine, get_semantic_engine
from app.keywords import extract_keywords, get_spacy_model
from app.schemas import (
    AnalyzeResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    SummaryResponse,
    TextRequest,
)
from app.sentiment import analyze_sentiment, get_sentiment_analyzer
from app.summarize import get_summarizer, summarize_text


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[override]
    """Warm critical models on startup to reduce first-request latency."""

    await asyncio.gather(
        asyncio.to_thread(get_sentiment_analyzer),
        asyncio.to_thread(get_spacy_model),
        asyncio.to_thread(get_summarizer),
        asyncio.to_thread(get_semantic_engine),
    )
    yield


app = FastAPI(
    title="AI Text Intelligence API",
    version="1.0.0",
    description="Production-ready NLP/LLM powered text intelligence service.",
    lifespan=lifespan,
)


@app.get("/health")
async def health() -> dict:
    """Simple health check endpoint."""

    return {"status": "ok"}


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(payload: TextRequest) -> AnalyzeResponse:
    """Analyze sentiment and extract keywords from text."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    sentiment_task = asyncio.to_thread(analyze_sentiment, text)
    keywords_task = asyncio.to_thread(extract_keywords, text)
    sentiment, keywords = await asyncio.gather(sentiment_task, keywords_task)

    return AnalyzeResponse(sentiment=sentiment, keywords=keywords)


@app.post("/summarize", response_model=SummaryResponse)
async def summarize(payload: TextRequest) -> SummaryResponse:
    """Summarize the provided text using a T5 model."""

    text = payload.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Text must not be empty.")

    summary = await asyncio.to_thread(summarize_text, text)
    return SummaryResponse(summary=summary)


@app.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    payload: SemanticSearchRequest,
    engine: SemanticSearchEngine = Depends(get_semantic_engine),
) -> SemanticSearchResponse:
    """Perform semantic search over the in-memory corpus."""

    query = payload.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query must not be empty.")

    results = await asyncio.to_thread(engine.search, query, 3)
    return SemanticSearchResponse(results=results)
