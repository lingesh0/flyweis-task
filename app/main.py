"""FastAPI entrypoint for the AI-Powered Text Intelligence API."""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import Depends, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

from app.embeddings import (
    SemanticSearchEngine,
    add_texts_to_corpus,
    get_corpus_size,
    get_semantic_engine,
)
from app.keywords import extract_keywords, get_spacy_model
from app.schemas import (
    AddCorpusRequest,
    AddCorpusResponse,
    AnalyzeResponse,
    RealtimeAnalysisMessage,
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
    description="Production-ready NLP/LLM powered text intelligence service with real-time support.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
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

    results = await asyncio.to_thread(engine.search, query, payload.top_k)
    return SemanticSearchResponse(results=results)


@app.post("/corpus/add", response_model=AddCorpusResponse)
async def add_corpus(payload: AddCorpusRequest) -> AddCorpusResponse:
    """Dynamically add texts to the semantic search corpus for real-time updates."""

    texts = [t.strip() for t in payload.texts if t.strip()]
    if not texts:
        raise HTTPException(status_code=400, detail="At least one non-empty text required.")

    total_size = await asyncio.to_thread(add_texts_to_corpus, texts)
    return AddCorpusResponse(count=len(texts), total_corpus_size=total_size)


@app.get("/corpus/size")
async def corpus_size() -> dict:
    """Get current semantic search corpus size."""

    size = await asyncio.to_thread(get_corpus_size)
    return {"corpus_size": size}


@app.websocket("/ws/analyze")
async def websocket_analyze(websocket: WebSocket) -> None:
    """Real-time sentiment analysis and keyword extraction via WebSocket.

    Client sends: {"text": "your text here"}
    Server responds with streaming sentiment and keywords.
    """

    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            text = data.get("text", "").strip()

            if not text:
                await websocket.send_json(
                    {
                        "type": "error",
                        "data": {"message": "Text must not be empty."},
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                )
                continue

            sentiment_task = asyncio.to_thread(analyze_sentiment, text)
            keywords_task = asyncio.to_thread(extract_keywords, text)

            sentiment, keywords = await asyncio.gather(
                sentiment_task, keywords_task, return_exceptions=True
            )

            await websocket.send_json(
                {
                    "type": "sentiment",
                    "data": {"sentiment": sentiment},
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

            await websocket.send_json(
                {
                    "type": "keywords",
                    "data": {"keywords": keywords},
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )

    except WebSocketDisconnect:
        pass
    except Exception as e:
        await websocket.send_json(
            {
                "type": "error",
                "data": {"message": str(e)},
                "timestamp": datetime.utcnow().isoformat(),
            }
        )
