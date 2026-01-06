"""Pydantic request and response schemas for the AI Text API."""
from typing import List, Literal

from pydantic import BaseModel, Field


class TextRequest(BaseModel):
    """Request body containing a single text field."""

    text: str = Field(..., min_length=1, description="Input text to process")


class AnalyzeResponse(BaseModel):
    """Response for the /analyze endpoint."""

    sentiment: Literal["positive", "negative", "neutral"]
    keywords: List[str]


class SummaryResponse(BaseModel):
    """Response for the /summarize endpoint."""

    summary: str


class SemanticSearchRequest(BaseModel):
    """Request for semantic search queries."""

    query: str = Field(..., min_length=1, description="Query text to search against the corpus")
    top_k: int = Field(default=3, ge=1, le=10, description="Number of top results to return")


class SemanticSearchResponse(BaseModel):
    """Response containing semantic search results."""

    results: List[str]


class AddCorpusRequest(BaseModel):
    """Request to dynamically add texts to the semantic search corpus."""

    texts: List[str] = Field(..., min_items=1, description="Texts to add to the corpus")


class AddCorpusResponse(BaseModel):
    """Response confirming texts were added to corpus."""

    count: int
    total_corpus_size: int


class RealtimeAnalysisMessage(BaseModel):
    """Message format for real-time WebSocket analysis."""

    type: Literal["sentiment", "keywords", "full_analysis", "error"]
    data: dict
    timestamp: str
