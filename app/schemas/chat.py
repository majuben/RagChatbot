"""
Pydantic request and response schemas for the RAG chatbot.
"""

from typing import List
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=1, description="The user's question")


class ChatResponse(BaseModel):
    question: str
    answer: str


class IngestRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Document text to ingest")


class BatchIngestRequest(BaseModel):
    texts: List[str] = Field(..., min_items=1, description="List of document texts")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
