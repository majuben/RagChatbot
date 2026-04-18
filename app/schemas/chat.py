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


class FileIngestResponse(BaseModel):
    status: str = Field(..., description="Ingestion status")
    message: str = Field(..., description="Result message")
    filename: str = Field(..., description="Uploaded file name")
    size_bytes: int = Field(..., description="Uploaded file size in bytes")
    extracted_chars: int = Field(..., description="Number of extracted characters")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    message: str = Field(..., description="Status message")
