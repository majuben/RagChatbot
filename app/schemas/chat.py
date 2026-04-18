"""
Pydantic request and response schemas for the RAG chatbot.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    question: str
    answer: str


class IngestRequest(BaseModel):
    text: str
