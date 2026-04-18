"""
FastAPI endpoints for chat and document ingestion.

These endpoints are the public REST interface for the RAG chatbot.
"""

from fastapi import APIRouter, HTTPException
from app.schemas.chat import ChatRequest, ChatResponse, IngestRequest
from app.services.rag import rag_service

router = APIRouter()

@router.post("/respond", response_model=ChatResponse)
async def respond(request: ChatRequest):
    try:
        answer = rag_service.respond(request.question)
        return ChatResponse(question=request.question, answer=answer)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))

@router.post("/ingest")
async def ingest(request: IngestRequest):
    try:
        rag_service.ingest([request.text])
        return {"status": "ok", "ingested_text": request.text}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
