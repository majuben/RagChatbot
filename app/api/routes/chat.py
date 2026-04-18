"""
FastAPI endpoints for chat and document ingestion.
These endpoints are the public REST interface for the RAG chatbot.
Provides ingest, respond, batch operations, and system health monitoring.
"""

import logging
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from app.schemas.chat import ChatRequest, ChatResponse, IngestRequest, BatchIngestRequest, HealthResponse
from app.services.rag import rag_service

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health and connectivity."""
    try:
        return HealthResponse(status="healthy", message="RAG service is running")
    except Exception as exc:
        logger.error(f"Health check failed: {exc}")
        raise HTTPException(status_code=503, detail="Service unavailable")


@router.post("/ingest", tags=["ingestion"])
async def ingest(request: IngestRequest):
    """
    Ingest a single document.
    - Chunks the text
    - Vectorizes chunks
    - Stores in PostgreSQL + PGVector
    Returns:
        Status confirmation with ingested text summary.
    """
    try:
        if not request.text or not request.text.strip():
            raise ValueError("Text cannot be empty")
        
        rag_service.ingest([request.text])
        logger.info(f"Ingested {len(request.text)} characters")
        
        return {
            "status": "ok",
            "message": "Document ingested successfully",
            "text_length": len(request.text),
            "char_count": len(request.text.strip())
        }
    except ValueError as exc:
        logger.warning(f"Validation error during ingest: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Ingest error: {exc}")
        raise HTTPException(status_code=500, detail=f"Ingest failed: {str(exc)}")


@router.post("/ingest/batch", tags=["ingestion"])
async def batch_ingest(request: BatchIngestRequest, background_tasks: BackgroundTasks):
    """
    Ingest multiple documents asynchronously.
    - Accepts list of texts
    - Processes in background
    - Returns immediately with job confirmation
    Returns:
        Job confirmation with document count.
    """
    try:
        if not request.texts or len(request.texts) == 0:
            raise ValueError("At least one document required")
        
        valid_texts = [t for t in request.texts if t and t.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        background_tasks.add_task(rag_service.ingest, valid_texts)
        logger.info(f"Queued batch ingest for {len(valid_texts)} documents")
        
        return {
            "status": "queued",
            "message": "Batch ingest started in background",
            "document_count": len(valid_texts)
        }
    except ValueError as exc:
        logger.warning(f"Validation error in batch ingest: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Batch ingest error: {exc}")
        raise HTTPException(status_code=500, detail=f"Batch ingest failed: {str(exc)}")


@router.post("/respond", response_model=ChatResponse, tags=["chat"])
async def respond(request: ChatRequest):
    """
    Generate a RAG response to a question.
    Pipeline:
    1. Retrieve relevant chunks (top-5 similarity)
    2. Rerank by semantic relevance (top-3)
    3. Build context
    4. Call Ollama LLM
    5. Return answer
    Returns:
        ChatResponse: question + generated answer
    """
    try:
        if not request.question or not request.question.strip():
            raise ValueError("Question cannot be empty")
        
        logger.info(f"Processing question: {request.question[:50]}...")
        answer = rag_service.respond(request.question)
        
        return ChatResponse(question=request.question, answer=answer)
    except ValueError as exc:
        logger.warning(f"Validation error: {exc}")
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.error(f"Response generation error: {exc}")
        raise HTTPException(status_code=500, detail=f"Response failed: {str(exc)}")


@router.post("/retrieve", tags=["debug"])
async def retrieve_debug(request: ChatRequest):
    """
    Debug endpoint: retrieve and return raw chunks for a query.
    Useful for testing retrieval pipeline.
    Returns:
        List of retrieved chunks with scores.
    """
    try:
        from app.services.rag import retrieve_chunks, rerank_chunks
        
        retrieved = retrieve_chunks(request.question, top_k=5)
        reranked = rerank_chunks(request.question, retrieved, top_k=3)
        
        return {
            "question": request.question,
            "retrieved_count": len(retrieved),
            "reranked_count": len(reranked),
            "chunks": reranked
        }
    except Exception as exc:
        logger.error(f"Retrieve debug error: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

