"""
FastAPI application entrypoint for the RAG chatbot service.
"""

from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes.chat import router as chat_router

load_dotenv()

app = FastAPI(
    title="RAG Chatbot",
    description="FastAPI endpoints for a LangChain RAG chatbot using Ollama and PostgreSQL.",
)

app.include_router(chat_router, prefix="/api/chat", tags=["chat"])

@app.get("/")
async def root():
    return {
        "message": "RAG Chatbot is running.",
        "routes": ["/api/chat/ingest", "/api/chat/respond"],
    }
