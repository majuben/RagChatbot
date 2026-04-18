"""
Ollama client helpers for the RAG chatbot.

This module builds LangChain connectors for an Ollama server that provides both the
large language model and the embeddings endpoint.
"""

import os
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM, OllamaEmbeddings


load_dotenv()

def get_ollama_model() -> str:
   
    return os.getenv("OLLAMA_MODEL", "qwen3:8b")

def get_ollama_embedding_model() -> str:
    # Dedicated embedding model
    return os.getenv("OLLAMA_EMBEDDING_MODEL", "nomic-embed-text")

def get_ollama_url() -> str:
    return os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")

def build_llm() -> OllamaLLM:
    return OllamaLLM(
        model=get_ollama_model(), 
        base_url=get_ollama_url()
    )

def build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=get_ollama_embedding_model(),
        base_url=get_ollama_url()
    )