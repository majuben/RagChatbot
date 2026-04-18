"""
RAG service implementation connecting Ollama and PostgreSQL.

This module exposes lightweight ingest and respond helpers for the chatbot REST API.
"""

import os
from typing import List
from langchain.chains import RetrievalQA
from langchain.docstore.document import Document
from langchain.embeddings import OllamaEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import PGVector

VECTORSTORE_TABLE_NAME = os.getenv("VECTORSTORE_TABLE_NAME", "rag_documents")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragbot")


def _build_embeddings() -> OllamaEmbeddings:
    return OllamaEmbeddings(
        model=os.getenv("OLLAMA_MODEL", "llama2"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )


def _build_llm() -> Ollama:
    return Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama2"),
        base_url=os.getenv("OLLAMA_URL", "http://localhost:11434"),
    )


def _get_vectorstore() -> PGVector:
    embeddings = _build_embeddings()
    return PGVector.from_documents(
        documents=[],
        embedding=embeddings,
        connection_string=DATABASE_URL,
        table_name=VECTORSTORE_TABLE_NAME,
        text_key="text",
    )


def ingest(texts: List[str]) -> None:
    vectorstore = _get_vectorstore()
    documents = [Document(page_content=text.strip()) for text in texts if text and text.strip()]
    if documents:
        vectorstore.add_documents(documents)


def respond(question: str) -> str:
    vectorstore = _get_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = _build_llm()
    qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type="stuff")
    return qa.run(question)


class RAGService:
    @staticmethod
    def ingest(texts: List[str]) -> None:
        ingest(texts)

    @staticmethod
    def respond(question: str) -> str:
        return respond(question)


rag_service = RAGService()
