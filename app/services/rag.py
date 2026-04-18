"""
RAG service implementation connecting Ollama and PostgreSQL.
This module handles the complete RAG pipeline:
- Document chunking and ingestion
- Vector embedding and indexing
- Similarity retrieval
- Reranking
- Context-aware LLM response generation
"""

import io
import os
from typing import List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaLLM as Ollama
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import PGVector
import fitz
import docx

from .ollama import build_llm, build_embeddings

VECTORSTORE_TABLE_NAME = os.getenv("VECTORSTORE_TABLE_NAME", "rag_documents")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/ragbot")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))
TOP_K_RETRIEVAL = int(os.getenv("TOP_K_RETRIEVAL", "5"))
TOP_K_FINAL = int(os.getenv("TOP_K_FINAL", "3"))
SUPPORTED_FILE_TYPES = [".pdf", ".docx"]


def _build_embeddings() -> OllamaEmbeddings:
    """Build Ollama embeddings connector using a dedicated embedding model."""
    return build_embeddings()


def _build_llm() -> Ollama:
    """Build Ollama LLM connector."""
    return build_llm()


def _get_vectorstore() -> PGVector:
    """Get or create PGVector store."""
    embeddings = _build_embeddings()
    return PGVector(
        collection_name=VECTORSTORE_TABLE_NAME,
        connection_string=DATABASE_URL,
        embedding_function=embeddings
    )


def _extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF file's byte stream."""
    with fitz.open(stream=file_bytes, filetype="pdf") as document:
        return "\n\n".join(page.get_text() for page in document)


def _extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file's byte stream."""
    document = docx.Document(io.BytesIO(file_bytes))
    return "\n\n".join(paragraph.text for paragraph in document.paragraphs if paragraph.text)


def _extract_text_from_file(file_name: str, file_bytes: bytes) -> str:
    """Detect file type and extract text from PDF or DOCX."""
    file_name_lower = file_name.lower()
    if file_name_lower.endswith(".pdf"):
        return _extract_text_from_pdf(file_bytes)
    if file_name_lower.endswith(".docx"):
        return _extract_text_from_docx(file_bytes)
    raise ValueError("Unsupported file type. Only PDF and DOCX are supported.")


def chunk_documents(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    # On définit des séparateurs logiques pour ne pas couper n'importe où
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ".", " ", ""],
        strip_whitespace=True
    )
    chunks = splitter.split_text(text)
    return [c.strip() for c in chunks if c and len(c.strip()) > 10] # On ignore les chunks vides ou trop courts


def retrieve_chunks(question: str, top_k: int = TOP_K_RETRIEVAL) -> List[str]:
    """
    Récupère les chunks avec leurs scores de distance pour le débug.
    """
    vectorstore = _get_vectorstore()
    
    # On utilise similarity_search_with_score au lieu de similarity_search
    # Retourne une liste de tuples (Document, score)
    results_with_scores = vectorstore.similarity_search_with_score(question, k=top_k)
    
    print(f"\n--- DEBUG RETRIEVAL (Top {top_k}) ---")
    for i, (doc, score) in enumerate(results_with_scores):
        # Plus le score est petit, plus le chunk est pertinent (distance)
        print(f"Rank {i+1} | Score (Distance): {score:.4f} | Preview: {doc.page_content[:70]}...")
    print("------------------------------------\n")

    return [doc.page_content for doc, score in results_with_scores]


def rerank_chunks(question: str, chunks: List[str], top_k: int = TOP_K_FINAL) -> List[str]:
    """
    Rerank retrieved chunks by semantic relevance (simple scoring for now).
    Returns:
        Top-k reranked chunks.
    """
    embeddings = _build_embeddings()
    query_embedding = embeddings.embed_query(question)
    
    scored_chunks = []
    for chunk in chunks:
        chunk_embedding = embeddings.embed_query(chunk)
        score = sum(a * b for a, b in zip(query_embedding, chunk_embedding)) / (
            (sum(a**2 for a in query_embedding)**0.5 * sum(b**2 for b in chunk_embedding)**0.5) + 1e-6
        )
        scored_chunks.append((chunk, score))
    
    ranked = sorted(scored_chunks, key=lambda x: x[1], reverse=True)
    return [chunk for chunk, _ in ranked[:top_k]]


def build_context(chunks: List[str]) -> str:
    """
    Assemble chunks into a coherent context for the LLM.
    Returns:
        Assembled context string.
    """
    return "\n\n".join([f"[Document {i+1}]\n{chunk}" for i, chunk in enumerate(chunks)])

def generate_response(question: str, context: str) -> str:
    """
    Génère une réponse LLM basée sur un contexte structuré.
    """
    llm = _build_llm()
    
    # Un prompt plus directif pour éviter que le modèle ne "décroche" sur les longs contextes
    prompt = f"""Tu es un assistant technique précis. Analyse les extraits de documents fournis ci-dessous pour répondre à la question.

### RÈGLES :
1. Utilise UNIQUEMENT les informations du contexte fourni.
2. Si la réponse n'est pas dans le contexte, réponds : "Je ne trouve pas assez d'informations dans les documents pour répondre précisément."
3. Structure ta réponse de manière logique et cite les [Document X] si nécessaire.

### CONTEXTE DES DOCUMENTS :
{context}

### QUESTION DE L'UTILISATEUR :
{question}

### RÉPONSE DE L'ASSISTANT :"""

    response = llm.invoke(prompt)
    return response.strip()

def ingest(texts: List[str], source: Optional[str] = None) -> None:
    """
    Ingest and index documents.
    1. Chunk each text
    2. Create Document objects
    3. Add to vector store
    """
    all_documents = []
    for text in texts:
        chunks = chunk_documents(text)
        for chunk in chunks:
            metadata = {"source": source} if source else {}
            all_documents.append(Document(page_content=chunk, metadata=metadata))

    if all_documents:
        vectorstore = _get_vectorstore()
        vectorstore.add_documents(all_documents)


def ingest_file(file_name: str, file_bytes: bytes) -> str:
    """Extract text from a supported file and ingest it into the vector store."""
    text = _extract_text_from_file(file_name, file_bytes)
    if not text or not text.strip():
        raise ValueError("No readable text found in uploaded file.")
    ingest([text], source=file_name)
    return text


def respond(question: str) -> str:
    """
    Pipeline RAG simplifié : Récupération directe -> Construction du contexte -> Génération.
    """
    
    retrieved_chunks = retrieve_chunks(question, top_k=TOP_K_RETRIEVAL)
    context = build_context(retrieved_chunks)
    response = generate_response(question, context)
    return response


class RAGService:
    """High-level RAG service for API integration."""

    @staticmethod
    def ingest(texts: List[str], source: Optional[str] = None) -> None:
        """Ingest documents into the RAG system."""
        ingest(texts, source=source)

    @staticmethod
    def ingest_file(file_name: str, file_bytes: bytes) -> str:
        """Ingest a file into the RAG system."""
        return ingest_file(file_name, file_bytes)

    @staticmethod
    def respond(question: str) -> str:
        """Generate a response using the RAG pipeline."""
        return respond(question)


rag_service = RAGService()