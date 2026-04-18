"""Simple Python client for testing the RAG chatbot API.
"""

import requests
import json
from typing import Optional

BASE_URL = "http://localhost:8000/api/chat"


class RAGChatbotClient:
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = requests.Session()

    def health(self) -> dict:
        """Check if the service is healthy."""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def ingest(self, text: str) -> dict:
        """Ingest a single document."""
        payload = {"text": text}
        response = self.session.post(f"{self.base_url}/ingest", json=payload)
        response.raise_for_status()
        return response.json()

    def batch_ingest(self, texts: list) -> dict:
        """Ingest multiple documents."""
        payload = {"texts": texts}
        response = self.session.post(f"{self.base_url}/ingest/batch", json=payload)
        response.raise_for_status()
        return response.json()

    def ask(self, question: str) -> dict:
        """Ask a question and get a response."""
        payload = {"question": question}
        response = self.session.post(f"{self.base_url}/respond", json=payload)
        response.raise_for_status()
        return response.json()

    def retrieve_debug(self, question: str) -> dict:
        """Debug: retrieve and show relevant chunks."""
        payload = {"question": question}
        response = self.session.post(f"{self.base_url}/retrieve", json=payload)
        response.raise_for_status()
        return response.json()


def demo():
    """Run a demo of the client."""
    client = RAGChatbotClient()

    print("=" * 60)
    print("RAG Chatbot Client Demo")
    print("=" * 60)

    print("\n[1] Health Check")
    try:
        health = client.health()
        print(f"✓ Status: {health['status']}")
        print(f"  Message: {health['message']}")
    except Exception as e:
        print(f"✗ Error: {e}")
        print("  Is the server running? (docker compose up --build)")
        return

    print("\n[2] Ingesting Documents")
    docs = [
        "Machine learning is a subset of artificial intelligence. It focuses on enabling computers to learn from data without being explicitly programmed.",
        "Python is a popular programming language for data science and machine learning. Libraries like scikit-learn and TensorFlow are widely used.",
        "RAG stands for Retrieval-Augmented Generation. It combines document retrieval with language models to provide accurate, context-aware responses.",
    ]

    for i, doc in enumerate(docs, 1):
        try:
            result = client.ingest(doc)
            print(f"✓ Document {i}: {result['char_count']} chars ingested")
        except Exception as e:
            print(f"✗ Document {i} error: {e}")

    print("\n[3] Debug Retrieval")
    test_query = "What is machine learning?"
    try:
        debug_result = client.retrieve_debug(test_query)
        print(f"Question: {debug_result['question']}")
        print(f"Retrieved chunks: {debug_result['retrieved_count']}")
        print(f"Reranked chunks: {debug_result['reranked_count']}")
        print("\nTop reranked chunk:")
        if debug_result['chunks']:
            print(f"  {debug_result['chunks'][0][:100]}...")
    except Exception as e:
        print(f"✗ Error: {e}")

    print("\n[4] Asking Questions")
    questions = [
        "What is machine learning?",
        "Why is Python popular for ML?",
        "Explain RAG in a sentence.",
    ]

    for question in questions:
        try:
            result = client.ask(question)
            print(f"\nQ: {result['question']}")
            print(f"A: {result['answer'][:150]}...")
        except Exception as e:
            print(f"✗ Error: {e}")

    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
