# RagChatbot

Projet de démonstration RAG avec FastAPI, LangChain, Ollama et PostgreSQL.

## Architecture

- `app/main.py` : point d'entrée FastAPI.
- `app/api/routes/chat.py` : endpoints pour l'ingestion de texte et les réponses de chat.
- `app/services/rag.py` : logique RAG utilisant Ollama et PGVector.
- `app/services/ollama.py` : configuration du LLM et des embeddings Ollama.
- `docker-compose.yml` : services `web`, `db` et `ollama`.

## Installation

1. Copier `.env` et mettre à jour si nécessaire.
2. Lancer `docker compose up --build`.
3. Accéder à `http://localhost:8000`.

## Endpoints

- `POST /api/chat/ingest` : ingérer du texte dans la base RAG.
- `POST /api/chat/respond` : poser une question et recevoir une réponse contextuelle.

## Exemple de payload

```json
{
  "text": "Voici un document d'exemple à stocker."
}
```

```json
{
  "question": "Que peut-on dire sur ce document ?"
}
```
