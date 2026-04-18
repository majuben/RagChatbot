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
2. Si Ollama tourne déjà sur votre machine, vérifiez que `OLLAMA_URL` pointe vers `http://host.docker.internal:11434`.
3. Lancer `docker compose up --build`.
4. Accéder à `http://localhost:8000`.

> Note: avec Docker, vous n'avez pas besoin d'installer les dépendances Python localement. Le conteneur installe tout automatiquement.
> Si vous voulez exécuter le frontend en local, assurez-vous d'abord de mettre à jour pip et setuptools, puis installez les dépendances :
>
> ```bash
> python -m pip install --upgrade pip setuptools wheel
> pip install -r requirements.txt
> ```
>
> Sous Windows, l'erreur `numpy` peut indiquer qu'il manque les outils de compilation ou que l'environnement Python n'a pas de roues compatibles.

## Endpoints

- `POST /api/chat/ingest` : ingérer du texte dans la base RAG.
- `POST /api/chat/ingest/file` : ingérer un fichier PDF ou DOCX.
- `POST /api/chat/respond` : poser une question et recevoir une réponse contextuelle.

## Frontend Streamlit

Un frontend simple est disponible dans `frontend/streamlit_app.py`.

Pour lancer ensemble le backend et le frontend avec Docker :

```bash
docker compose up --build
```

Le backend sera exposé sur `http://localhost:8000` et l'interface Streamlit sur `http://localhost:8501`.

Si vous exécutez le frontend en local sans Docker, utilisez :

```bash
pip install -r requirements.txt
streamlit run frontend/streamlit_app.py
```