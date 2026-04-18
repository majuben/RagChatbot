"""
Simple Streamlit frontend for the RAG chatbot.

This app connects to the FastAPI endpoints to ingest text and ask questions.
"""

import os
import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot UI", layout="centered")

DEFAULT_API_URL = os.getenv("API_URL", "http://localhost:8000/api/chat")


def post_request(endpoint: str, payload: dict) -> dict:
    url = f"{endpoint}"
    response = requests.post(url, json=payload, timeout=30)
    response.raise_for_status()
    return response.json()


def main():
    st.title("RAG Chatbot")
    st.markdown(
        "Cette interface Streamlit se connecte à l'API FastAPI pour ingérer du texte et poser des questions au modèle RAG."
    )

    api_url = st.text_input("API URL", value=DEFAULT_API_URL)
    st.caption("Assurez-vous que votre backend FastAPI est démarré et accessible à cette URL.")

    st.divider()
    st.subheader("1. Ingestion de document")
    ingest_text = st.text_area("Texte à ingérer", height=200)
    if st.button("Ingest"):
        if not ingest_text.strip():
            st.warning("Le texte d'ingestion ne peut pas être vide.")
        else:
            try:
                result = post_request(f"{api_url}/ingest", {"text": ingest_text})
                st.success("Document ingéré avec succès.")
                st.json(result)
            except Exception as exc:
                st.error(f"Erreur d'ingestion : {exc}")

    st.divider()
    st.subheader("2. Poser une question")
    question = st.text_input("Question", value="Qu'est-ce que le RAG ?")
    if st.button("Poser la question"):
        if not question.strip():
            st.warning("La question ne peut pas être vide.")
        else:
            try:
                answer = post_request(f"{api_url}/respond", {"question": question})
                st.success("Réponse reçue")
                st.markdown(f"**Question :** {answer.get('question')}\n\n**Réponse :** {answer.get('answer')}")
            except Exception as exc:
                st.error(f"Erreur de réponse : {exc}")

    st.divider()
    st.subheader("3. Debug retrieval")
    debug_question = st.text_input("Question de debug récupération", value="Explique RAG en une phrase.")
    if st.button("Voir les chunks"):
        if not debug_question.strip():
            st.warning("La question de debug ne peut pas être vide.")
        else:
            try:
                result = post_request(f"{api_url}/retrieve", {"question": debug_question})
                st.success("Chunks récupérés")
                st.json(result)
            except Exception as exc:
                st.error(f"Erreur debug retrieval : {exc}")


if __name__ == "__main__":
    main()
