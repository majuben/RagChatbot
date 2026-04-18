"""
Simple Streamlit frontend for the RAG chatbot.

This app connects to the FastAPI endpoints to ingest text and ask questions.
"""

import os
import streamlit as st
import requests

st.set_page_config(page_title="RAG Chatbot UI", layout="centered")

API_URL = "http://web:8000/api/chat"


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

    api_url = st.text_input("API URL", value=API_URL or "http://web:8000/api/chat")
    st.caption("Assurez-vous que votre backend FastAPI est démarré et accessible à cette URL.")

    st.divider()
    st.subheader("1. Ingestion de fichier")
    uploaded_file = st.file_uploader("Téléchargez un fichier PDF ou DOCX", type=["pdf", "docx"])
    if st.button("Ingest file"):
        if uploaded_file is None:
            st.warning("Veuillez sélectionner un fichier PDF ou DOCX.")
        else:
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                response = requests.post(f"{api_url}/ingest/file", files=files, timeout=60)
                response.raise_for_status()
                st.success("Fichier ingéré avec succès.")
                st.json(response.json())
            except Exception as exc:
                st.error(f"Erreur d'ingestion de fichier : {exc}")
    
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




if __name__ == "__main__":
    main()
