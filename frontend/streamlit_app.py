"""
RAG Chatbot — Interface Streamlit Premium
Design: Dark Luxury avec accents ambrés, typographie soignée, animations CSS
"""

import streamlit as st
import requests
import time
from datetime import datetime

# ─── Configuration de la page ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Assistant",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS Global — Design System ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300&family=DM+Mono:wght@300;400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & Root Variables ── */
:root {
    --bg-base:       #0a0908;
    --bg-surface:    #111009;
    --bg-card:       #161410;
    --bg-hover:      #1e1b14;
    --amber:         #c8922a;
    --amber-light:   #e8b35a;
    --amber-muted:   #8a6320;
    --amber-glow:    rgba(200, 146, 42, 0.15);
    --text-primary:  #f0e8d8;
    --text-secondary:#a89880;
    --text-muted:    #5c5040;
    --border:        rgba(200, 146, 42, 0.18);
    --border-strong: rgba(200, 146, 42, 0.40);
    --radius:        6px;
    --radius-lg:     12px;
    --font-display:  'Cormorant Garamond', Georgia, serif;
    --font-body:     'DM Sans', sans-serif;
    --font-mono:     'DM Mono', monospace;
    --shadow:        0 8px 40px rgba(0,0,0,0.6);
    --shadow-amber:  0 0 30px rgba(200, 146, 42, 0.12);
}

/* ── App Background ── */
.stApp {
    background-color: var(--bg-base);
    background-image:
        radial-gradient(ellipse 80% 50% at 50% -20%, rgba(200,146,42,0.06) 0%, transparent 60%),
        repeating-linear-gradient(
            0deg,
            transparent,
            transparent 80px,
            rgba(200,146,42,0.015) 80px,
            rgba(200,146,42,0.015) 81px
        );
    font-family: var(--font-body);
    color: var(--text-primary);
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding: 2rem 3rem 4rem 3rem;
    max-width: 1400px;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg-surface) !important;
    border-right: 1px solid var(--border) !important;
    box-shadow: 4px 0 30px rgba(0,0,0,0.4);
}
[data-testid="stSidebar"] .block-container {
    padding: 2rem 1.5rem;
}

/* ── Sidebar Brand ── */
.sidebar-brand {
    text-align: center;
    padding: 1.5rem 0 2rem 0;
    border-bottom: 1px solid var(--border);
    margin-bottom: 2rem;
}
.sidebar-brand .logo {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 300;
    color: var(--amber);
    line-height: 1;
    text-shadow: 0 0 40px rgba(200,146,42,0.4);
    letter-spacing: 0.05em;
}
.sidebar-brand .tagline {
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-muted);
    letter-spacing: 0.3em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

/* ── Sidebar Section Labels ── */
.sidebar-section {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: var(--amber-muted);
    margin: 1.5rem 0 0.75rem 0;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid var(--border);
}

/* ── Status Badge ── */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-secondary);
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.3rem 0.8rem;
    margin-top: 0.5rem;
}
.status-dot {
    width: 6px;
    height: 6px;
    border-radius: 50%;
    background: #4ade80;
    box-shadow: 0 0 8px #4ade80;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.6; transform: scale(0.85); }
}

/* ── Page Header ── */
.page-header {
    margin-bottom: 2.5rem;
    padding-bottom: 1.5rem;
    border-bottom: 1px solid var(--border);
}
.page-header h1 {
    font-family: var(--font-display);
    font-size: 3rem;
    font-weight: 300;
    font-style: italic;
    color: var(--text-primary);
    letter-spacing: -0.01em;
    line-height: 1.1;
    margin: 0 0 0.3rem 0;
}
.page-header h1 span {
    color: var(--amber);
}
.page-header p {
    font-family: var(--font-body);
    font-size: 0.875rem;
    color: var(--text-muted);
    margin: 0;
    letter-spacing: 0.02em;
}

/* ── Inputs ── */
.stTextInput input, .stTextArea textarea {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
    font-size: 0.9rem !important;
    padding: 0.75rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border-color: var(--amber) !important;
    box-shadow: 0 0 0 3px var(--amber-glow) !important;
    outline: none !important;
}
.stTextInput label, .stTextArea label {
    color: var(--text-secondary) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.15em !important;
    text-transform: uppercase !important;
}

/* ── Buttons ── */
.stButton button {
    background: linear-gradient(135deg, var(--amber), #a07020) !important;
    color: #0a0908 !important;
    border: none !important;
    border-radius: var(--radius) !important;
    font-family: var(--font-mono) !important;
    font-size: 0.75rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    padding: 0.65rem 1.5rem !important;
    cursor: pointer !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 2px 12px rgba(200,146,42,0.25) !important;
}
.stButton button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 24px rgba(200,146,42,0.35) !important;
    background: linear-gradient(135deg, var(--amber-light), var(--amber)) !important;
}
.stButton button:active {
    transform: translateY(0px) !important;
}

/* ── File Uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg-card) !important;
    border: 1.5px dashed var(--border-strong) !important;
    border-radius: var(--radius-lg) !important;
    padding: 1rem !important;
    transition: border-color 0.2s, background 0.2s !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: var(--amber) !important;
    background: var(--bg-hover) !important;
}
[data-testid="stFileUploader"] label {
    color: var(--text-secondary) !important;
    font-family: var(--font-body) !important;
}
[data-testid="stFileUploadDropzone"] {
    background: transparent !important;
}

/* ── Chat Bubbles ── */
.chat-container {
    display: flex;
    flex-direction: column;
    gap: 1.25rem;
    padding: 1.5rem 0;
}
.message-row {
    display: flex;
    align-items: flex-start;
    gap: 0.85rem;
    animation: slideIn 0.3s ease;
}
@keyframes slideIn {
    from { opacity: 0; transform: translateY(8px); }
    to   { opacity: 1; transform: translateY(0); }
}
.message-row.user { flex-direction: row-reverse; }

.avatar {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.8rem;
    flex-shrink: 0;
    font-family: var(--font-display);
}
.avatar.ai {
    background: linear-gradient(135deg, var(--amber), #6b4a10);
    color: #0a0908;
    font-weight: 600;
    box-shadow: 0 0 16px rgba(200,146,42,0.3);
}
.avatar.user {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-secondary);
}

.bubble {
    max-width: 72%;
    padding: 0.9rem 1.2rem;
    border-radius: var(--radius-lg);
    font-family: var(--font-body);
    font-size: 0.9rem;
    line-height: 1.65;
}
.bubble.ai {
    background: var(--bg-card);
    border: 1px solid var(--border);
    color: var(--text-primary);
    border-top-left-radius: var(--radius);
}
.bubble.user {
    background: linear-gradient(135deg, rgba(200,146,42,0.15), rgba(200,146,42,0.08));
    border: 1px solid var(--border-strong);
    color: var(--text-primary);
    border-top-right-radius: var(--radius);
}

.msg-time {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    color: var(--text-muted);
    margin-top: 0.3rem;
    padding: 0 0.2rem;
}
.message-row.user .msg-time { text-align: right; }

/* ── Sources card ── */
.sources-card {
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 0.75rem 1rem;
    margin-top: 0.6rem;
    font-family: var(--font-mono);
    font-size: 0.68rem;
    color: var(--text-muted);
}
.sources-card .src-label {
    color: var(--amber-muted);
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}

/* ── Chat Input Zone ── */
.chat-input-zone {
    position: sticky;
    bottom: 0;
    background: linear-gradient(to top, var(--bg-base) 70%, transparent);
    padding: 1.5rem 0 0.5rem 0;
    margin-top: 1rem;
}

/* ── Document pill ── */
.doc-pill {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: 100px;
    padding: 0.3rem 0.85rem;
    font-family: var(--font-mono);
    font-size: 0.65rem;
    color: var(--text-secondary);
    margin: 0.2rem;
    animation: slideIn 0.2s ease;
}
.doc-pill .doc-icon { color: var(--amber); }

/* ── Stats bar ── */
.stats-bar {
    display: flex;
    gap: 1.5rem;
    padding: 1rem 1.2rem;
    background: var(--bg-card);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    margin-bottom: 1.5rem;
}
.stat-item {
    display: flex;
    flex-direction: column;
    gap: 0.1rem;
}
.stat-value {
    font-family: var(--font-display);
    font-size: 1.5rem;
    font-weight: 600;
    color: var(--amber);
    line-height: 1;
}
.stat-label {
    font-family: var(--font-mono);
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: var(--text-muted);
}
.stat-divider {
    width: 1px;
    background: var(--border);
    margin: 0.2rem 0;
}

/* ── Alerts ── */
.stAlert {
    border-radius: var(--radius) !important;
    font-family: var(--font-body) !important;
    font-size: 0.85rem !important;
}

/* ── Spinner ── */
.stSpinner > div {
    border-color: var(--amber) !important;
    border-top-color: transparent !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg-base); }
::-webkit-scrollbar-thumb {
    background: var(--amber-muted);
    border-radius: 4px;
}

/* ── Horizontal rule ── */
hr { border-color: var(--border) !important; }

/* ── Column gap fix ── */
[data-testid="column"] { padding: 0 0.5rem !important; }

/* ── Select box ── */
.stSelectbox select {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
    font-family: var(--font-body) !important;
}

/* ── Success / Error colors ── */
.stSuccess {
    background: rgba(74, 222, 128, 0.08) !important;
    border: 1px solid rgba(74, 222, 128, 0.2) !important;
    color: #4ade80 !important;
}
.stError {
    background: rgba(248, 113, 113, 0.08) !important;
    border: 1px solid rgba(248, 113, 113, 0.2) !important;
    color: #f87171 !important;
}

/* ── Thinking animation ── */
.thinking {
    display: flex;
    gap: 5px;
    align-items: center;
    padding: 0.4rem 0;
}
.thinking span {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--amber);
    animation: bounce 1.2s ease-in-out infinite;
}
.thinking span:nth-child(2) { animation-delay: 0.15s; }
.thinking span:nth-child(3) { animation-delay: 0.3s; }
@keyframes bounce {
    0%, 60%, 100% { transform: translateY(0); opacity: 0.5; }
    30% { transform: translateY(-6px); opacity: 1; }
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 4rem 2rem;
    color: var(--text-muted);
}
.empty-state .empty-icon {
    font-family: var(--font-display);
    font-size: 4rem;
    color: var(--border-strong);
    margin-bottom: 1rem;
}
.empty-state p {
    font-family: var(--font-body);
    font-size: 0.875rem;
    line-height: 1.6;
    max-width: 300px;
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)


# ─── Session State Init ──────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "documents" not in st.session_state:
    st.session_state.documents = []
if "api_url" not in st.session_state:
    st.session_state.api_url = "http://web:8000/api/chat"


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <div class="logo">◈</div>
        <div style="font-family: var(--font-display); font-size: 1.4rem; font-weight: 300; color: var(--text-primary); margin-top: 0.4rem;">RAG Assistant</div>
        <div class="tagline">Retrieval-Augmented Generation</div>
        <div style="margin-top: 0.8rem;">
            <span class="status-badge">
                <span class="status-dot"></span>
                Connecté
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # API Config
    st.markdown('<div class="sidebar-section">⚙ Configuration</div>', unsafe_allow_html=True)
    api_url = st.text_input(
        "URL de l'API",
        value=st.session_state.api_url,
        placeholder="http://web:8000/api/chat",
        label_visibility="visible",
    )
    st.session_state.api_url = api_url

    # Document ingestion
    st.markdown('<div class="sidebar-section">📂 Ingestion de Documents</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Déposer un fichier",
        type=["pdf", "docx", "txt"],
        help="Formats supportés: PDF, DOCX, TXT",
        label_visibility="collapsed",
    )

    if uploaded_file:
        col_btn, col_void = st.columns([1, 0.3])
        with col_btn:
            if st.button("▲ Ingérer le document", use_container_width=True):
                with st.spinner("Indexation en cours…"):
                    try:
                        files = {
                            "file": (
                                uploaded_file.name,
                                uploaded_file.getvalue(),
                                uploaded_file.type,
                            )
                        }
                        response = requests.post(
                            f"{api_url}/ingest/file", files=files, timeout=60
                        )
                        response.raise_for_status()
                        st.session_state.documents.append(uploaded_file.name)
                        st.success(f"✓ Indexé avec succès")
                    except Exception as exc:
                        st.error(f"Erreur: {exc}")

    # Documents indexés
    if st.session_state.documents:
        st.markdown('<div class="sidebar-section">📑 Documents Indexés</div>', unsafe_allow_html=True)
        for doc in st.session_state.documents:
            ext = doc.split(".")[-1].upper()
            icon = {"PDF": "📄", "DOCX": "📝", "TXT": "📃"}.get(ext, "📎")
            st.markdown(
                f'<div class="doc-pill"><span class="doc-icon">{icon}</span> {doc}</div>',
                unsafe_allow_html=True,
            )

    # Stats
    st.markdown('<div class="sidebar-section">📊 Session</div>', unsafe_allow_html=True)
    n_msg = len(st.session_state.messages)
    n_usr = sum(1 for m in st.session_state.messages if m["role"] == "user")
    n_doc = len(st.session_state.documents)
    st.markdown(f"""
    <div class="stats-bar" style="flex-direction: column; gap: 0.8rem;">
        <div class="stat-item">
            <div class="stat-value">{n_msg}</div>
            <div class="stat-label">Messages échangés</div>
        </div>
        <div class="stat-divider"></div>
        <div class="stat-item">
            <div class="stat-value">{n_doc}</div>
            <div class="stat-label">Documents indexés</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Clear
    st.markdown("<div style='margin-top: 1.5rem;'></div>", unsafe_allow_html=True)
    if st.button("✕ Effacer la conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ─── Main Area ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>Dialogue avec <span>vos documents</span></h1>
    <p>Posez vos questions — le modèle interroge votre base de connaissances en temps réel.</p>
</div>
""", unsafe_allow_html=True)

# ── Chat History Display ──
chat_area = st.container()

with chat_area:
    if not st.session_state.messages:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">◈</div>
            <p>Aucune conversation pour le moment.<br>
            Indexez un document dans la barre latérale, puis posez votre première question.</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown('<div class="chat-container">', unsafe_allow_html=True)
        for msg in st.session_state.messages:
            role = msg["role"]
            content = msg["content"]
            ts = msg.get("time", "")
            sources = msg.get("sources", [])

            if role == "user":
                st.markdown(f"""
                <div class="message-row user">
                    <div>
                        <div class="bubble user">{content}</div>
                        <div class="msg-time">{ts}</div>
                    </div>
                    <div class="avatar user">U</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                sources_html = ""
                if sources:
                    src_list = " · ".join(f"<span>{s}</span>" for s in sources)
                    sources_html = f"""
                    <div class="sources-card">
                        <div class="src-label">Sources consultées</div>
                        {src_list}
                    </div>
                    """
                st.markdown(f"""
                <div class="message-row ai">
                    <div class="avatar ai">◈</div>
                    <div>
                        <div class="bubble ai">{content}</div>
                        {sources_html}
                        <div class="msg-time">{ts}</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)


# ── Input Zone ───────────────────────────────────────────────────────────────
st.markdown("<div style='height: 2rem;'></div>", unsafe_allow_html=True)

with st.container():
    col_input, col_btn = st.columns([6, 1])

    with col_input:
        question = st.text_input(
            "Question",
            placeholder="Ex: Quel est le résumé de ce document ? Quels sont les points clés ?",
            label_visibility="collapsed",
            key="question_input",
        )

    with col_btn:
        send = st.button("Envoyer →", use_container_width=True)

    # Suggestions rapides
    if not st.session_state.messages:
        st.markdown("<div style='margin-top: 0.75rem; display: flex; gap: 0.5rem; flex-wrap: wrap;'>", unsafe_allow_html=True)
        suggestions = [
            "Résume ce document",
            "Quels sont les points clés ?",
            "Donne-moi les conclusions",
            "Y a-t-il des données chiffrées ?",
        ]
        cols = st.columns(len(suggestions))
        for i, sug in enumerate(suggestions):
            with cols[i]:
                if st.button(sug, key=f"sug_{i}", use_container_width=True):
                    question = sug
                    send = True
        st.markdown("</div>", unsafe_allow_html=True)


# ─── Send Logic ──────────────────────────────────────────────────────────────
if send and question and question.strip():
    now = datetime.now().strftime("%H:%M")

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": question,
        "time": now,
    })

    # Call API
    with st.spinner(""):
        st.markdown("""
        <div class="message-row ai" style="padding: 1rem 0;">
            <div class="avatar ai">◈</div>
            <div class="bubble ai">
                <div class="thinking">
                    <span></span><span></span><span></span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        try:
            response = requests.post(
                f"{st.session_state.api_url}/respond",
                json={"question": question},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "Aucune réponse reçue.")
            sources = data.get("sources", [])
        except requests.exceptions.ConnectionError:
            answer = "⚠ Impossible de contacter l'API. Vérifiez l'URL dans la barre latérale."
            sources = []
        except requests.exceptions.Timeout:
            answer = "⚠ La requête a expiré. L'API met trop de temps à répondre."
            sources = []
        except Exception as exc:
            answer = f"⚠ Erreur inattendue : {exc}"
            sources = []

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "time": datetime.now().strftime("%H:%M"),
        "sources": sources,
    })

    st.rerun()