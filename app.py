"""
app.py

Streamlit frontend for the Consulting Intelligence Assistant.

Run with: streamlit run app.py
"""

import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from agent.graph import run_agent
from ingestion.chunker import chunk_documents
from ingestion.embedder import VectorStore
from ingestion.pdf_loader import load_pdf

load_dotenv()

# ── Page configuration ────────────────────────────────────────────────────────
# This must be the FIRST Streamlit call in the script.
# It sets the browser tab title, icon, and layout.
# layout="wide" gives us the full browser width — better for a chat + sidebar layout.
st.set_page_config(
    page_title="Consulting Intelligence Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Session state initialisation ──────────────────────────────────────────────
# session_state persists across Streamlit reruns (each user interaction).
# We initialise all keys here at the top so every part of the script
# can safely read them without checking if they exist first.

if "messages" not in st.session_state:
    # Chat history — list of dicts with 'role' and 'content' keys.
    # role is either 'user' or 'assistant'.
    st.session_state.messages = []

if "vector_store" not in st.session_state:
    # The VectorStore instance — created once and reused across all questions.
    st.session_state.vector_store = None

if "ingested_files" not in st.session_state:
    # List of filenames that have been successfully ingested.
    # Used to show the user which documents are loaded.
    st.session_state.ingested_files = []

if "last_response_metadata" not in st.session_state:
    # Metadata from the most recent agent run — route, confidence, sources.
    # Displayed below the latest answer for demo purposes.
    st.session_state.last_response_metadata = None

# ── Cached resource loader ────────────────────────────────────────────────────
# @st.cache_resource means: run this function once, cache the result,
# and return the cached result on every subsequent call.
# Without this, the VectorStore (and its embedding model) would reload
# on every single user interaction — extremely slow.


@st.cache_resource
def get_vector_store():
    """
    Creates and returns a single VectorStore instance.
    Called once on app startup, then cached for the session lifetime.
    """
    return VectorStore()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("📁 Document Manager")
    st.markdown("---")

    # File uploader — accepts multiple PDFs at once.
    # type=["pdf"] restricts to PDF files only.
    # accept_multiple_files=True lets the user upload several at once.
    uploaded_files = st.file_uploader(
        "Upload PDF documents",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more consulting reports, strategy papers, or any business PDFs",
    )

    # Ingest button — only shown when files are uploaded
    if uploaded_files:
        if st.button("⚡ Ingest Documents", type="primary", use_container_width=True):
            # Get or create the vector store
            vector_store = get_vector_store()

            # Show a progress bar while ingesting
            progress = st.progress(0, text="Starting ingestion...")

            all_chunks = []
            for i, uploaded_file in enumerate(uploaded_files):
                # Streamlit gives us file objects in memory, not file paths.
                # PyMuPDF needs a file path, so we write to a temporary file.
                # tempfile.NamedTemporaryFile creates a temp file that auto-deletes.
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    tmp_path = tmp_file.name

                try:
                    progress.progress(
                        (i / len(uploaded_files)) * 0.6,
                        text=f"Loading {uploaded_file.name}...",
                    )

                    # Load and chunk the PDF
                    docs = load_pdf(tmp_path)

                    # Rename the source metadata to the original filename
                    # (not the temp file path, which is meaningless to the user)
                    for doc in docs:
                        doc.metadata["source"] = uploaded_file.name
                        doc.metadata["file_name"] = uploaded_file.name

                    chunks = chunk_documents(docs)
                    all_chunks.extend(chunks)

                    # Track which files have been ingested
                    if uploaded_file.name not in st.session_state.ingested_files:
                        st.session_state.ingested_files.append(uploaded_file.name)

                finally:
                    # Always clean up the temp file, even if an error occurred
                    os.unlink(tmp_path)

            # Embed and store all chunks
            progress.progress(0.8, text="Embedding and storing chunks...")
            vector_store.add_documents(all_chunks)
            st.session_state.vector_store = vector_store

            progress.progress(1.0, text="Done!")
            st.success(
                f"✅ Ingested {len(all_chunks)} chunks from {len(uploaded_files)} file(s)"
            )

    # Show currently loaded documents
    if st.session_state.ingested_files:
        st.markdown("### 📚 Loaded Documents")
        for fname in st.session_state.ingested_files:
            st.markdown(f"✅ `{fname}`")

    st.markdown("---")

    # Settings section
    st.markdown("### ⚙️ Settings")
    k_chunks = st.slider(
        "Chunks to retrieve (k)",
        min_value=1,
        max_value=10,
        value=4,
        help="How many document chunks to retrieve per question. Higher = more context but slower.",
    )
    min_similarity = st.slider(
        "Minimum similarity",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.05,
        help="Filter out chunks below this similarity score. Raise to get more precise results.",
    )

    # Clear chat button
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.last_response_metadata = None
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.title("🧠 Consulting Intelligence Assistant")
st.markdown(
    "Upload consulting documents in the sidebar, then ask questions below. "
    "All answers are grounded in your documents with source citations."
)

# Show a warning if no documents are loaded yet
if not st.session_state.ingested_files:
    st.info("👈 Upload and ingest PDF documents using the sidebar to get started.")

# ── Chat history display ───────────────────────────────────────────────────────
# Loop through all previous messages and render them.
# st.chat_message() creates a styled message bubble.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        # For assistant messages, show the source citations if stored
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📎 Sources used", expanded=False):
                for i, source in enumerate(message["sources"]):
                    st.markdown(
                        f"**[{i + 1}]** `{source['file_name']}` | "
                        f"Page {source['page']} | "
                        f"Similarity: {source['similarity_score']}"
                    )
                    st.caption(source["preview"])

        # Show agent metadata for the most recent answer
        if message["role"] == "assistant" and "metadata" in message:
            meta = message["metadata"]
            cols = st.columns(3)
            cols[0].metric("🔀 Route", meta.get("route_used", "?"))
            cols[1].metric("📊 Confidence", f"{meta.get('confidence_score', 0):.0%}")
            cols[2].metric("🔄 Retried", "Yes" if meta.get("retried") else "No")

# ── Chat input ────────────────────────────────────────────────────────────────
# st.chat_input() renders a fixed input box at the bottom of the page.
# It returns the user's message when they press Enter, otherwise None.
if prompt := st.chat_input("Ask a question about your documents..."):
    # Check if documents are loaded before allowing questions
    if not st.session_state.vector_store:
        st.error("Please upload and ingest documents first using the sidebar.")
    else:
        # Add user message to chat history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response with a loading spinner
        with st.chat_message("assistant"):
            with st.spinner("🔍 Thinking..."):
                # Run the agent
                final_state = run_agent(
                    question=prompt,
                    vector_store=st.session_state.vector_store,
                )

                answer = final_state.get(
                    "final_answer", "I could not generate an answer."
                )
                metadata = final_state.get("metadata", {})
                sources = final_state.get("retrieved_docs", [])

            # Display the answer
            st.markdown(answer)

            # Format sources for storage and display
            formatted_sources = []
            for doc in sources:
                formatted_sources.append(
                    {
                        "file_name": doc.metadata.get("file_name", "Unknown"),
                        "page": doc.metadata.get("page", "?"),
                        "similarity_score": doc.metadata.get("similarity_score", 0),
                        "preview": doc.page_content[:200] + "...",
                    }
                )

            # Show sources expander
            if formatted_sources:
                with st.expander("📎 Sources used", expanded=False):
                    for i, source in enumerate(formatted_sources):
                        st.markdown(
                            f"**[{i + 1}]** `{source['file_name']}` | "
                            f"Page {source['page']} | "
                            f"Similarity: {source['similarity_score']}"
                        )
                        st.caption(source["preview"])

            # Show agent metadata metrics
            cols = st.columns(3)
            cols[0].metric("🔀 Route", metadata.get("route_used", "?"))
            cols[1].metric(
                "📊 Confidence", f"{metadata.get('confidence_score', 0):.0%}"
            )
            cols[2].metric("🔄 Retried", "Yes" if metadata.get("retried") else "No")

        # Store the complete message in chat history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "sources": formatted_sources,
                "metadata": metadata,
            }
        )
