"""
setup_check.py

Run this script to verify your entire environment is correctly configured
before writing any application code.

Usage:
    python setup_check.py
"""

import sys


def check(label, fn):
    """Runs a test function and prints a pass/fail result."""
    try:
        fn()
        print(f"  ✅  {label}")
    except Exception as e:
        print(f"  ❌  {label}")
        print(f"       → {e}")


print("\n🔍 Consulting Intelligence Assistant — Environment Check\n")

# ── Python version ────────────────────────────────────────────────────────────
# We need 3.10+ for some modern Python syntax we'll use later (e.g. match/case)
check(
    f"Python {sys.version.split()[0]} (need 3.10+)",
    lambda: (_ for _ in ()).throw(RuntimeError("Need 3.10+"))
    if sys.version_info < (3, 10)
    else None,
)

# ── Package imports ───────────────────────────────────────────────────────────
# These just confirm the packages installed correctly and can be imported.
# If any fail here, the package didn't install properly.
check("langchain", lambda: __import__("langchain"))
check("langchain_groq", lambda: __import__("langchain_groq"))
check("langgraph", lambda: __import__("langgraph"))
check("chromadb", lambda: __import__("chromadb"))
check("sentence_transformers", lambda: __import__("sentence_transformers"))
check("streamlit", lambda: __import__("streamlit"))
check("fitz (PyMuPDF)", lambda: __import__("fitz"))
check("pdfplumber", lambda: __import__("pdfplumber"))
check("dotenv", lambda: __import__("dotenv"))

# ── .env file & API key ───────────────────────────────────────────────────────
# python-dotenv reads your .env file and loads GROQ_API_KEY into the
# environment so our app can access it without hardcoding secrets in code.
import os

from dotenv import load_dotenv

load_dotenv()

check(
    "GROQ_API_KEY found in .env",
    lambda: (_ for _ in ()).throw(RuntimeError("Key missing or .env not found"))
    if not os.getenv("GROQ_API_KEY")
    else None,
)


# ── Live Groq API call ────────────────────────────────────────────────────────
# This actually sends a tiny request to Groq's servers to confirm:
#   1. Your API key is valid
#   2. The model is accessible
#   3. You can receive a response
# ChatGroq is the LangChain wrapper around Groq's API.
# temperature=0 means "don't be creative, be deterministic" — useful for tests.
def test_groq():
    from langchain_groq import ChatGroq

    llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0)
    response = llm.invoke("Say OK and nothing else.")
    assert response.content.strip() != "", "Got empty response from Groq"


check("Groq API live call → llama-3.3-70b-versatile", test_groq)


# ── Embedding model ───────────────────────────────────────────────────────────
# This loads the all-MiniLM-L6-v2 model locally (downloads ~90MB on first run)
# and converts a test sentence into a vector of 384 numbers.
# We assert the length is 384 to confirm the right model loaded.
def test_embeddings():
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("all-MiniLM-L6-v2")
    vector = model.encode("This is a test sentence.")
    assert len(vector) == 384, f"Expected 384 dimensions, got {len(vector)}"


check("Embeddings: all-MiniLM-L6-v2 (384 dimensions)", test_embeddings)


# ── ChromaDB ──────────────────────────────────────────────────────────────────
# Creates a temporary in-memory ChromaDB instance (nothing written to disk),
# adds one document, queries it, then deletes the collection.
# This confirms the vector database can store and retrieve data correctly.
def test_chromadb():
    import chromadb

    client = chromadb.Client()  # in-memory only
    col = client.create_collection("setup_test")  # create a collection (like a table)
    col.add(documents=["hello world"], ids=["doc_1"])  # store a document
    result = col.query(query_texts=["hello"], n_results=1)  # search for it
    assert len(result["documents"]) > 0, "Query returned no results"
    client.delete_collection("setup_test")  # clean up


check("ChromaDB: store and retrieve (in-memory)", test_chromadb)

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n✨ If everything is green, good to go!\n")
print("   If anything is red, there are issues to fix.\n")
