"""
rag/rag_chain.py

Connecting retrieval to the LLM

The core RAG chain: takes a user question, retrieves relevant context,
and uses the Groq LLM to generate a grounded, cited answer.

This is the 'synthesis layer' — it transforms raw retrieved chunks into a coherent, trustworthy answer.

This file is where everything comes together. It takes the formatted context from the retriever,
packs it into a carefully designed prompt, sends it to Groq, and returns a structured answer with citations.

"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# StrOutputParser — when the LLM responds, it returns a complex AIMessage object with lots of internal metadata.
# StrOutputParser strips all that away and gives us back a plain Python string. Simple but essential.
from langchain_core.prompts import ChatPromptTemplate

# ChatPromptTemplate — this is LangChain's template system.
# Instead of building prompt strings manually with f-strings, we define a template with named placeholders like {context} and {question}.
# LangChain fills them in at runtime. This makes the prompt reusable, testable, and clean.
from langchain_groq import ChatGroq

# ChatGroq — the LangChain wrapper around Groq's API.
# When we call it with a prompt, it sends a request to Groq's servers, runs it through Llama 3.3 70B, and returns the response.
from ingestion.embedder import VectorStore
from rag.retriever import Retriever

# Load environment variables from .env so GROQ_API_KEY is available.
# This must happen before we initialise ChatGroq — it reads the key
# from the environment at construction time.
load_dotenv()

# This is the most important design decision in the RAG chain.
# The prompt template controls exactly how the LLM behaves —
# what role it plays, what constraints it operates under,
# and how it should format its response.
#
# We use a two-message structure:
# - 'system': sets the LLM's role and rules (like a job description)
# - 'human': contains the actual context and question (like a task brief)
#
# This mirrors how a consultant would brief a junior analyst:
# "Here are your instructions. Here are the documents. Answer this question."

# Rule 1 forces grounding in the context
# Rule 2 forces honest admission when the answer isn't there
# Rule 3 forces citations so every claim is traceable
# Rule 5 explicitly bans hallucination

SYSTEM_PROMPT = """You are an expert consulting analyst assistant.
Your job is to answer questions about business documents with precision and clarity.

STRICT RULES:
1. Answer ONLY using the information in the CONTEXT section below.
2. If the context does not contain enough information to answer, say clearly:
   "The uploaded documents do not contain enough information to answer this question."
3. ALWAYS cite your sources using the format: [Source: filename | Page X]
4. Be concise but complete. Use bullet points for lists of findings.
5. Never make up facts, statistics, or claims not present in the context.

Your answers should reflect the analytical rigour expected at a top consulting firm."""

HUMAN_PROMPT = """CONTEXT (retrieved from uploaded documents):
{context}

QUESTION:
{question}

ANSWER (cite sources inline using [Source: filename | Page X]):"""


# Build the prompt template from our two message strings.
# ChatPromptTemplate.from_messages() takes a list of (role, content) tuples.
# 'system' sets the AI's behaviour; 'human' is the user's input.
PROMPT_TEMPLATE = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        ("human", HUMAN_PROMPT),
    ]
)


class RAGChain:
    """
    The complete RAG pipeline: retrieve → prompt → generate → parse.

    Takes a user question, finds relevant document chunks, injects them
    into a carefully designed prompt, and returns a cited answer.
    """

    def __init__(
        self, vector_store: VectorStore, k: int = 4, min_similarity: float = 0.0
    ):
        """
        Args:
            vector_store:   an initialised VectorStore instance
            k:              number of chunks to retrieve per question
            min_similarity: minimum similarity score to include a chunk
        """

        # Initialise the retriever with our vector store.
        # The retriever handles finding and formatting relevant chunks.
        self.retriever = Retriever(
            vector_store=vector_store,
            k=k,
            min_similarity=min_similarity,
        )

        # Initialise the Groq LLM.
        # model: we use llama-3.3-70b-versatile — Groq's most capable
        #        free-tier model. 70 billion parameters.
        # temperature: controls randomness in the output.
        #   0.0 = fully deterministic, always picks the most likely next word.
        #   1.0 = very creative and varied.
        #   We use 0.1 — nearly deterministic but with a tiny bit of
        #   natural variation so answers don't feel robotic.
        # For a consulting analyst, we want precision over creativity.
        self.llm = ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0.1,
        )

        # The output parser strips the LLM's response object down to
        # a plain string. Every LLM call returns an AIMessage object —
        # StrOutputParser extracts just the .content text from it.
        self.output_parser = StrOutputParser()

        # Build the chain using LangChain's pipe operator (|).
        # This is called an LCEL chain (LangChain Expression Language).
        # The | operator means "pass the output of the left side as
        # input to the right side" — like a Unix pipe in the terminal.
        #
        # So: prompt | llm | parser means:
        #   1. Fill in the prompt template with context + question
        #   2. Send the filled prompt to the LLM
        #   3. Parse the LLM's response object into a plain string
        self.chain = PROMPT_TEMPLATE | self.llm | self.output_parser

    def ask(self, question: str) -> dict:
        """
        Ask a question and get a cited answer grounded in the documents.

        Args:
            question: the user's natural language question

        Returns:
            a dictionary with:
            - 'answer':   the LLM's response string
            - 'sources':  list of Document objects used as context
            - 'question': the original question (useful for display)
        """

        print(f"\n🔍 Question: {question}")
        print("  🔄  Retrieving relevant chunks...")

        # Step 1: retrieve the most relevant chunks for this question
        source_docs = self.retriever.retrieve(question)

        # Step 2: format those chunks into a single context string
        # This is what gets injected into the {context} placeholder
        # in our prompt template
        context = self.retriever.format_context(source_docs)

        print("  🔄  Generating answer...")

        # Step 3: run the chain
        # We pass a dict with keys matching our prompt template placeholders.
        # LangChain fills {context} and {question} into the template,
        # sends it to the LLM, and the parser returns a plain string.
        answer = self.chain.invoke(
            {
                "context": context,
                "question": question,
            }
        )

        print("  ✅  Answer generated")

        # Step 4: return everything — the answer and the source chunks.
        # Returning the source chunks lets the UI display exactly which
        # passages the answer was based on, enabling full traceability.
        return {
            "question": question,
            "answer": answer,
            "sources": source_docs,
        }


def print_response(response: dict) -> None:
    """
    Pretty-print a RAG chain response to the terminal.
    Useful for testing and development.

    Args:
        response: the dict returned by RAGChain.ask()
    """

    print("\n" + "=" * 60)
    print("QUESTION:")
    print(f"  {response['question']}")

    print("\nANSWER:")
    # Indent each line of the answer for readability
    for line in response["answer"].split("\n"):
        print(f"  {line}")

    print("\nSOURCES USED:")
    for i, doc in enumerate(response["sources"]):
        print(
            f"  [{i + 1}] {doc.metadata.get('file_name', '?')} "
            f"| Page {doc.metadata.get('page', '?')} "
            f"| Similarity: {doc.metadata.get('similarity_score', '?')}"
        )

    print("=" * 60)
