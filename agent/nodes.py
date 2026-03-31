"""
agent/nodes.py

Defines the four nodes of the LangGraph agent:
- Router:    reads the question and decides which path to take
- Retriever: fetches relevant chunks from ChromaDB
- Analyst:   synthesizes a cited answer using the LLM
- Evaluator: scores the answer quality and decides whether to retry
"""

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq

from agent.state import AgentState
from ingestion.embedder import VectorStore
from rag.retriever import Retriever

load_dotenv()

# We initialise the LLM once here at module level.
# This means it's created once when the file is first imported,
# not every time a node function runs. Creating the LLM object
# involves reading the API key and setting up the HTTP client —
# doing that once and reusing it is more efficient.
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.1,
)

# ── ROUTER NODE ──────────────────────────────────────────────────────────────

ROUTER_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a query routing assistant for a consulting document system.
Your job is to classify a user's question into exactly one of these three categories:

- retrieval: a specific question that requires finding relevant passages from documents
  Examples: "What is the revenue forecast?", "What risks were identified?", 
            "Who are the key stakeholders?"

- comparison: a question that explicitly asks to compare, contrast, or analyse 
  differences between multiple documents or sections
  Examples: "Compare the strategies in both reports", 
            "How do the two approaches differ?"

- summary: a request for a broad overview or summary of document content
  Examples: "Summarise the main findings", "Give me an overview of this report",
            "What are the key takeaways?"

Respond with ONLY one word: retrieval, comparison, or summary.
No explanation. No punctuation. Just the single word.""",
        ),
        ("human", "Question: {question}"),
    ]
)


def router_node(state: AgentState) -> dict:
    """
    Reads the user's question and classifies it into a route.

    This is the first node that runs. It sets state['route'] which
    the conditional edges read to decide which node runs next.

    Args:
        state: the current AgentState (only 'question' is populated)

    Returns:
        dict with 'route' and 'metadata' keys updated
    """

    question = state["question"]
    print("\n🔀 Router: classifying question...")

    # Build and run the routing chain.
    # We use temperature=0 here specifically — routing is a classification task,
    # not a creative one. We want the most deterministic answer possible.
    router_chain = (
        ROUTER_PROMPT
        | ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,  # fully deterministic for classification
        )
        | StrOutputParser()
    )

    # Get the route — strip whitespace and lowercase to normalise the response.
    # Even with temperature=0, LLMs sometimes add a trailing newline or space.
    route = router_chain.invoke({"question": question}).strip().lower()

    # Validate the route — if the LLM returned something unexpected,
    # fall back to "retrieval" which is the most general-purpose option.
    valid_routes = {"retrieval", "comparison", "summary"}
    if route not in valid_routes:
        print(f"  ⚠️  Unexpected route '{route}', defaulting to 'retrieval'")
        route = "retrieval"

    print(f"  ✅  Route decided: '{route}'")

    return {
        "route": route,
        "metadata": {
            **state.get("metadata", {}),  # preserve any existing metadata
            "route_used": route,
        },
    }


# ── RETRIEVER NODE ────────────────────────────────────────────────────────────


def retriever_node(state: AgentState, vector_store: VectorStore) -> dict:
    """
    Fetches relevant document chunks from ChromaDB based on the question.

    For 'retrieval' and 'comparison' routes, we search the vector store.
    For 'summary' route, we retrieve a larger number of chunks to get
    broad coverage of the document.

    Args:
        state:        current AgentState (has 'question' and 'route')
        vector_store: the initialised VectorStore instance

    Returns:
        dict with 'retrieved_docs' and 'context' keys updated
    """

    question = state["question"]
    route = state["route"]

    print(f"\n📚 Retriever: fetching chunks for route='{route}'...")

    # For summary requests, we fetch more chunks to get broader coverage.
    # For specific questions, 4 chunks is usually precise enough.
    k = 8 if route == "summary" else 4

    retriever = Retriever(vector_store=vector_store, k=k)
    docs = retriever.retrieve(question)
    context = retriever.format_context(docs)

    print(f"  ✅  Retrieved {len(docs)} chunks")

    return {
        "retrieved_docs": docs,
        "context": context,
        "metadata": {
            **state.get("metadata", {}),
            "chunks_retrieved": len(docs),
        },
    }


# ── ANALYST NODE ──────────────────────────────────────────────────────────────

# We define three different prompt templates — one per route.
# This gives the LLM the right instructions for each task type.

RETRIEVAL_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert consulting analyst.
Answer the question using ONLY the information in the provided context.
Always cite sources using: [Source: filename | Page X]
If the context lacks sufficient information, say so clearly.
Be precise and analytical.""",
        ),
        (
            "human",
            """CONTEXT:
{context}

QUESTION:
{question}

ANSWER (with inline citations):""",
        ),
    ]
)

COMPARISON_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert consulting analyst specialising in comparative analysis.
Compare and contrast the information from different sources in the context.
Structure your response with clear headings.
Always cite sources using: [Source: filename | Page X]
Highlight similarities, differences, and strategic implications.""",
        ),
        (
            "human",
            """CONTEXT:
{context}

COMPARISON REQUEST:
{question}

COMPARATIVE ANALYSIS (with citations):""",
        ),
    ]
)

SUMMARY_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an expert consulting analyst.
Provide a comprehensive executive summary of the provided document context.
Structure it as:
1. Key Findings (3-5 bullet points)
2. Main Themes
3. Strategic Implications
Always cite sources using: [Source: filename | Page X]""",
        ),
        (
            "human",
            """CONTEXT:
{context}

SUMMARY REQUEST:
{question}

EXECUTIVE SUMMARY (with citations):""",
        ),
    ]
)

# Map route names to their corresponding prompts.
# This lets us select the right prompt with a simple dictionary lookup
# instead of a chain of if/elif statements.
ROUTE_TO_PROMPT = {
    "retrieval": RETRIEVAL_PROMPT,
    "comparison": COMPARISON_PROMPT,
    "summary": SUMMARY_PROMPT,
}


def analyst_node(state: AgentState) -> dict:
    """
    Generates a cited answer using the appropriate prompt for the route.

    This node reads the context and question from state, selects the
    right prompt template, and calls the LLM to synthesize an answer.

    Args:
        state: current AgentState (has 'question', 'route', 'context')

    Returns:
        dict with 'answer' key updated
    """

    question = state["question"]
    context = state["context"]
    route = state.get("route", "retrieval")

    print(f"\n🧠 Analyst: generating answer (route='{route}')...")

    # Select the prompt template for this route.
    # .get() with a fallback ensures we always have a valid prompt
    # even if route somehow has an unexpected value.
    prompt = ROUTE_TO_PROMPT.get(route, RETRIEVAL_PROMPT)

    # Build and run the analysis chain.
    analyst_chain = prompt | llm | StrOutputParser()

    answer = analyst_chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    print(f"  ✅  Answer generated ({len(answer)} characters)")

    return {
        "answer": answer,
        "metadata": {
            **state.get("metadata", {}),
            "prompt_type": route,
        },
    }


# ── EVALUATOR NODE ────────────────────────────────────────────────────────────

EVALUATOR_PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are a quality evaluator for a consulting AI system.
Your job is to score an answer on a scale from 0.0 to 1.0 based on:

1. Groundedness (0.4 points): Is the answer based on the provided context?
   Does it avoid making claims not supported by the context?
2. Completeness (0.3 points): Does it fully address the question?
3. Citation quality (0.3 points): Are sources cited correctly and specifically?

Respond with ONLY a number between 0.0 and 1.0 (e.g. 0.85).
No explanation. Just the number.""",
        ),
        (
            "human",
            """CONTEXT PROVIDED:
{context}

QUESTION:
{question}

ANSWER TO EVALUATE:
{answer}

SCORE (0.0 to 1.0):""",
        ),
    ]
)


def evaluator_node(state: AgentState) -> dict:
    """
    Scores the answer quality and decides whether to approve or retry.

    Uses an LLM to evaluate the answer on three dimensions:
    groundedness, completeness, and citation quality.

    If confidence is below the threshold AND we haven't retried yet,
    it signals a retry by not setting 'final_answer'.
    If confidence is acceptable OR we've already retried, it approves
    the answer by setting 'final_answer'.

    Args:
        state: current AgentState (has 'answer', 'context', 'question')

    Returns:
        dict with 'confidence', 'retry_count', and optionally 'final_answer'
    """

    answer = state["answer"]
    context = state["context"]
    question = state["question"]
    retry_count = state.get("retry_count", 0)

    # Threshold below which we trigger a retry.
    # 0.7 means "at least 70% quality" — a reasonable bar for consulting use.
    CONFIDENCE_THRESHOLD = 0.7

    print("\n⚖️  Evaluator: scoring answer quality...")

    evaluator_chain = (
        EVALUATOR_PROMPT
        | ChatGroq(
            model="llama-3.3-70b-versatile",
            temperature=0,  # deterministic — scoring should be consistent
        )
        | StrOutputParser()
    )

    # Get the score and safely parse it to a float.
    # We wrap in try/except because the LLM might occasionally return
    # something that can't be converted to a float (e.g. "0.85." with a period)
    try:
        score_str = evaluator_chain.invoke(
            {
                "context": context,
                "question": question,
                "answer": answer,
            }
        ).strip()
        confidence = float(score_str)
        # Clamp to valid range in case LLM returns something like 1.2
        confidence = max(0.0, min(1.0, confidence))
    except (ValueError, TypeError):
        print("  ⚠️  Could not parse confidence score, defaulting to 0.5")
        confidence = 0.5

    print(
        f"  📊  Confidence score: {confidence:.2f} (threshold: {CONFIDENCE_THRESHOLD})"
    )

    # Decision logic:
    # - If score is good enough → approve the answer
    # - If score is too low AND we haven't retried yet → signal retry
    # - If score is too low BUT we already retried once → approve anyway
    #   (we never want an infinite loop — one retry is enough)
    if confidence >= CONFIDENCE_THRESHOLD or retry_count >= 1:
        if retry_count >= 1 and confidence < CONFIDENCE_THRESHOLD:
            print("  ⚠️  Still below threshold after retry — approving anyway")
        else:
            print("  ✅  Answer approved")

        return {
            "confidence": confidence,
            "retry_count": retry_count,
            "final_answer": answer,  # setting this signals "we're done"
            "metadata": {
                **state.get("metadata", {}),
                "confidence_score": confidence,
                "retried": retry_count > 0,
            },
        }
    else:
        print("  🔄  Below threshold — triggering retry")
        return {
            "confidence": confidence,
            "retry_count": retry_count + 1,  # increment retry counter
            # Note: we do NOT set 'final_answer' here.
            # The graph checks for final_answer to decide whether to retry.
            "metadata": {
                **state.get("metadata", {}),
                "confidence_score": confidence,
                "retried": True,
            },
        }
