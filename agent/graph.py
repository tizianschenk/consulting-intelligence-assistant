"""
agent/graph.py

Assembles the LangGraph agent by connecting nodes with edges.

This is where the 'graph' in LangGraph becomes literal — we define
nodes (functions) and edges (connections between them) to build
a stateful, multi-step reasoning agent.
"""

from functools import partial

from langgraph.graph import END, StateGraph

from agent.nodes import (
    analyst_node,
    evaluator_node,
    retriever_node,
    router_node,
)
from agent.state import AgentState
from ingestion.embedder import VectorStore


def route_question(state: AgentState) -> str:
    """
    Conditional edge function: reads state['route'] and returns
    the name of the next node to execute.

    LangGraph calls this function after the Router node completes.
    Whatever string this returns must match a node name in the graph
    OR be the END constant.

    Args:
        state: current AgentState (has 'route' set by router_node)

    Returns:
        string — the name of the next node to run
    """

    route = state.get("route", "retrieval")

    # All three routes go to the retriever first —
    # even summaries need to fetch chunks, just more of them.
    # The retriever node itself handles the k difference.
    if route == "retrieval":
        return "retriever"
    elif route == "comparison":
        return "retriever"
    elif route == "summary":
        return "retriever"
    else:
        return "retriever"  # safe default


def should_retry(state: AgentState) -> str:
    """
    Conditional edge function: reads state after the Evaluator node
    and decides whether to retry the Analyst or finish.

    If 'final_answer' is set in state, the Evaluator approved the answer
    and we're done. If not, we go back to the Analyst for a retry.

    Args:
        state: current AgentState (has 'final_answer' if approved)

    Returns:
        "analyst" to retry, or END to finish
    """

    if state.get("final_answer"):
        print("\n✅  Graph: final answer approved, ending.")
        return END
    else:
        print("\n🔄  Graph: retrying analyst node...")
        return "analyst"


def build_agent(vector_store: VectorStore) -> StateGraph:
    """
    Constructs and compiles the LangGraph agent.

    This function:
    1. Creates a StateGraph with our AgentState schema
    2. Adds all four nodes
    3. Connects them with edges (fixed and conditional)
    4. Compiles it into a runnable object

    Args:
        vector_store: initialised VectorStore to pass to the retriever node

    Returns:
        a compiled LangGraph runnable
    """

    # Step 1: Create the graph with our state schema.
    # Passing AgentState tells LangGraph what fields exist in the state
    # and how to merge partial updates from each node.
    graph = StateGraph(AgentState)

    # Step 2: Add nodes.
    # add_node(name, function) — the name is what edges reference,
    # the function is what runs when that node is activated.

    # Router node — straightforward, just needs state
    graph.add_node("router", router_node)

    # Retriever node — needs vector_store in addition to state.
    # partial() pre-fills the vector_store argument so the function
    # signature becomes just (state) → dict, which is what LangGraph expects.
    graph.add_node("retriever", partial(retriever_node, vector_store=vector_store))

    # Analyst and Evaluator nodes — just need state
    graph.add_node("analyst", analyst_node)
    graph.add_node("evaluator", evaluator_node)

    # Step 3: Set the entry point.
    # This tells LangGraph which node runs first when we invoke the agent.
    graph.set_entry_point("router")

    # Step 4: Add edges.
    # Fixed edge: router always goes to... a conditional check.
    # add_conditional_edges(from_node, condition_fn, mapping) means:
    # "after 'router' runs, call route_question(state) and use its
    # return value to look up the next node in the mapping dict."
    graph.add_conditional_edges(
        "router",  # after this node runs...
        route_question,  # call this function to get a string...
        {  # and use that string to find the next node
            "retriever": "retriever",
        },
    )

    # Fixed edge: retriever always goes to analyst
    graph.add_edge("retriever", "analyst")

    # Fixed edge: analyst always goes to evaluator
    graph.add_edge("analyst", "evaluator")

    # Conditional edge: evaluator either ends or retries analyst
    graph.add_conditional_edges(
        "evaluator",  # after this node runs...
        should_retry,  # call this to decide what happens next...
        {  # map the return value to the next node
            "analyst": "analyst",
            END: END,
        },
    )

    # Step 5: Compile the graph into a runnable.
    # This validates the graph structure (checks for disconnected nodes,
    # missing edges etc.) and returns an object with an .invoke() method.
    compiled = graph.compile()

    print("✅  Agent graph compiled successfully")
    return compiled


def run_agent(question: str, vector_store: VectorStore) -> dict:
    """
    Convenience function to build and run the agent for a single question.

    Args:
        question:     the user's natural language question
        vector_store: initialised VectorStore instance

    Returns:
        the final AgentState dict with all fields populated
    """

    # Build the agent graph
    agent = build_agent(vector_store)

    # Define the initial state — only 'question' is set at the start.
    # All other fields start empty and get filled in as nodes run.
    initial_state = {
        "question": question,
        "route": "",
        "retrieved_docs": [],
        "context": "",
        "answer": "",
        "confidence": 0.0,
        "retry_count": 0,
        "final_answer": "",
        "metadata": {},
    }

    print(f"\n{'=' * 60}")
    print("AGENT STARTED")
    print(f"Question: {question}")
    print(f"{'=' * 60}")

    # Invoke the graph with the initial state.
    # LangGraph runs each node in sequence, updating the state,
    # until it reaches END. It then returns the final state.
    final_state = agent.invoke(initial_state)

    print(f"\n{'=' * 60}")
    print("AGENT FINISHED")
    print(f"Route used:   {final_state.get('metadata', {}).get('route_used', '?')}")
    print(
        f"Chunks used:  {final_state.get('metadata', {}).get('chunks_retrieved', '?')}"
    )
    print(
        f"Confidence:   {final_state.get('metadata', {}).get('confidence_score', '?')}"
    )
    print(f"Retried:      {final_state.get('metadata', {}).get('retried', False)}")
    print("\nFINAL ANSWER:")
    for line in final_state.get("final_answer", "No answer generated").split("\n"):
        print(f"  {line}")
    print(f"{'=' * 60}\n")

    return final_state
