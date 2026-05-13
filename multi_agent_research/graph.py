"""
graph.py
--------
LangGraph graph construction and compilation for the Multi-Agent
Business Research Assistant.

Graph topology (matches assignment spec exactly)
-------------------------------------------------

  START
    └──> [clarity]
              ├── needs_clarification ──> [human_clarification]
              │                                └──> [clarity]  (re-evaluate)
              └── clear ──> [research]
                                ├── confidence >= 6 ──> [synthesis] ──> END
                                └── confidence < 6  ──> [validator]
                                          ├── sufficient OR attempts >= 3 ──> [synthesis] ──> END
                                          └── insufficient AND attempts < 3 ──> [research] (retry)

Key design decisions
--------------------
- Research Agent routes DIRECTLY to Synthesis if confidence >= 6, skipping
  the Validator entirely (per assignment spec).  Only low-confidence results
  (< 6) go through the Validator quality gate.
- ``interrupt_before=["human_clarification"]`` declares the pause point at
  compile time; LangGraph pauses BEFORE the node executes and resumes after
  main.py injects the user's answer via update_state().
- ``MemorySaver`` checkpointer persists the full conversation across turns
  within a session using a stable ``thread_id``.
- All routing functions are pure (read-only state, no side effects).
"""

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from langchain_core.messages import HumanMessage

from state import ResearchState
from agents import (
    clarity_agent_node,
    research_agent_node,
    validator_agent_node,
    synthesis_agent_node,
)


# ---------------------------------------------------------------------------
# Routing functions (pure — read-only, no side effects)
# ---------------------------------------------------------------------------

def route_after_clarity(state: ResearchState) -> str:
    """Decide the next node after the Clarity Agent runs.

    Parameters
    ----------
    state : ResearchState
        Current graph state (read-only inside routing functions).

    Returns
    -------
    str
        ``"human_clarification"`` if the query is ambiguous, otherwise
        ``"research"`` to proceed directly to the Research Agent.
    """
    if state.get("clarity_status") == "needs_clarification":
        return "human_clarification"
    return "research"


def route_after_research(state: ResearchState) -> str:
    """Decide the next node after the Research Agent runs.

    Per assignment spec:
    - confidence >= 6  → go directly to Synthesis (good enough data)
    - confidence < 6   → go to Validator for quality assessment

    Parameters
    ----------
    state : ResearchState
        Current graph state (read-only).

    Returns
    -------
    str
        ``"synthesis"`` or ``"validator"``.
    """
    confidence = state.get("confidence_score") or 0.0

    # High-confidence results skip the Validator entirely
    if confidence >= 6.0:
        return "synthesis"

    # Low-confidence results go through the Validator quality gate
    return "validator"


def route_after_validation(state: ResearchState) -> str:
    """Decide the next node after the Validator Agent runs.

    Logic
    -----
    - ``validation_result == "sufficient"``  → Synthesis
    - ``research_attempts >= 3``             → Synthesis (max retries exhausted)
    - Otherwise                              → Research Agent (retry)

    Parameters
    ----------
    state : ResearchState
        Current graph state (read-only).

    Returns
    -------
    str
        ``"synthesis"`` or ``"research"``.
    """
    validation_result = state.get("validation_result", "insufficient")
    attempts = state.get("research_attempts", 0)

    # Proceed to synthesis if quality is acceptable OR retries are exhausted
    if validation_result == "sufficient" or attempts >= 3:
        return "synthesis"

    # Loop back for another research attempt
    return "research"


# ---------------------------------------------------------------------------
# Human clarification node
# ---------------------------------------------------------------------------

def human_clarification_node(state: ResearchState) -> dict:
    """LangGraph node: receive and store the user's clarification answer.

    This node is listed in ``interrupt_before`` at compile time, so LangGraph
    automatically pauses BEFORE executing this function and waits for the
    caller (main.py) to inject the user's answer via ``graph.update_state()``.

    By the time this function body executes, ``state["clarified_query"]`` has
    already been written by main.py.  We simply propagate it and append a
    HumanMessage to the conversation history.

    Reads
    -----
    state["clarified_query"]        : Answer injected by main.py.
    state["clarification_question"] : Fallback if no answer provided.

    Writes (returned dict)
    ------
    clarified_query : Confirmed user answer.
    messages        : Appends one HumanMessage with the clarification.

    Parameters
    ----------
    state : ResearchState
        Current graph state snapshot.

    Returns
    -------
    dict
        Partial state update dict.
    """
    question = state.get(
        "clarification_question",
        "Could you please clarify your request?"
    )

    # By the time this runs, main.py has already injected clarified_query
    # via graph.update_state(as_node="human_clarification").
    user_answer: str = state.get("clarified_query") or question

    return {
        "clarified_query": user_answer,
        "messages": [HumanMessage(content=user_answer)],
    }


# ---------------------------------------------------------------------------
# Graph construction
# ---------------------------------------------------------------------------

def build_graph():
    """Construct, compile, and return the LangGraph research graph.

    Returns
    -------
    CompiledStateGraph
        Compiled graph with MemorySaver checkpointer and interrupt_before
        declaration, ready for ``.stream()`` with a ``thread_id`` config.
    """
    builder = StateGraph(ResearchState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("clarity",             clarity_agent_node)
    builder.add_node("human_clarification", human_clarification_node)
    builder.add_node("research",            research_agent_node)
    builder.add_node("validator",           validator_agent_node)
    builder.add_node("synthesis",           synthesis_agent_node)

    # ── Entry point ───────────────────────────────────────────────────────
    builder.add_edge(START, "clarity")

    # ── After Clarity: clear → research, ambiguous → pause for human ──────
    builder.add_conditional_edges(
        "clarity",
        route_after_clarity,
        {
            "human_clarification": "human_clarification",
            "research":            "research",
        },
    )

    # ── After clarification: re-evaluate with the new query ───────────────
    builder.add_edge("human_clarification", "clarity")

    # ── After Research: high confidence → synthesis, low → validator ──────
    builder.add_conditional_edges(
        "research",
        route_after_research,
        {
            "synthesis": "synthesis",   # confidence >= 6 — fast path
            "validator": "validator",   # confidence < 6  — quality check
        },
    )

    # ── After Validator: sufficient/max-retries → synthesis, else retry ───
    builder.add_conditional_edges(
        "validator",
        route_after_validation,
        {
            "synthesis": "synthesis",
            "research":  "research",    # retry loop (max 3 attempts)
        },
    )

    # ── Synthesis is always terminal ──────────────────────────────────────
    builder.add_edge("synthesis", END)

    # ── Compile with persistence and interrupt support ────────────────────
    # interrupt_before=["human_clarification"]: LangGraph pauses BEFORE this
    # node, surfaces the clarification question, and waits for update_state().
    # recursion_limit is set per-call in main.py CONFIG (default 25).
    checkpointer = MemorySaver()
    graph = builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_clarification"],
    )

    return graph


# Module-level compiled graph — imported directly by main.py
research_graph = build_graph()
