"""
agents/clarity_agent.py
------------------------
Clarity Agent — the first node in the research graph.

Responsibilities
----------------
1. Receive the raw ``state["query"]`` and the full conversation history
   from ``state["messages"]``.
2. Ask the LLM to decide whether the query unambiguously identifies a
   company and has a researchable intent.
3. Parse the LLM's strict JSON response.
4. Write ``clarity_status`` and (optionally) ``clarification_question``
   back to state.
5. Append an AIMessage summarising its decision to ``state["messages"]``.

Multi-turn awareness
--------------------
The full conversation history is injected into the prompt so that
follow-up queries like "What about their CEO?" correctly inherit the
company established in a prior turn — no re-clarification is needed.
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from state import ResearchState
from utils.helpers import build_history_text, extract_json_block, format_agent_header
from utils.llm_factory import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt template
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a query clarity evaluator for a business research assistant.

Your ONLY job is to decide whether the user's latest query contains:
  1. A clearly identifiable company name (or one already established in
     the conversation history), AND
  2. A specific, researchable business intent (e.g. financials, news,
     products, leadership, competitors, market position).

Rules
-----
- If the conversation history already mentions a company, treat any
  follow-up question (e.g. "What about their CEO?") as CLEAR — you
  should NOT ask for clarification again.
- If the query is completely ambiguous (e.g. "Tell me about the company"
  with no prior context), ask for the missing information.
- Respond ONLY with a single valid JSON object — no prose, no markdown.

JSON schema
-----------
{
  "clarity_status": "clear" | "needs_clarification",
  "clarification_question": "<question to ask user>"  // only when needs_clarification
}
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def clarity_agent_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node: evaluate whether the user's query is researchable.

    Reads
    -----
    state["query"]    : Raw user query for this turn.
    state["messages"] : Full conversation history for context.

    Writes (returned dict — LangGraph merges into state)
    ------
    clarity_status          : "clear" or "needs_clarification"
    clarification_question  : Question string (or None)
    messages                : Appends one AIMessage

    Parameters
    ----------
    state : ResearchState
        Current graph state snapshot.

    Returns
    -------
    Dict[str, Any]
        Partial state update dict consumed by LangGraph.
    """
    print(format_agent_header("Clarity Agent", "Evaluating query clarity…"))

    # If the user already provided a clarification answer, treat it as
    # unambiguously clear — we asked; they answered.  No need to call the
    # LLM again; just mark clear and use the clarified query as context.
    clarified_query = state.get("clarified_query")
    if clarified_query:
        print(f"  ✔  clarity_status = clear (clarification received: {clarified_query!r})")
        return {
            "clarity_status":         "clear",
            "clarification_question": None,
            "query":                  clarified_query,  # promote to active query
            "messages": [AIMessage(
                content=(
                    f"[Clarity Agent] Clarification received: '{clarified_query}'. "
                    "Query is now clear. Proceeding to research."
                )
            )],
        }

    # No clarification in flight — evaluate the raw query normally.
    # Use clarified_query if present (shouldn't reach here, but defensive).
    query = clarified_query or state.get("query", "")
    history_text = build_history_text(state.get("messages", []))

    # Build user message that includes history + current query
    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"Latest user query: {query}\n\n"
        "Evaluate the clarity of the latest query given the history above."
    )

    try:
        response = get_llm(temperature=0.0).invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        raw_text: str = response.content
    except Exception as exc:  # pylint: disable=broad-except
        # If the LLM call fails, default to "clear" to avoid blocking the
        # user, and surface the error in state.
        error_msg = f"Clarity Agent LLM error: {exc}"
        print(f"  ⚠  {error_msg}")
        return {
            "clarity_status": "clear",
            "clarification_question": None,
            "error": error_msg,
            "messages": [AIMessage(content=f"[Clarity Agent] {error_msg}")],
        }

    # Parse the LLM's JSON response safely
    parsed = extract_json_block(raw_text)

    clarity_status: str = parsed.get("clarity_status", "clear")
    clarification_question: str | None = parsed.get("clarification_question")

    # Build a human-readable summary for the message history
    if clarity_status == "clear":
        summary = (
            "[Clarity Agent] Query is clear. Proceeding to research."
        )
    else:
        summary = (
            f"[Clarity Agent] Query needs clarification. "
            f"Question: {clarification_question}"
        )

    print(f"  ✔  clarity_status = {clarity_status}")

    return {
        "clarity_status": clarity_status,
        "clarification_question": clarification_question,
        "messages": [AIMessage(content=summary)],
    }
