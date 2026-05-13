"""
agents/research_agent.py
-------------------------
Research Agent — searches the web and synthesises findings.

Responsibilities
----------------
1. Derive the effective company + topic from ``clarified_query`` or
   ``query``, falling back to conversation history when needed.
2. Build an informative Tavily search query and call ``run_search()``.
3. Feed the search results + conversation history to the LLM, which
   synthesises findings and self-assigns a confidence score (0–10).
4. Parse the strict JSON response from the LLM.
5. Increment ``research_attempts`` by 1.
6. Write ``research_findings`` and ``confidence_score`` to state.
7. Append an AIMessage summarising what was found.

Confidence scoring guidance (embedded in prompt)
-------------------------------------------------
Score > 6  → recent, specific, verifiable data found
Score ≤ 6  → vague, outdated, or off-topic results
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from state import ResearchState
from tools.search import run_search
from utils.helpers import build_history_text, extract_json_block, format_agent_header
from utils.llm_factory import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a senior business intelligence analyst.

You will be given:
  1. Conversation history between the user and the assistant.
  2. A set of fresh web search results about a company.

Your task is to synthesise the search results into a concise but
comprehensive research narrative covering:
  - What the company does
  - Recent news or developments (last 6–12 months if available)
  - Financial highlights or performance signals
  - Leadership or strategic direction clues

Then assign a confidence score (float, 0–10):
  - Score > 6 ONLY if the results contain recent (≤12 months), specific,
    and verifiable information directly about the target company.
  - Score ≤ 6 if results are vague, outdated (>2 years), or mostly
    off-topic / about different companies with similar names.

Respond ONLY with a single valid JSON object — no prose, no markdown.

JSON schema
-----------
{
  "research_findings": "<synthesised narrative>",
  "confidence_score": <float 0.0–10.0>
}
"""


def _build_search_query(state: ResearchState) -> str:
    """Derive a precise Tavily search query using LLM-powered context resolution.

    For follow-up queries like "What about their CEO?", the function
    uses the LLM to resolve the company name from conversation history
    and rewrite the query into a specific, searchable string.

    Parameters
    ----------
    state : ResearchState
        Current graph state.

    Returns
    -------
    str
        A fully resolved search string like
        ``"Tesla CEO Elon Musk recent news 2024 2025"``.
    """
    raw_query    = state.get("clarified_query") or state.get("query", "")
    history_text = build_history_text(state.get("messages", []))

    # Ask the LLM to rewrite the query into an explicit search string,
    # resolving any company references from conversation history.
    resolution_prompt = (
        f"Conversation history:\n{history_text}\n\n"
        f"Latest user query: {raw_query}\n\n"
        "Rewrite the latest query into a specific web search query that:\n"
        "1. Explicitly names the company (resolved from history if needed)\n"
        "2. Includes the specific topic the user is asking about\n"
        "3. Appends '2024 2025' for recency\n\n"
        "Output ONLY the search query string — no explanation, no quotes."
    )
    try:
        response = get_llm(temperature=0.0).invoke(resolution_prompt)
        resolved = response.content.strip().strip('"').strip("'")
        if resolved and len(resolved) > 5:
            return resolved
    except Exception:
        pass  # Fall back to raw query on LLM failure

    # Fallback: append year hints to raw query
    if "2024" not in raw_query and "2025" not in raw_query:
        raw_query = f"{raw_query} 2024 2025"
    return raw_query.strip()


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def research_agent_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node: search the web and synthesise business findings.

    Reads
    -----
    state["clarified_query"] : Preferred query (post-clarification).
    state["query"]           : Fallback raw user query.
    state["messages"]        : Full conversation history.
    state["research_attempts"]: Current attempt count (0-based).

    Writes (returned dict)
    ------
    research_findings  : Synthesised text narrative.
    confidence_score   : Self-assessed float 0–10.
    research_attempts  : Incremented by 1.
    messages           : Appends one AIMessage.

    Parameters
    ----------
    state : ResearchState
        Current graph state snapshot.

    Returns
    -------
    Dict[str, Any]
        Partial state update dict.
    """
    attempt = state.get("research_attempts", 0) + 1
    print(
        format_agent_header(
            "Research Agent",
            f"Searching the web… (attempt {attempt}/3)"
        )
    )

    search_query = _build_search_query(state)
    print(f"  🔍 Tavily query: {search_query!r}")

    # ── Step 1: Call Tavily ──────────────────────────────────────────────
    search_results = run_search(search_query)

    # ── Step 2: Ask LLM to synthesise ───────────────────────────────────
    history_text = build_history_text(state.get("messages", []))

    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"Search query used: {search_query}\n\n"
        f"Search results:\n{search_results}\n\n"
        "Synthesise the findings and provide your confidence score."
    )

    try:
        response = get_llm(temperature=0.2).invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        raw_text: str = response.content
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"Research Agent LLM error: {exc}"
        print(f"  ⚠  {error_msg}")
        return {
            "research_findings": "Research failed due to an LLM error.",
            "confidence_score": 0.0,
            "research_attempts": attempt,
            "error": error_msg,
            "messages": [AIMessage(content=f"[Research Agent] {error_msg}")],
        }

    # ── Step 3: Parse JSON ───────────────────────────────────────────────
    parsed = extract_json_block(raw_text)

    research_findings: str = parsed.get(
        "research_findings",
        "No findings could be extracted from the search results.",
    )
    # Clamp confidence score to valid range 0–10
    raw_score = parsed.get("confidence_score", 5.0)
    try:
        confidence_score = float(max(0.0, min(10.0, raw_score)))
    except (TypeError, ValueError):
        confidence_score = 5.0

    summary = (
        f"[Research Agent] Attempt {attempt}/3 complete. "
        f"Confidence score: {confidence_score:.1f}/10.\n"
        f"Findings preview: {research_findings[:300]}…"
    )
    print(f"  ✔  confidence_score = {confidence_score:.1f}")

    return {
        "research_findings": research_findings,
        "confidence_score": confidence_score,
        "research_attempts": attempt,
        "messages": [AIMessage(content=summary)],
    }
