"""
agents/validator_agent.py
--------------------------
Validator Agent — quality gate between research and synthesis.

Responsibilities
----------------
1. Receive ``state["research_findings"]`` and the original user query.
2. Ask the LLM to critically evaluate the findings on three axes:
   - Completeness  — does it answer what was asked?
   - Relevance     — is it actually about the right company/topic?
   - Recency       — is the data fresh enough to be actionable?
3. Parse the strict JSON verdict.
4. Write ``validation_result`` ("sufficient" | "insufficient") to state.
5. Append an AIMessage with the verdict and reason.

Routing implications (handled in graph.py)
-------------------------------------------
- "sufficient"  OR research_attempts ≥ 3  → synthesis_agent
- "insufficient" AND research_attempts < 3 → back to research_agent
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
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are a critical research quality evaluator.

You will receive:
  1. The user's original research query.
  2. A narrative of research findings produced by a research agent.

Evaluate the findings on three dimensions:
  1. Completeness — Does it substantively answer what the user asked?
  2. Relevance    — Is it actually about the correct company/topic?
  3. Recency      — Is the information recent enough to be useful
                    (within the last 1–2 years ideally)?

Mark as "sufficient" if ALL three dimensions are at least partially met
and the user would receive genuine value from this research.

Mark as "insufficient" if the research is clearly lacking, off-topic,
too vague, or entirely outdated.

Respond ONLY with a single valid JSON object — no prose, no markdown.

JSON schema
-----------
{
  "validation_result": "sufficient" | "insufficient",
  "reason": "<one or two sentence explanation>"
}
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def validator_agent_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node: validate research quality before synthesis.

    Reads
    -----
    state["query"]              : Original user query.
    state["clarified_query"]    : Clarified query (preferred if set).
    state["research_findings"]  : Text produced by the Research Agent.
    state["research_attempts"]  : Used for informational logging.
    state["messages"]           : Full conversation history.

    Writes (returned dict)
    ------
    validation_result : "sufficient" or "insufficient"
    messages          : Appends one AIMessage

    Parameters
    ----------
    state : ResearchState
        Current graph state snapshot.

    Returns
    -------
    Dict[str, Any]
        Partial state update dict.
    """
    attempts = state.get("research_attempts", 1)
    print(
        format_agent_header(
            "Validator Agent",
            f"Validating research quality (attempt {attempts}/3)…"
        )
    )

    query = state.get("clarified_query") or state.get("query", "")
    findings = state.get("research_findings", "No findings provided.")
    history_text = build_history_text(state.get("messages", []))

    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"User research query: {query}\n\n"
        f"Research findings to validate:\n{findings}\n\n"
        "Evaluate the findings and return your JSON verdict."
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
        # On LLM failure, default to "sufficient" at max attempts to
        # avoid an infinite retry loop; otherwise mark insufficient.
        fallback = "sufficient" if attempts >= 3 else "insufficient"
        error_msg = f"Validator Agent LLM error: {exc}"
        print(f"  ⚠  {error_msg}")
        return {
            "validation_result": fallback,
            "error": error_msg,
            "messages": [AIMessage(content=f"[Validator Agent] {error_msg}")],
        }

    # Parse strict JSON verdict
    parsed = extract_json_block(raw_text)

    # Default to "sufficient" if parsing fails — avoids stuck loops
    validation_result: str = parsed.get("validation_result", "sufficient")
    reason: str = parsed.get("reason", "No reason provided.")

    summary = (
        f"[Validator Agent] Verdict: {validation_result.upper()}. "
        f"Reason: {reason}"
    )
    print(f"  ✔  validation_result = {validation_result}")
    print(f"     Reason: {reason}")

    return {
        "validation_result": validation_result,
        "messages": [AIMessage(content=summary)],
    }
