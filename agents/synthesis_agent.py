"""
agents/synthesis_agent.py
--------------------------
Synthesis Agent — terminal node that produces the final user-facing answer.

Responsibilities
----------------
1. Receive the complete conversation history and validated research
   findings from state.
2. Instruct the LLM to produce a clean, well-structured Markdown
   response organised into canonical business-intelligence sections.
3. Store the result in ``state["final_summary"]``.
4. Append an AIMessage with the full summary.

This node always routes to END — it is the terminal step in the graph
regardless of whether the research was marked sufficient or insufficient
(after 3 failed attempts, we synthesise with what we have).
"""

import os
from typing import Dict, Any

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, SystemMessage, HumanMessage

from state import ResearchState
from utils.helpers import build_history_text, format_agent_header
from utils.llm_factory import get_llm

load_dotenv()

# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------
_SYSTEM_PROMPT = """You are an expert business intelligence writer.

You will receive:
  1. The full conversation history (user queries + prior agent responses).
  2. Validated research findings about a company.

Your task is to write a comprehensive, professional Markdown summary
that directly answers the user's question. Organise it using these
sections (include only sections where you actually have data):

---
## 🏢 Company Overview
Brief description of what the company does, its industry, and scale.

## 📰 Recent News & Developments
Bullet points of the most important recent events, announcements, or
strategic moves (prioritise the last 6–12 months).

## 💰 Financial Highlights
Key financial metrics, revenue trends, profitability signals, or
valuation data if available. If unavailable, state that clearly.

## 👥 Leadership & Strategy
CEO, key executives, and any notable strategic direction or vision.

## 🔑 Key Takeaways
3–5 concise bullet points summarising the most important insights for
the user.
---

Rules
-----
- Use professional but accessible language.
- Do NOT hallucinate facts — if data is missing, say so explicitly.
- Cite sources by mentioning the news outlet or website name when
  referencing specific claims.
- If research confidence was low, include a brief disclaimer at the top.
- Respond with the Markdown content only — no JSON wrapper.
"""


# ---------------------------------------------------------------------------
# Node function
# ---------------------------------------------------------------------------

def synthesis_agent_node(state: ResearchState) -> Dict[str, Any]:
    """LangGraph node: produce the final Markdown summary for the user.

    This is the terminal node — it always routes to END after execution.

    Reads
    -----
    state["research_findings"] : Validated research narrative.
    state["confidence_score"]  : Used to decide if a disclaimer is needed.
    state["messages"]          : Full conversation history.
    state["query"]             : Original user query for context.

    Writes (returned dict)
    ------
    final_summary : Complete Markdown business intelligence report.
    messages      : Appends one AIMessage containing the full summary.

    Parameters
    ----------
    state : ResearchState
        Current graph state snapshot.

    Returns
    -------
    Dict[str, Any]
        Partial state update dict.
    """
    print(format_agent_header("Synthesis Agent", "Generating final report…"))

    findings = state.get("research_findings", "No research findings available.")
    confidence = state.get("confidence_score", 5.0)
    query = state.get("clarified_query") or state.get("query", "")
    history_text = build_history_text(state.get("messages", []))

    # Add a confidence disclaimer hint so the LLM can flag low-quality data
    confidence_hint = (
        f"Note: Research confidence score was {confidence:.1f}/10. "
        + (
            "Data quality is high — proceed with confidence."
            if confidence > 6
            else "Data quality is moderate or low — consider flagging uncertainty."
        )
    )

    user_content = (
        f"Conversation history:\n{history_text}\n\n"
        f"User's research question: {query}\n\n"
        f"{confidence_hint}\n\n"
        f"Research findings:\n{findings}\n\n"
        "Generate the final structured Markdown summary now."
    )

    try:
        response = get_llm(temperature=0.4).invoke(
            [
                SystemMessage(content=_SYSTEM_PROMPT),
                HumanMessage(content=user_content),
            ]
        )
        final_summary: str = response.content
    except Exception as exc:  # pylint: disable=broad-except
        error_msg = f"Synthesis Agent LLM error: {exc}"
        print(f"  ⚠  {error_msg}")
        # Degrade gracefully — return whatever findings we have
        final_summary = (
            f"## Research Summary\n\n"
            f"An error occurred during final synthesis: {exc}\n\n"
            f"### Raw Findings\n\n{findings}"
        )
        return {
            "final_summary": final_summary,
            "error": error_msg,
            "messages": [AIMessage(content=final_summary)],
        }

    print("  ✔  Final report generated successfully.")

    return {
        "final_summary": final_summary,
        "messages": [AIMessage(content=final_summary)],
    }
