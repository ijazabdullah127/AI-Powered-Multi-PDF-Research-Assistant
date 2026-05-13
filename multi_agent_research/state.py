"""
state.py
--------
Defines the shared ResearchState TypedDict that flows through every node
in the LangGraph graph.  All agents read from and write back to this
single, immutable-snapshot state object; LangGraph merges updates via
the Annotated reducer declared on the `messages` field.

Key design decisions
--------------------
- `messages` uses `operator.add` as its reducer so that each agent can
  *append* new messages without overwriting history from prior nodes.
- All optional fields default to None / 0 so that the graph can be
  initialised with only `query` and `messages` populated.
- `research_attempts` starts at 0 and is incremented by the Research
  Agent; the Validator routing function caps retries at 3.
"""

import operator
from typing import Annotated, List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class ResearchState(TypedDict):
    """Shared state passed between every node in the research graph.

    Fields
    ------
    messages : List[BaseMessage]
        Full conversation history (HumanMessage + AIMessage).  New
        messages are *appended* via the `operator.add` reducer, never
        replaced.
    query : str
        The raw user query for the current turn.
    clarified_query : Optional[str]
        The refined query produced after the human clarification
        interrupt (or None if the original query was already clear).
    clarity_status : Optional[str]
        Set by the Clarity Agent to either ``"clear"`` or
        ``"needs_clarification"``.
    clarification_question : Optional[str]
        The question the Clarity Agent wants to ask the user.  Only
        populated when ``clarity_status == "needs_clarification"``.
    research_findings : Optional[str]
        Raw research narrative produced by the Research Agent after
        synthesising Tavily search results.
    confidence_score : Optional[float]
        Self-assessed quality score (0–10) assigned by the Research
        Agent.  Scores > 6 indicate reliable, recent findings.
    validation_result : Optional[str]
        Set by the Validator Agent to either ``"sufficient"`` or
        ``"insufficient"``.
    research_attempts : int
        Counts how many times the Research Agent has run this turn.
        Initialised to 0; capped at 3 by the routing logic.
    final_summary : Optional[str]
        Markdown-formatted final answer produced by the Synthesis Agent.
    error : Optional[str]
        Stores a human-readable description of any runtime error so
        that the graph can surface it to the user instead of crashing.
    """

    # ---------- conversation history (append-only via reducer) ----------
    messages: Annotated[List[BaseMessage], operator.add]

    # ---------- query lifecycle ----------
    query: str
    clarified_query: Optional[str]

    # ---------- clarity agent outputs ----------
    clarity_status: Optional[str]
    clarification_question: Optional[str]

    # ---------- research agent outputs ----------
    research_findings: Optional[str]
    confidence_score: Optional[float]

    # ---------- validator agent outputs ----------
    validation_result: Optional[str]

    # ---------- retry counter (guarded at max 3 in routing) ----------
    research_attempts: int

    # ---------- synthesis agent output (terminal) ----------
    final_summary: Optional[str]

    # ---------- error reporting ----------
    error: Optional[str]
