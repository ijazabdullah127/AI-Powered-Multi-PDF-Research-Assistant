"""
utils/helpers.py
----------------
Shared utility functions used across multiple agent modules.

Responsibilities
----------------
- ``build_history_text``  — serialise ``state["messages"]`` into a
  plain-text block suitable for injection into LLM system/user prompts.
- ``format_agent_header`` — produce a consistent CLI progress banner so
  the user can see which agent is currently running.
- ``extract_json_block``  — safely parse a JSON object from an LLM
  response string that may contain surrounding prose or markdown fences.
"""

import json
import re
from typing import Any, Dict, List

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


def build_history_text(messages: List[BaseMessage]) -> str:
    """Convert a list of LangChain messages into a readable text block.

    Each message is rendered as ``[Role]: <content>`` on its own line.
    This text is injected verbatim into LLM prompts so that every agent
    has full awareness of the conversation so far.

    Parameters
    ----------
    messages : List[BaseMessage]
        The ``state["messages"]`` list from the current ResearchState.

    Returns
    -------
    str
        Multi-line string with one line per message, or ``"(No prior
        conversation.)"`` when the list is empty.

    Side Effects
    ------------
    None.
    """
    if not messages:
        return "(No prior conversation.)"

    lines: List[str] = []
    for msg in messages:
        # Determine role label based on message type
        if isinstance(msg, HumanMessage):
            role = "User"
        elif isinstance(msg, AIMessage):
            role = "Assistant"
        else:
            # Covers SystemMessage, ToolMessage, etc.
            role = type(msg).__name__.replace("Message", "")

        # Truncate very long messages to avoid prompt overflow
        content = str(msg.content)
        if len(content) > 2000:
            content = content[:2000] + " ... [truncated]"

        lines.append(f"[{role}]: {content}")

    return "\n".join(lines)


def format_agent_header(agent_name: str, action: str) -> str:
    """Return a formatted CLI progress banner for the given agent.

    Example output::

        ──────────────────────────────────────────
        [Clarity Agent]: Evaluating query clarity…
        ──────────────────────────────────────────

    Parameters
    ----------
    agent_name : str
        Human-readable agent name, e.g. ``"Clarity Agent"``.
    action : str
        Short description of what the agent is doing right now.

    Returns
    -------
    str
        Multi-line banner string ready to be printed to stdout.

    Side Effects
    ------------
    None.
    """
    separator = "─" * 50
    return f"\n{separator}\n[{agent_name}]: {action}\n{separator}"


def extract_json_block(text: str) -> Dict[str, Any]:
    """Extract and parse the first JSON object found in ``text``.

    LLMs sometimes wrap JSON in markdown fences (```json … ```) or add
    explanatory prose before/after the object.  This function handles
    all common cases robustly.

    Strategy
    --------
    1. Strip markdown code fences (``` or ```json).
    2. Try to find the first ``{…}`` balanced block via regex.
    3. Fall back to parsing the entire stripped text.
    4. Return an empty dict on any failure — callers must handle this.

    Parameters
    ----------
    text : str
        Raw string output from an LLM invocation.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object, or ``{}`` if parsing fails entirely.

    Side Effects
    ------------
    None.
    """
    if not text:
        return {}

    # Step 1 — Remove markdown fences (```json … ``` or ``` … ```)
    cleaned = re.sub(r"```(?:json)?\s*", "", text).replace("```", "").strip()

    # Step 2 — Attempt to extract the first balanced JSON object
    # by finding the outermost { … } block.
    brace_match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            pass  # Fall through to step 3

    # Step 3 — Try to parse the whole cleaned text as-is
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Step 4 — Give up gracefully; caller receives empty dict
        return {}
