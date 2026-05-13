"""
main.py
-------
Entry point for the Multi-Agent Business Research Assistant.

This module runs an interactive CLI loop that:
1. Accepts a free-text query from the user.
2. Streams the query through the compiled LangGraph research graph.
3. Detects interrupt signals (human clarification required) and prompts
   the user inline, then resumes graph execution via update_state().
4. Prints the final Markdown summary when the graph reaches END.
5. Maintains a single thread_id for the entire session so conversation
   history (stored in MemorySaver) persists across turns.

Usage
-----
    python main.py

Keyboard interrupt (Ctrl+C) exits the loop cleanly.
"""

import sys
import os
from typing import Any

# ── Force UTF-8 on Windows (cp1252 can't encode box-drawing chars / emoji) ──
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:
        pass  # Python < 3.7 — best-effort

from dotenv import load_dotenv

# Load environment variables before importing graph (LLM/tool keys needed)
load_dotenv()

from graph import research_graph   # noqa: E402  (must come after load_dotenv)
from state import ResearchState    # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# A single thread ID keeps all turns in the same MemorySaver checkpoint,
# enabling full multi-turn conversation context across every agent call.
THREAD_ID = "session_001"

# recursion_limit: each turn can traverse at most 25 nodes.
# With 3 research retries the worst-case path is:
#   clarity(1) -> research(3x) -> validator(3x) -> synthesis(1) = ~10 nodes
# 25 gives comfortable headroom without letting true infinite loops run wild.
CONFIG = {
    "configurable": {"thread_id": THREAD_ID},
    "recursion_limit": 25,
}

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════════╗
║       🔍  Multi-Agent Business Research Assistant  🔍        ║
║                                                              ║
║  Ask about any company: financials, news, leadership, etc.  ║
║  Type  'exit' or press Ctrl+C to quit.                       ║
╚══════════════════════════════════════════════════════════════╝
"""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _print_separator(char: str = "─", width: int = 60) -> None:
    """Print a full-width separator line to stdout.

    Parameters
    ----------
    char : str
        Character to repeat.
    width : int
        Total line width.
    """
    print(char * width)


def _stream_graph(query: str) -> None:
    """Feed a user query into the graph and handle the stream output.

    This function drives the entire agent pipeline for one user turn:
    1. Streams events from graph.stream().
    2. On interrupt: prompts for clarification and resumes the graph.
    3. On END: prints the final summary.

    The loop inside handles the nested "stream → interrupt → resume →
    stream" pattern that LangGraph uses for human-in-the-loop flows.

    Parameters
    ----------
    query : str
        The raw user input string for this turn.

    Side Effects
    ------------
    - Prints agent progress banners and the final summary to stdout.
    - Modifies graph state via update_state() when an interrupt occurs.
    """
    # ── Build the initial state payload for this turn ────────────────────
    # We only provide `query` and reset `research_attempts`; MemorySaver
    # will merge this with prior turns' message history automatically.
    initial_state: dict[str, Any] = {
        "query": query,
        "research_attempts": 0,    # Reset retry counter for each new query
        "clarified_query": None,
        "clarity_status": None,
        "clarification_question": None,
        "research_findings": None,
        "confidence_score": None,
        "validation_result": None,
        "final_summary": None,
        "error": None,
        "messages": [],            # operator.add reducer will append to history
    }

    # ── Stream loop ──────────────────────────────────────────────────────
    # We use a while-True loop to support the interrupt/resume cycle:
    # after an interrupt, we update state and stream again from the
    # current checkpoint position.
    streaming_input: Any = initial_state
    first_run: bool = True

    while True:
        interrupted: bool = False

        try:
            # graph.stream() yields (node_name, state_update) tuples for
            # every node that executes.  We use stream_mode="updates"
            # (default) which gives us per-node output dicts.
            for event in research_graph.stream(
                streaming_input if first_run else None,
                config=CONFIG,
                stream_mode="updates",
            ):
                first_run = False  # After first chunk, pass None to resume

                # Normal events are {node_name: state_update_dict}.
                # interrupt_before emits {"__interrupt__": (Interrupt(...),)}
                # where the value is a TUPLE — detect it here so we do not
                # pass it to _handle_node_output which expects a dict.
                for node_name, node_output in event.items():
                    if node_name == "__interrupt__":
                        interrupted = True   # graph paused; handle below
                    else:
                        _handle_node_output(node_name, node_output)

        except Exception as exc:
            exc_type = type(exc).__name__

            # LangGraph raises GraphRecursionError when the node traversal
            # count exceeds recursion_limit.  Surface it clearly instead of
            # silently dropping it.
            if "GraphRecursionError" in exc_type or "recursion" in str(exc).lower():
                print(
                    "\n  [!] Graph recursion limit hit. "
                    "This usually means the retry loop ran too many times. "
                    "Falling through to synthesis with available data."
                )
                return

            # LangGraph raises a specific Interrupt exception (not a subclass
            # of GraphInterrupt in all versions) when interrupt() is called.
            # Detect it by name so we don't depend on a specific import path.
            if (
                "interrupt" in exc_type.lower()
                or "graphinterrupt" in exc_type.lower()
                or hasattr(exc, "value")
            ):
                interrupted = True
            else:
                print(f"\n  [!] Unexpected error during streaming: {exc_type}: {exc}")
                return

        # ── Check for interrupt via graph state ───────────────────────
        graph_state = research_graph.get_state(CONFIG)
        is_interrupted = (
            graph_state.next
            and "human_clarification" in graph_state.next
        )

        if is_interrupted or interrupted:
            # Retrieve the clarification question from state
            current_values = graph_state.values
            question = current_values.get(
                "clarification_question",
                "Could you please provide more details?"
            )

            _print_separator()
            print(f"\n🤖 Assistant: {question}")
            _print_separator()

            # Prompt the user for their clarification answer
            try:
                user_answer = input("You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\nGoodbye!")
                sys.exit(0)

            if user_answer.lower() in ("exit", "quit", "q"):
                print("\nGoodbye!")
                sys.exit(0)

            # Inject the user's answer back into the graph state so the
            # human_clarification node can read it when execution resumes.
            research_graph.update_state(
                CONFIG,
                {
                    "clarified_query": user_answer,
                    # Append the user's answer to conversation history
                    "messages": [],  # Will be handled by the node itself
                },
                as_node="human_clarification",
            )

            # Set streaming_input to None so the next stream() call
            # resumes from the current checkpoint (not re-initialised).
            streaming_input = None
            first_run = False
            continue   # Re-enter the stream loop to resume execution

        # No interrupt — graph has reached END (or errored); break out.
        break

    # ── Print final summary ───────────────────────────────────────────────
    final_state = research_graph.get_state(CONFIG).values
    final_summary = final_state.get("final_summary")

    if final_summary:
        print("\n")
        _print_separator("═")
        print("🎯  RESEARCH SUMMARY")
        _print_separator("═")
        print(final_summary)
        _print_separator("═")
    else:
        error = final_state.get("error")
        if error:
            print(f"\n⚠  Research encountered an error: {error}")
        else:
            print("\n  (No summary was generated.)")


def _handle_node_output(node_name: str, output: dict) -> None:
    """Process and display output from a single graph node.

    Called once per node event emitted by graph.stream().  Prints
    minimal progress info so the terminal doesn't become overwhelming
    while still showing the user something is happening.

    Parameters
    ----------
    node_name : str
        The registered name of the node that just executed.
    output : dict | tuple
        The partial state update dict from the node, OR a tuple of
        Interrupt objects when LangGraph fires interrupt_before.
        The __interrupt__ key is handled in _stream_graph before this
        function is called, so this should always be a dict in practice.

    Side Effects
    ------------
    Prints to stdout.
    """
    display_names = {
        "clarity": "Clarity Agent",
        "human_clarification": "Human Clarification",
        "research": "Research Agent",
        "validator": "Validator Agent",
        "synthesis": "Synthesis Agent",
    }

    friendly = display_names.get(node_name, node_name.replace("_", " ").title())

    # Guard: only inspect dict outputs — internal LangGraph events
    # (__start__, __interrupt__, etc.) may carry non-dict payloads.
    if not isinstance(output, dict):
        return

    # Surface any agent-level errors inline
    if output.get("error"):
        print(f"  [!] [{friendly}] Error: {output['error']}")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Run the interactive CLI loop for the research assistant.

    Reads user input, drives the graph, and handles graceful exit on
    KeyboardInterrupt or 'exit' command.

    Side Effects
    ------------
    Writes to stdout and reads from stdin interactively.
    """
    print(WELCOME_BANNER)

    # Quick pre-flight check for required API keys
    missing_keys = []
    if not os.getenv("GITHUB_TOKEN"):
        missing_keys.append("GITHUB_TOKEN")
    if not os.getenv("TAVILY_API_KEY"):
        missing_keys.append("TAVILY_API_KEY")

    if missing_keys:
        print(
            f"⚠  Missing environment variables: {', '.join(missing_keys)}\n"
            "   Please add them to your .env file and restart.\n"
        )
        sys.exit(1)

    print("  Session ID :", THREAD_ID)
    print("  Conversation history persists across all turns in this session.\n")

    while True:
        _print_separator()
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\nGoodbye! 👋")
            break

        if not user_input:
            print("  (Please enter a query to get started.)")
            continue

        if user_input.lower() in ("exit", "quit", "q"):
            print("\nGoodbye! 👋")
            break

        print()   # Visual spacing before agent output
        _stream_graph(user_input)
        print()   # Visual spacing after summary


if __name__ == "__main__":
    main()
