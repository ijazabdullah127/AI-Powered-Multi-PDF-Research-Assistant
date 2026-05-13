"""
test_suite.py
-------------
Automated test suite for the Multi-Agent Business Research Assistant.

Tests covered
-------------
1.  Graph compilation & node registration
2.  Routing functions (unit tests — pure logic, no LLM calls)
3.  State schema integrity
4.  JSON extraction utility
5.  Conversation history builder
6.  Live API connectivity (GitHub Models GPT-4o)
7.  Clarity Agent — clear query
8.  Clarity Agent — ambiguous query (needs_clarification)
9.  Research Agent — live Tavily + LLM synthesis
10. Validator Agent — sufficient research
11. Synthesis Agent — markdown report generation
12. Full pipeline — end-to-end (clear query, no interrupt)
13. Multi-turn context — follow-up resolves company from history

Run with:
    python test_suite.py
"""

import os
import sys
import time
import traceback
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv("../.env")  # Load from parent dir where .env lives

# ── Colour helpers (Windows-safe fallback) ────────────────────────────────
try:
    import colorama
    colorama.init()
    GREEN  = "\033[92m"
    RED    = "\033[91m"
    YELLOW = "\033[93m"
    CYAN   = "\033[96m"
    RESET  = "\033[0m"
    BOLD   = "\033[1m"
except ImportError:
    GREEN = RED = YELLOW = CYAN = RESET = BOLD = ""


# ── Test runner bookkeeping ───────────────────────────────────────────────
_results: List[Tuple[str, bool, str]] = []


def run_test(name: str, fn) -> bool:
    """Execute a single test function and record the result."""
    print(f"\n{CYAN}  [ TEST ] {name}{RESET}")
    try:
        fn()
        print(f"  {GREEN}PASS{RESET}")
        _results.append((name, True, ""))
        return True
    except AssertionError as e:
        msg = str(e) or "Assertion failed"
        print(f"  {RED}FAIL{RESET}  {msg}")
        _results.append((name, False, msg))
        return False
    except Exception as e:
        msg = f"{type(e).__name__}: {e}"
        print(f"  {RED}ERROR{RESET} {msg}")
        traceback.print_exc()
        _results.append((name, False, msg))
        return False


def section(title: str):
    print(f"\n{BOLD}{YELLOW}{'='*60}{RESET}")
    print(f"{BOLD}{YELLOW}  {title}{RESET}")
    print(f"{BOLD}{YELLOW}{'='*60}{RESET}")


def print_summary():
    passed = sum(1 for _, ok, _ in _results if ok)
    total  = len(_results)
    print(f"\n{BOLD}{'='*60}")
    print(f"  RESULTS: {passed}/{total} tests passed")
    print(f"{'='*60}{RESET}")
    for name, ok, msg in _results:
        icon = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {icon}  {name}")
        if msg:
            print(f"       {RED}↳ {msg}{RESET}")
    print()
    return passed == total


# =============================================================================
# SECTION 1 — Infrastructure
# =============================================================================
section("1. Infrastructure & Graph Compilation")


def test_graph_compiles():
    """Graph must compile with all 5 expected nodes."""
    from graph import research_graph
    nodes = list(research_graph.nodes.keys())
    for expected in ["clarity", "human_clarification", "research", "validator", "synthesis"]:
        assert expected in nodes, f"Missing node: {expected}"


def test_state_schema():
    """ResearchState TypedDict must have all required fields."""
    from state import ResearchState
    import typing
    hints = typing.get_type_hints(ResearchState)
    required = [
        "messages", "query", "clarified_query", "clarity_status",
        "clarification_question", "research_findings", "confidence_score",
        "validation_result", "research_attempts", "final_summary", "error",
    ]
    for field in required:
        assert field in hints, f"Missing field in ResearchState: {field}"


def test_env_keys_present():
    """Both GITHUB_TOKEN and TAVILY_API_KEY must be set."""
    token  = os.getenv("GITHUB_TOKEN")
    tavily = os.getenv("TAVILY_API_KEY")
    assert token  and token  != "your_github_pat_here",  "GITHUB_TOKEN not set"
    assert tavily and tavily != "your_tavily_key_here", "TAVILY_API_KEY not set"


run_test("Graph compiles with all 5 nodes",           test_graph_compiles)
run_test("ResearchState has all required fields",     test_state_schema)
run_test("API keys present in environment",           test_env_keys_present)


# =============================================================================
# SECTION 2 — Routing Logic (unit tests, no LLM)
# =============================================================================
section("2. Routing Function Unit Tests")


def test_route_clarity_clear():
    from graph import route_after_clarity
    state = {"clarity_status": "clear", "messages": [], "query": "Tesla", "research_attempts": 0}
    assert route_after_clarity(state) == "research"


def test_route_clarity_needs_clarification():
    from graph import route_after_clarity
    state = {"clarity_status": "needs_clarification", "messages": [], "query": "a company", "research_attempts": 0}
    assert route_after_clarity(state) == "human_clarification"


def test_route_research_high_confidence():
    """confidence >= 6 must go directly to synthesis (skipping validator)."""
    from graph import route_after_research
    state = {"confidence_score": 7.5, "research_attempts": 1}
    assert route_after_research(state) == "synthesis", \
        "High confidence should bypass validator and go to synthesis"


def test_route_research_low_confidence():
    """confidence < 6 must go to validator."""
    from graph import route_after_research
    state = {"confidence_score": 4.2, "research_attempts": 1}
    assert route_after_research(state) == "validator"


def test_route_research_exactly_6():
    """confidence == 6.0 is the boundary — should go to synthesis."""
    from graph import route_after_research
    state = {"confidence_score": 6.0, "research_attempts": 1}
    assert route_after_research(state) == "synthesis"


def test_route_validator_sufficient():
    from graph import route_after_validation
    state = {"validation_result": "sufficient", "research_attempts": 1}
    assert route_after_validation(state) == "synthesis"


def test_route_validator_max_attempts():
    """After 3 attempts, always synthesise regardless of validation result."""
    from graph import route_after_validation
    state = {"validation_result": "insufficient", "research_attempts": 3}
    assert route_after_validation(state) == "synthesis", \
        "Should synthesise after max retries even if insufficient"


def test_route_validator_retry():
    from graph import route_after_validation
    state = {"validation_result": "insufficient", "research_attempts": 1}
    assert route_after_validation(state) == "research"


run_test("route_after_clarity — clear query",              test_route_clarity_clear)
run_test("route_after_clarity — needs_clarification",      test_route_clarity_needs_clarification)
run_test("route_after_research — confidence >= 6 → synthesis", test_route_research_high_confidence)
run_test("route_after_research — confidence < 6 → validator",  test_route_research_low_confidence)
run_test("route_after_research — exactly 6.0 boundary",        test_route_research_exactly_6)
run_test("route_after_validation — sufficient",            test_route_validator_sufficient)
run_test("route_after_validation — max attempts exhausted",test_route_validator_max_attempts)
run_test("route_after_validation — retry (insufficient)",  test_route_validator_retry)


# =============================================================================
# SECTION 3 — Utility Functions
# =============================================================================
section("3. Utility Functions")


def test_extract_json_clean():
    from utils.helpers import extract_json_block
    result = extract_json_block('{"clarity_status": "clear"}')
    assert result.get("clarity_status") == "clear"


def test_extract_json_with_fence():
    from utils.helpers import extract_json_block
    text = '```json\n{"confidence_score": 7.5}\n```'
    result = extract_json_block(text)
    assert result.get("confidence_score") == 7.5


def test_extract_json_with_prose():
    from utils.helpers import extract_json_block
    text = 'Here is the result: {"validation_result": "sufficient", "reason": "good"} — done.'
    result = extract_json_block(text)
    assert result.get("validation_result") == "sufficient"


def test_extract_json_invalid_returns_empty():
    from utils.helpers import extract_json_block
    result = extract_json_block("This is not JSON at all!")
    assert result == {}, f"Expected empty dict, got {result}"


def test_build_history_empty():
    from utils.helpers import build_history_text
    result = build_history_text([])
    assert "No prior" in result


def test_build_history_with_messages():
    from langchain_core.messages import HumanMessage, AIMessage
    from utils.helpers import build_history_text
    msgs = [HumanMessage(content="Tell me about Tesla"), AIMessage(content="Tesla is...")]
    result = build_history_text(msgs)
    assert "[User]" in result and "Tesla" in result
    assert "[Assistant]" in result


run_test("extract_json_block — clean JSON",           test_extract_json_clean)
run_test("extract_json_block — markdown fenced",      test_extract_json_with_fence)
run_test("extract_json_block — prose-wrapped JSON",   test_extract_json_with_prose)
run_test("extract_json_block — invalid → empty dict", test_extract_json_invalid_returns_empty)
run_test("build_history_text — empty list",           test_build_history_empty)
run_test("build_history_text — with messages",        test_build_history_with_messages)


# =============================================================================
# SECTION 4 — Live API Tests
# =============================================================================
section("4. Live API Connectivity")


def test_llm_api():
    """GitHub Models GPT-4o must respond to a basic prompt."""
    from utils.llm_factory import get_llm
    llm = get_llm(temperature=0.0)
    resp = llm.invoke("Reply with just the word PONG.")
    assert resp.content and len(resp.content) > 0, "Empty LLM response"
    print(f"    LLM response: {resp.content.strip()!r}")


def test_tavily_search():
    """Tavily must return at least one result for a known company."""
    from tools.search import run_search
    result = run_search("Apple Inc latest news 2025")
    assert "[Search Error]" not in result, f"Search failed: {result[:200]}"
    assert len(result) > 100, "Search result too short"
    print(f"    Search returned {len(result)} chars")


run_test("GitHub Models GPT-4o API responds",  test_llm_api)
run_test("Tavily search returns results",       test_tavily_search)


# =============================================================================
# SECTION 5 — Individual Agent Tests
# =============================================================================
section("5. Individual Agent Node Tests")


def _base_state(**overrides) -> dict:
    """Build a minimal valid ResearchState for testing."""
    base = {
        "messages":               [],
        "query":                  "Tell me about Tesla",
        "clarified_query":        None,
        "clarity_status":         None,
        "clarification_question": None,
        "research_findings":      None,
        "confidence_score":       None,
        "validation_result":      None,
        "research_attempts":      0,
        "final_summary":          None,
        "error":                  None,
    }
    base.update(overrides)
    return base


def test_clarity_agent_clear():
    """Clarity Agent must return 'clear' for a named company query."""
    from agents.clarity_agent import clarity_agent_node
    state  = _base_state(query="Tell me about Apple Inc financials")
    result = clarity_agent_node(state)
    assert result.get("clarity_status") == "clear", \
        f"Expected 'clear', got: {result.get('clarity_status')}"
    assert len(result.get("messages", [])) == 1, "Should append exactly one AIMessage"


def test_clarity_agent_ambiguous():
    """Clarity Agent must return 'needs_clarification' for a vague query."""
    from agents.clarity_agent import clarity_agent_node
    state  = _base_state(query="Tell me about the company")
    result = clarity_agent_node(state)
    assert result.get("clarity_status") == "needs_clarification", \
        f"Expected 'needs_clarification', got: {result.get('clarity_status')}"
    assert result.get("clarification_question"), "Must provide a clarification question"
    print(f"    Question: {result['clarification_question']!r}")


def test_research_agent_runs():
    """Research Agent must populate research_findings and confidence_score."""
    from agents.research_agent import research_agent_node
    state  = _base_state(query="Tesla financial results 2024")
    result = research_agent_node(state)
    assert result.get("research_findings"),    "research_findings must be populated"
    assert result.get("confidence_score") is not None, "confidence_score must be set"
    assert result.get("research_attempts") == 1, "research_attempts must be incremented"
    score = result["confidence_score"]
    assert 0.0 <= score <= 10.0, f"confidence_score out of range: {score}"
    print(f"    confidence_score = {score:.1f}")
    print(f"    findings preview: {result['research_findings'][:100]}...")


def test_validator_agent_runs():
    """Validator Agent must return 'sufficient' or 'insufficient'."""
    from agents.validator_agent import validator_agent_node
    state = _base_state(
        query="Tell me about Tesla",
        research_findings="Tesla reported record revenue of $97.69B in FY2023. CEO Elon Musk announced new Gigafactories.",
        research_attempts=1,
    )
    result = validator_agent_node(state)
    assert result.get("validation_result") in ("sufficient", "insufficient"), \
        f"Unexpected validation_result: {result.get('validation_result')}"
    assert len(result.get("messages", [])) == 1
    print(f"    validation_result = {result['validation_result']}")


def test_synthesis_agent_runs():
    """Synthesis Agent must produce a non-empty markdown final_summary."""
    from agents.synthesis_agent import synthesis_agent_node
    state = _base_state(
        query="Tell me about Tesla",
        research_findings=(
            "Tesla Inc. is an EV and clean energy company. FY2024 revenue: $97.7B. "
            "CEO Elon Musk. Cybertruck launched in 2024. Full Self-Driving v12 released. "
            "Gigafactories in Texas, Nevada, Berlin, Shanghai operational."
        ),
        confidence_score=8.0,
    )
    result = synthesis_agent_node(state)
    summary = result.get("final_summary", "")
    assert summary and len(summary) > 100, "final_summary too short or empty"
    assert "##" in summary, "Summary should contain markdown headers"
    assert len(result.get("messages", [])) == 1
    print(f"    Summary length: {len(summary)} chars")
    print(f"    First 200 chars: {summary[:200]}...")


run_test("Clarity Agent — clear company query",   test_clarity_agent_clear)
run_test("Clarity Agent — ambiguous query",        test_clarity_agent_ambiguous)
run_test("Research Agent — live search + LLM",     test_research_agent_runs)
run_test("Validator Agent — quality assessment",   test_validator_agent_runs)
run_test("Synthesis Agent — markdown report",      test_synthesis_agent_runs)


# =============================================================================
# SECTION 6 — End-to-End Pipeline Tests
# =============================================================================
section("6. End-to-End Pipeline Tests")


def test_e2e_clear_query():
    """Full pipeline must run to END for a clear, named-company query."""
    from graph import build_graph
    from langgraph.checkpoint.memory import MemorySaver

    graph  = build_graph()
    config = {"configurable": {"thread_id": "test_e2e_clear"}, "recursion_limit": 25}

    initial = {
        "messages":               [],
        "query":                  "Tell me about Microsoft Corporation",
        "clarified_query":        None,
        "clarity_status":         None,
        "clarification_question": None,
        "research_findings":      None,
        "confidence_score":       None,
        "validation_result":      None,
        "research_attempts":      0,
        "final_summary":          None,
        "error":                  None,
    }

    nodes_visited = []
    for event in graph.stream(initial, config=config, stream_mode="updates"):
        for node_name in event.keys():
            if node_name not in ("__interrupt__",):
                nodes_visited.append(node_name)

    print(f"    Nodes visited: {nodes_visited}")

    # Verify the pipeline reached synthesis
    assert "synthesis" in nodes_visited, f"synthesis not reached. Visited: {nodes_visited}"
    assert "clarity"   in nodes_visited, "clarity node should have run"
    assert "research"  in nodes_visited, "research node should have run"

    # Verify final state
    final = graph.get_state(config).values
    assert final.get("final_summary"), "final_summary must be set"
    assert final.get("research_findings"), "research_findings must be set"
    print(f"    Final summary length: {len(final['final_summary'])} chars")
    print(f"    Confidence: {final.get('confidence_score'):.1f}")


def test_e2e_multi_turn_followup():
    """Follow-up query must resolve company from prior conversation history."""
    from graph import build_graph
    from langchain_core.messages import HumanMessage, AIMessage

    graph  = build_graph()
    config = {"configurable": {"thread_id": "test_multiturn"}, "recursion_limit": 25}

    # --- Turn 1: Establish Tesla ---
    turn1 = {
        "messages":               [HumanMessage(content="Tell me about Tesla")],
        "query":                  "Tell me about Tesla",
        "clarified_query":        None,
        "clarity_status":         None,
        "clarification_question": None,
        "research_findings":      None,
        "confidence_score":       None,
        "validation_result":      None,
        "research_attempts":      0,
        "final_summary":          None,
        "error":                  None,
    }
    for event in graph.stream(turn1, config=config, stream_mode="updates"):
        pass  # consume stream

    turn1_state = graph.get_state(config).values
    print(f"    Turn 1 clarity_status: {turn1_state.get('clarity_status')}")
    assert turn1_state.get("clarity_status") == "clear", \
        "'Tell me about Tesla' should be clear"

    # --- Turn 2: Follow-up referencing prior context ---
    turn2 = {
        "messages":               [],
        "query":                  "What about their CEO?",
        "clarified_query":        None,
        "clarity_status":         None,
        "clarification_question": None,
        "research_findings":      None,
        "confidence_score":       None,
        "validation_result":      None,
        "research_attempts":      0,
        "final_summary":          None,
        "error":                  None,
    }
    for event in graph.stream(turn2, config=config, stream_mode="updates"):
        pass  # consume stream

    turn2_state = graph.get_state(config).values
    print(f"    Turn 2 clarity_status: {turn2_state.get('clarity_status')}")
    # The follow-up should be marked clear because Tesla is in the message history
    assert turn2_state.get("clarity_status") == "clear", \
        "Follow-up 'What about their CEO?' should be clear given Tesla context in history"
    assert turn2_state.get("final_summary"), "Should have produced a summary for turn 2"
    print(f"    Turn 2 summary preview: {turn2_state['final_summary'][:150]}...")


run_test("E2E — clear company query reaches synthesis", test_e2e_clear_query)
run_test("E2E — multi-turn follow-up uses history",     test_e2e_multi_turn_followup)


# =============================================================================
# SUMMARY
# =============================================================================
print()
all_passed = print_summary()
sys.exit(0 if all_passed else 1)
