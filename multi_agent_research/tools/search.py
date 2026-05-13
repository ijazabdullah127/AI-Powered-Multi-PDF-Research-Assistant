"""
tools/search.py
---------------
Thin wrapper around Tavily's LangChain integration.

Responsibilities
----------------
1. Instantiate ``TavilySearch`` with ``max_results=5`` using the
   ``TAVILY_API_KEY`` environment variable (never hardcoded).
2. Expose ``run_search(query)`` — a plain string-in / string-out helper
   that formats results into a readable block containing title, URL, and
   snippet for each hit.
3. Handle all exceptions gracefully, returning a descriptive error
   string instead of raising so that the Research Agent can surface the
   failure in state rather than crashing the graph.
"""

import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from langchain_tavily import TavilySearch

# Load .env so the API key is available when this module is imported
load_dotenv()


def _build_tavily_tool() -> TavilySearch:
    """Instantiate the Tavily search tool.

    Returns
    -------
    TavilySearch
        Configured tool object ready to be invoked.

    Raises
    ------
    EnvironmentError
        If ``TAVILY_API_KEY`` is not set in the environment.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "TAVILY_API_KEY is not set. Add it to your .env file."
        )
    # The tool reads the key from the environment automatically; we
    # validate presence explicitly so the error is actionable.
    return TavilySearch(max_results=5)


# Module-level cache — populated lazily on first call to run_search()
# so that importing this module never raises an EnvironmentError.
_tavily_tool: "TavilySearch | None" = None


def _get_tavily_tool() -> TavilySearch:
    """Return the TavilySearch singleton, creating it on first call.

    Lazy initialisation means the module can be imported safely even
    before the .env file is loaded; the error only fires when an actual
    search is attempted.

    Returns
    -------
    TavilySearch
        Configured Tavily tool ready to invoke.

    Raises
    ------
    EnvironmentError
        If ``TAVILY_API_KEY`` is not set.
    """
    global _tavily_tool
    if _tavily_tool is None:
        _tavily_tool = _build_tavily_tool()
    return _tavily_tool


def run_search(query: str) -> str:
    """Execute a Tavily web search and return formatted results.

    Each result is rendered as:

        [1] <title>
            URL     : <url>
            Snippet : <snippet>

    Parameters
    ----------
    query : str
        The search query string (should be specific and descriptive).

    Returns
    -------
    str
        A newline-delimited block of formatted search results, or an
        error message prefixed with ``[Search Error]`` if the call fails.

    Side Effects
    ------------
    None — this is a pure read operation against the Tavily API.
    """
    try:
        # TavilySearch.invoke() returns a dict like:
        # {
        #   "query": "...",
        #   "results": [{"title": ..., "url": ..., "content": ...}, ...]
        # }
        # Older versions returned a bare list — handle both cases.
        raw: Any = _get_tavily_tool().invoke(query)

        if isinstance(raw, dict):
            raw_results: List[Dict[str, Any]] = raw.get("results", [])
        elif isinstance(raw, list):
            raw_results = raw
        else:
            raw_results = []

        if not raw_results:
            return "No results found for the given query."

        lines: List[str] = []
        for idx, item in enumerate(raw_results, start=1):
            title   = item.get("title",   "No title")
            url     = item.get("url",     "No URL")
            snippet = item.get("content", "No snippet available.")
            lines.append(
                f"[{idx}] {title}\n"
                f"    URL     : {url}\n"
                f"    Snippet : {snippet}\n"
            )

        return "\n".join(lines)

    except Exception as exc:  # pylint: disable=broad-except
        # Return a safe error string — never let a tool failure crash
        # the entire graph execution.
        return f"[Search Error] Tavily search failed: {exc}"
