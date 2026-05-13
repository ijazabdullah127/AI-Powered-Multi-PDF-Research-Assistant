# 🔍 Multi-Agent Business Research Assistant

A production-grade multi-agent AI pipeline built with **LangGraph** and **LangChain** that researches any publicly traded or well-known company on demand.

---

## Architecture

```
START
  └──> [Clarity Agent]
              ├── needs_clarification ──> [Human Clarification] ──> [Clarity Agent]
              └── clear ──> [Research Agent]
                                ├── confidence >= 6 ──> [Synthesis Agent] ──> END
                                └── confidence <  6 ──> [Validator Agent]
                                          ├── sufficient OR attempts >= 3 ──> [Synthesis Agent] ──> END
                                          └── insufficient AND attempts <  3 ──> [Research Agent]
```

| Agent | Role |
|---|---|
| **Clarity Agent** | Determines if the query has a clear company name and intent. Uses full conversation history to resolve follow-ups like "What about their CEO?" |
| **Research Agent** | Runs a Tavily web search and synthesises findings with a self-assessed confidence score (0–10). |
| **Validator Agent** | Quality-gates the research on completeness, relevance, and recency. Routes to retry or synthesis. |
| **Synthesis Agent** | Generates the final structured Markdown report. Terminal node. |

---

## Project Structure

```
multi_agent_research/
├── main.py                  # Entry point — interactive CLI loop
├── graph.py                 # LangGraph graph construction & compilation
├── state.py                 # Shared ResearchState TypedDict schema
├── agents/
│   ├── __init__.py
│   ├── clarity_agent.py     # Query clarity evaluation
│   ├── research_agent.py    # Tavily search + LLM synthesis
│   ├── validator_agent.py   # Research quality gate
│   └── synthesis_agent.py   # Final Markdown report generation
├── tools/
│   ├── __init__.py
│   └── search.py            # Tavily search wrapper (max_results=5)
├── utils/
│   ├── __init__.py
│   └── helpers.py           # History builder, JSON extractor, CLI formatter
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone / navigate to the project

```bash
cd multi_agent_research
```

### 2. Create a virtual environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure API keys

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

```env
GITHUB_TOKEN=github_pat_...
TAVILY_API_KEY=tvly-...
```

> **GITHUB_TOKEN** — Create a fine-grained Personal Access Token at https://github.com/settings/tokens  
> &nbsp;&nbsp;&nbsp;&nbsp;Grant **Models: Read** permission (under Repository permissions).  
> **TAVILY_API_KEY** — Get your key at https://app.tavily.com

### 5. Run

```bash
python main.py
```

---

## Example Interactions

### Ambiguous query → clarification interrupt

```
You: Tell me about the company

[Clarity Agent]: needs_clarification
🤖 Assistant: Which company are you referring to? Please provide the company name.
You: Apple Inc

[Research Agent]: Searching… (attempt 1/3)
[Validator Agent]: sufficient
[Synthesis Agent]: Generating final report…

═══════════════════════════════
🎯  RESEARCH SUMMARY
═══════════════════════════════
## 🏢 Company Overview
Apple Inc. is a multinational technology company...
```

### Follow-up question using prior context

```
You: What about their CEO?

[Clarity Agent]: Query is clear (Tesla established in prior turn)
[Research Agent]: Searching "Tesla CEO Elon Musk 2024 2025"…
```

### Low-confidence retry loop

```
[Research Agent]: confidence_score = 4.2  → Validator: insufficient → retry
[Research Agent]: confidence_score = 7.8  → Validator: sufficient → Synthesis
```

---

## Tech Stack

| Component | Version |
|---|---|
| Python | 3.11+ |
| LangGraph | ≥ 0.2.0 |
| LangChain | ≥ 0.3.0 |
| langchain-openai | ≥ 0.2.0 |
| langchain-tavily | ≥ 0.1.0 |
| langchain-community | ≥ 0.3.0 |
| Tavily Python SDK | ≥ 0.3.0 |
| GitHub Models (GPT-4o) | openai/gpt-4o via GitHub Models API |
| python-dotenv | ≥ 1.0.0 |
| Pydantic | v2 |

---

## Key Features

- ✅ **Multi-turn memory** — `MemorySaver` checkpointer keeps full conversation history across all turns in a session
- ✅ **Human-in-the-loop interrupts** — uses `langgraph.types.interrupt()` for clarification pauses
- ✅ **Retry loop** — up to 3 research attempts with quality gating
- ✅ **Confidence scoring** — Research Agent self-assesses result quality
- ✅ **Graceful error handling** — every LLM and tool call is wrapped in try/except; errors surface in state, never crash the graph
- ✅ **Strict JSON parsing** — robust `extract_json_block()` handles markdown fences and prose wrapping
- ✅ **No hardcoded keys** — all secrets via environment variables
