"""
utils/llm_factory.py
---------------------
Centralised LLM factory for the research pipeline.

All agents import ``get_llm()`` from here instead of constructing
their own ``ChatOpenAI`` instances.  This means the GitHub Models
endpoint URL and token are configured in exactly one place — easy to
swap back to the standard OpenAI API or any other provider later.

GitHub Models endpoint details
-------------------------------
- Base URL : https://models.github.ai/inference
- Auth     : GitHub Personal Access Token with ``models:read`` permission
             Set GITHUB_TOKEN in your .env file.
             Create at: https://github.com/settings/tokens
             Fine-grained token -> Repository permissions -> Models -> Read
- Model ID : openai/gpt-4o   (provider namespace prefix is required)
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

# Official GitHub Models inference endpoint (updated May 2025)
GITHUB_MODELS_BASE_URL = "https://models.github.ai/inference"

# Model name MUST include the provider namespace prefix
GITHUB_MODEL_NAME = "openai/gpt-4o"


@lru_cache(maxsize=4)
def get_llm(temperature: float = 0.0) -> ChatOpenAI:
    """Return a ``ChatOpenAI`` instance pointed at the GitHub Models API.

    Uses ``lru_cache`` so that agents sharing the same temperature value
    reuse the same client object (avoids redundant instantiation).

    Parameters
    ----------
    temperature : float
        Sampling temperature (0 = deterministic, 1 = creative).
        Default is 0.0 for predictable structured outputs.

    Returns
    -------
    ChatOpenAI
        Configured LLM client ready to call GitHub-hosted GPT-4o.

    Raises
    ------
    EnvironmentError
        If ``GITHUB_TOKEN`` is not set in the environment.
    """
    token = os.getenv("GITHUB_TOKEN")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN is not set. "
            "Create a fine-grained PAT at https://github.com/settings/tokens "
            "with 'Models: Read' permission and add it to your .env file."
        )

    return ChatOpenAI(
        model=GITHUB_MODEL_NAME,
        temperature=temperature,
        base_url=GITHUB_MODELS_BASE_URL,
        api_key=token,          # GitHub PAT used as the bearer token
    )
