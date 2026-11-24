"""Settings loaded from the environment / .env file.

Every model name and path the project uses lives here, so switching
provider (e.g. Gemini free tier -> Claude Haiku 5.5) is a one-line change in .env.
"""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

# Model used for answers AND the prompt-only baseline. Must be the same for all 3 setups.
ANSWER_MODEL = os.getenv("DATELINE_MODEL", "gemini/gemini-flash-latest")
# Cheaper model for judges (can differ from the answer model).
JUDGE_MODEL = os.getenv("DATELINE_JUDGE_MODEL", ANSWER_MODEL)

# Tried once when ANSWER_MODEL keeps failing (e.g. OpenRouter down): the model's direct API.
# Defaults to Claude Haiku 5.5 on Anthropic's API when ANTHROPIC_API_KEY is set, else none.
FALLBACK_MODEL = os.getenv("DATELINE_FALLBACK_MODEL") or (
    "anthropic/claude-haiku-5-5" if os.getenv("ANTHROPIC_API_KEY") else None
)
# Reasoning ("thinking") setting sent with OpenRouter calls. OpenRouter turns Claude Haiku 5.5's adaptive
# thinking on by default; its tokens count against max_tokens, so a capped call can end with an empty
# reply, and they add seconds to every answer. "none" turns thinking off; an empty value sends nothing.
REASONING_EFFORT = os.getenv("DATELINE_REASONING_EFFORT", "none")
# Seconds before one model call is abandoned (batch default; interactive callers pass less).
CALL_TIMEOUT_S = 60

# Seconds between LLM calls. 4.2s keeps us under ~15 requests/minute on the Gemini free tier.
MIN_INTERVAL_S = float(os.getenv("DATELINE_MIN_INTERVAL", "4.2"))

# "fake" = canned answers (dateline/fake_engine.py); "real" = the engine module below.
ENGINE = os.getenv("DATELINE_ENGINE", "fake")
# Python module that exposes the real answer(question, cutoff, guard, event) function.
ENGINE_MODULE = os.getenv("DATELINE_ENGINE_MODULE") or "pipeline.engine"

# The 1912 article table (parquet), used by search and by the quote-validity check.
CORPUS_PATH = os.getenv("DATELINE_CORPUS", str(ROOT / "data" / "articles.parquet"))

CACHE_DIR = ROOT / "cache" / "llm"
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
