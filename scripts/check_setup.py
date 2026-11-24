"""Check that the configured model answers.

Usage:  python scripts/check_setup.py
Makes one uncached test call with DATELINE_MODEL (e.g. openrouter/anthropic/claude-haiku-5.5).
For a gemini/ model it first checks GEMINI_API_KEY and lists the Flash models that key can use.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from dateline import config, llm  # noqa: E402


def main() -> None:
    print(f"Engine: DATELINE_ENGINE={config.ENGINE}, DATELINE_ENGINE_MODULE={config.ENGINE_MODULE}")
    if config.ANSWER_MODEL.startswith("gemini/"):
        list_gemini_models()
    print(f"\nTest call with DATELINE_MODEL={config.ANSWER_MODEL} ...")
    reply = llm.complete("In one short sentence: what ship sank in April 1912?", use_cache=False)
    print("Reply:", reply.strip())


def list_gemini_models() -> None:
    key = os.getenv("GEMINI_API_KEY", "").strip()
    if not key:
        sys.exit("GEMINI_API_KEY is not set. Add it to the .env file in the project folder.")
    r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                     headers={"x-goog-api-key": key}, params={"pageSize": 200}, timeout=30)
    if r.status_code != 200:
        sys.exit(f"Key check failed: HTTP {r.status_code}: {r.text[:300]}")
    models = [m["name"].removeprefix("models/") for m in r.json().get("models", [])
              if "generateContent" in m.get("supportedGenerationMethods", [])]
    flash = [m for m in models if "flash" in m]
    print("Key works. Flash models available:")
    for m in flash:
        print("  gemini/" + m)


if __name__ == "__main__":
    main()
