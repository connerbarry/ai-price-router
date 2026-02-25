"""
Task Classifier
---------------
Uses a local Ollama model to classify incoming tasks into routing categories.
Replaces keyword-based detect_task_type() in route_request.py.

Single responsibility: takes a task string, returns one category word.
Runs entirely locally — zero API cost, no network latency, no data leaving machine.

Falls back to keyword matching if Ollama is unavailable.
"""

import urllib.request
import urllib.error
import json
import re

from config import OLLAMA_URL, OLLAMA_MODEL

# ── Config ────────────────────────────────────────────────────────────────────
OLLAMA_GENERATE_URL = f"{OLLAMA_URL}/api/generate"
OLLAMA_TAGS_URL     = f"{OLLAMA_URL}/api/tags"

VALID_CATEGORIES = {"coding", "math", "reasoning", "knowledge", "simple", "general"}

CLASSIFICATION_PROMPT = """You are a task classifier for an AI routing system. Your only job is to classify tasks into exactly one category.

Categories:
- coding: programming, debugging, scripts, algorithms, SQL, APIs, code review, implementation. Also includes "write a [data structure/algorithm/program]", "build a [system/component]", "create a [script/function]", "implement [anything technical]"
- math: calculations, equations, proofs, statistics, formulas, integrals, derivatives
- reasoning: analysis, evaluation, strategy, pros/cons, planning, architecture decisions, critical thinking
- knowledge: facts, history, explanations, definitions, "who is", "what is", "what are", "tell me about", "what caused"
- simple: summarization, translation, formatting, rephrasing, short Q&A, classification, labeling
- general: anything that doesn't clearly fit the above categories

Rules:
- Reply with ONLY the category name
- No explanation, no punctuation, no extra words
- Must be one of: coding, math, reasoning, knowledge, simple, general

Task: {task}
Category:"""

# ── Keyword fallback (from route_request.py) ──────────────────────────────────
KEYWORD_SIGNALS = {
    "coding": [
        "code", "function", "script", "program", "implement", "debug",
        "refactor", "class", "algorithm", "api", "endpoint", "parse",
        "regex", "sql", "query", "html", "css", "javascript", "python",
        "typescript", "bash", "shell"
    ],
    "math": [
        "math", "calculate", "equation", "integral", "derivative", "solve",
        "proof", "algebra", "calculus", "statistics", "probability",
        "matrix", "linear algebra", "arithmetic", "formula"
    ],
    "reasoning": [
        "reason", "analyze", "evaluate", "compare", "assess", "critique",
        "argument", "logic", "infer", "deduce", "implication", "strategy",
        "plan", "architect", "design", "think through", "pros and cons"
    ],
    "knowledge": [
        "explain", "what is", "how does", "describe", "define", "history",
        "who was", "when did", "facts about", "overview", "summary of",
        "what caused", "why did", "what happened", "who is", "what was",
        "tell me about", "background on", "origins of", "what are the"
    ],
    "simple": [
        "summarize", "translate", "rephrase", "list", "format", "convert",
        "classify", "categorize", "label", "tag", "short", "quick", "brief"
    ],
}


def _keyword_classify(task: str) -> str:
    """Keyword-based fallback classifier."""
    task_lower = task.lower()
    scores = {t: 0 for t in KEYWORD_SIGNALS}
    for task_type, signals in KEYWORD_SIGNALS.items():
        for signal in signals:
            if signal in task_lower:
                scores[task_type] += 1
    best = max(scores, key=lambda t: scores[t])
    return best if scores[best] > 0 else "general"


def _ollama_available() -> bool:
    """Quick check if Ollama server is responding."""
    try:
        req = urllib.request.Request(OLLAMA_TAGS_URL, method="GET")
        with urllib.request.urlopen(req, timeout=2):
            return True
    except Exception:
        return False


def _call_ollama(task: str) -> str | None:
    """
    Call local Ollama model to classify the task.
    Returns category string or None if call fails.
    """
    prompt = CLASSIFICATION_PROMPT.format(task=task[:500])
    payload = json.dumps({
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.0,   # deterministic
            "num_predict": 10,    # only need one word
        }
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_GENERATE_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode("utf-8"))
            raw = result.get("response", "").strip().lower()
            # Extract first word, strip punctuation
            word = re.split(r"[\s\.,;:!?]", raw)[0].strip()
            if word in VALID_CATEGORIES:
                return word
            # Try to find a valid category anywhere in the response
            for category in VALID_CATEGORIES:
                if category in raw:
                    return category
            return None
    except Exception:
        return None


def classify(task: str, verbose: bool = False) -> tuple[str, str]:
    """
    Classify a task into a routing category.

    Returns (category, method) where method is 'ollama' or 'keyword'.

    Usage:
        from classifier import classify
        task_type, method = classify("Write a fibonacci function")
        # → ("coding", "ollama")
    """
    if _ollama_available():
        result = _call_ollama(task)
        if result:
            if verbose:
                print(f"  [classifier] ollama → {result}")
            return result, "ollama"
        if verbose:
            print("  [classifier] ollama returned invalid response, falling back to keywords")
    else:
        if verbose:
            print("  [classifier] ollama unavailable, using keyword fallback")

    result = _keyword_classify(task)
    if verbose:
        print(f"  [classifier] keyword → {result}")
    return result, "keyword"


# ── CLI for testing ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    test_tasks = [
        "Write a recursive fibonacci function",
        "Solve this integral: x squared times sin x",
        "Summarize this email in two sentences",
        "What caused the 2008 financial crisis",
        "Evaluate the pros and cons of microservices architecture",
        "Help me debug this React component",
        "Who is Ada Lovelace",
        "What are the symptoms of diabetes",
        "Tell me about the history of the Roman Empire",
        "Translate this paragraph to French",
    ]

    # Allow passing a custom task via CLI
    tasks = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else test_tasks

    ollama_up = _ollama_available()
    print(f"\n  Ollama: {'✅ available' if ollama_up else '❌ unavailable (keyword fallback)'}")
    print(f"  Model:  {OLLAMA_MODEL}\n")
    print(f"  {'Task':<55} {'Category':<12} {'Method'}")
    print(f"  {'─'*55} {'─'*12} {'─'*8}")

    for task in tasks:
        category, method = classify(task)
        print(f"  {task:<55} {category:<12} {method}")
    print()