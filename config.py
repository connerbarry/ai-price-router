"""
config.py
---------
Central configuration for mr-price.
All paths, model settings, and API endpoints live here.

After installing mr-price, edit this file once to match your environment.
Everything else reads from here — no other files need path changes.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# ── Base directory ─────────────────────────────────────────────────────────────
# Resolves to wherever this config.py file lives (the mr-price repo root).
BASE_DIR = Path(__file__).parent.resolve()

# ── Environment / API keys ─────────────────────────────────────────────────────
# Looks for .env in the repo root. Copy .env.example → .env and fill in keys.
ENV_FILE = BASE_DIR / ".env"
load_dotenv(ENV_FILE)

def get_key(name: str) -> str | None:
    return os.getenv(name)

# ── Database ───────────────────────────────────────────────────────────────────
DB_FILE  = BASE_DIR / "mr_price.db"
LOG_FILE = BASE_DIR / "mr_price.log"

# ── API endpoints ──────────────────────────────────────────────────────────────
OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"
LITELLM_PRICES_URL    = "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
AA_MODELS_URL         = "https://artificialanalysis.ai/api/v2/data/llms/models"

# ── Ollama (local classifier) ──────────────────────────────────────────────────
OLLAMA_URL   = os.getenv("OLLAMA_URL",   "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

# ── Routing ────────────────────────────────────────────────────────────────────
# Which providers are accessible via your active API keys.
# mr-price reads these from .env — set to "true" for each key you have.
ACTIVE_KEYS = {
    "openrouter": os.getenv("OPENROUTER_API_KEY") is not None,
    "anthropic":  os.getenv("ANTHROPIC_API_KEY")  is not None,
    "openai":     os.getenv("OPENAI_API_KEY")      is not None,
    "google":     os.getenv("GOOGLE_API_KEY")      is not None,
    "deepseek":   os.getenv("DEEPSEEK_API_KEY")    is not None,
    "x-ai":       os.getenv("XAI_API_KEY")         is not None,
    "mistralai":  os.getenv("MISTRAL_API_KEY")     is not None,
    "qwen":       os.getenv("QWEN_API_KEY")        is not None,
}

# Baseline model for savings calculations (what you'd pay without routing)
BASELINE_MODEL = os.getenv("BASELINE_MODEL", "anthropic/claude-opus-4.6")

# Average request size for cost estimates (tokens)
AVG_INPUT_TOKENS  = int(os.getenv("AVG_INPUT_TOKENS",  "500"))
AVG_OUTPUT_TOKENS = int(os.getenv("AVG_OUTPUT_TOKENS", "300"))

# ── Quality floors per task type ───────────────────────────────────────────────
# Minimum score a model must achieve to be considered for a task type.
# Raise floors for higher quality requirements, lower for more cost savings.
TASK_FLOORS = {
    "coding": {
        "coding_index":       float(os.getenv("FLOOR_CODING_INDEX",       "38.0")),
        "intelligence_index": float(os.getenv("FLOOR_CODING_INTEL",       "30.0")),
    },
    "math": {
        "math_index":         float(os.getenv("FLOOR_MATH_INDEX",         "85.0")),
        "intelligence_index": float(os.getenv("FLOOR_MATH_INTEL",         "30.0")),
    },
    "reasoning": {
        "intelligence_index": float(os.getenv("FLOOR_REASONING_INTEL",    "35.0")),
    },
    "knowledge": {
        "mmlu_pro":           float(os.getenv("FLOOR_KNOWLEDGE_MMLU",     "0.65")),
        "intelligence_index": float(os.getenv("FLOOR_KNOWLEDGE_INTEL",    "28.0")),
    },
    "simple": {
        "intelligence_index": float(os.getenv("FLOOR_SIMPLE_INTEL",       "15.0")),
    },
    "general": {
        "intelligence_index": float(os.getenv("FLOOR_GENERAL_INTEL",      "25.0")),
    },
}

# ── Schedule ───────────────────────────────────────────────────────────────────
# Cron/Task Scheduler time for daily price collection (24h format)
SCHEDULE_HOUR   = int(os.getenv("SCHEDULE_HOUR",   "6"))
SCHEDULE_MINUTE = int(os.getenv("SCHEDULE_MINUTE", "0"))


# ── Validation ─────────────────────────────────────────────────────────────────
def validate() -> list[str]:
    """
    Check config is valid. Returns list of warnings (not errors — mr-price
    degrades gracefully when keys are missing).
    """
    warnings = []
    if not get_key("OPENROUTER_API_KEY"):
        warnings.append("OPENROUTER_API_KEY not set — OpenRouter pricing unavailable")
    if not get_key("ARTIFICIAL_ANALYSIS_API_KEY"):
        warnings.append("ARTIFICIAL_ANALYSIS_API_KEY not set — quality scores unavailable")
    if not DB_FILE.exists():
        warnings.append(f"Database not found at {DB_FILE} — run agent_mr_price.py first")
    return warnings


if __name__ == "__main__":
    print(f"\n  mr-price config")
    print(f"  {'─'*40}")
    print(f"  Base dir : {BASE_DIR}")
    print(f"  DB file  : {DB_FILE}  {'✅' if DB_FILE.exists() else '❌ not found'}")
    print(f"  Env file : {ENV_FILE}  {'✅' if ENV_FILE.exists() else '❌ not found'}")
    print(f"\n  Active keys:")
    for provider, active in ACTIVE_KEYS.items():
        print(f"    {provider:<12} {'✅' if active else '❌'}")
    print(f"\n  Ollama:")
    print(f"    URL  : {OLLAMA_URL}")
    print(f"    Model: {OLLAMA_MODEL}")
    warnings = validate()
    if warnings:
        print(f"\n  ⚠️  Warnings:")
        for w in warnings:
            print(f"    {w}")
    else:
        print(f"\n  ✅ Config looks good")
    print()
