"""
AI Cost Router
--------------
Routes LLM requests to the optimal model based on:
  1. Availability  — only models accessible via your active API keys
  2. Quality floor — minimum score for the task type (no cheap-but-useless picks)
  3. Cost optimize — best quality-per-dollar among qualified candidates

Pulls live pricing and quality scores from mr_price.db (built by agent_mr_price.py).

Usage:
    from router import route
    model, cost, reason = route("Write a Python function to parse JSON", task_type="coding")

    python router.py --task "Explain quantum entanglement" --type reasoning
    python router.py --list   # show all available models with scores
"""

import sqlite3
import argparse
from pathlib import Path
from dataclasses import dataclass

from config import DB_FILE, ACTIVE_KEYS, TASK_FLOORS

# ── AA UUID → OpenRouter model ID mapping ─────────────────────────────────────
AA_TO_OR: dict[str, str] = {
    # ── Anthropic ──────────────────────────────────────────────────────────────
    # Claude Opus 4.6 variants → best available Opus on OR
    "53c98840-47af-49aa-94e6-469fb17e9a1b": "anthropic/claude-opus-4.6",   # Opus 4.6 Adaptive Reasoning Max
    "4386585e-71b4-4a0c-8a63-afb333419cd6": "anthropic/claude-opus-4.6",   # Opus 4.6 Non-reasoning High
    "2660d74f-ce79-48a8-8b53-6e668e2071a2": "anthropic/claude-opus-4.5",   # Opus 4.5 Reasoning
    "4077490a-bbfb-404e-979a-a97a20e3b5de": "anthropic/claude-opus-4.5",   # Opus 4.5 Non-reasoning
    # Claude Sonnet 4.6 variants
    "df8d14e0-3997-4e4d-b4ad-9c047acc9c69": "anthropic/claude-sonnet-4.6", # Sonnet 4.6 Adaptive Max
    "2e40e695-3cec-43da-83f9-615af30b8e91": "anthropic/claude-sonnet-4.6", # Sonnet 4.6 Non-reasoning High
    "f2e21112-192e-4aed-ae82-68ca3b38e667": "anthropic/claude-sonnet-4.6", # Sonnet 4.6 Non-reasoning Low
    "90e078f2-051b-4c63-8919-76618971cb3f": "anthropic/claude-sonnet-4.5", # Claude 4.5 Sonnet Reasoning
    "a7564055-f8ba-4c4b-9e2d-060f61263645": "anthropic/claude-sonnet-4",   # Claude 4 Sonnet Reasoning
    # Claude Haiku
    # (not in top-40 intel but useful for cheap tasks)

    # ── OpenAI ─────────────────────────────────────────────────────────────────
    "498862c3-f9ac-49d2-852f-16a02bb0c38f": "openai/gpt-5.2",              # GPT-5.2 xhigh
    "84e3f11e-d659-4941-8988-1dbfabbaf538": "openai/gpt-5.2",              # GPT-5.2 medium
    "7f3c9423-3ee3-4369-a6d9-3f2a40aff00e": "openai/gpt-5.2",              # GPT-5.2 low (maps to same OR model)
    "019e86f6-e66b-42d8-8a50-235a06b53003": "openai/gpt-5.2-codex",        # GPT-5.2 Codex xhigh
    "4dc12a38-b18f-4c43-8e1b-678f8434b5b1": "openai/gpt-5.1",              # GPT-5.1 high
    "04d023f3-025c-4d78-9571-53edda3eaf2a": "openai/gpt-5.1-codex",        # GPT-5.1 Codex high
    "48e50f00-1fd1-4acc-b337-61078aa341e6": "openai/gpt-5",                # GPT-5 high
    "5d11e7a1-4f70-4e5a-9364-e193761d6757": "openai/gpt-5-codex",          # GPT-5 Codex high
    "5e965af0-ca5c-4f47-9ba9-06000508b84a": "openai/gpt-5",                # GPT-5 medium
    "29855680-7469-43eb-8b88-cd3fb1d99da3": "openai/gpt-5-mini",           # GPT-5 mini high
    "c3274a19-6d3c-4d01-ab9b-5055a0a40429": "openai/gpt-5-mini",           # GPT-5 mini medium
    "fd4454ff-e703-46c0-a7f5-fa69af09486d": "openai/gpt-5.1-codex-mini",   # GPT-5.1 Codex mini high
    "ca04852c-eaae-4881-a208-f9b2ca3b7cd6": "openai/o3-pro",               # o3-pro
    "12adec16-19fe-4d92-aeff-5ef3eb7e780a": "openai/gpt-oss-120b",         # MiniMax-M2.5 → closest OSS

    # ── Google ─────────────────────────────────────────────────────────────────
    "bbd93ebe-80da-4594-bb19-61e69d0331df": "google/gemini-3.1-pro-preview",# Gemini 3.1 Pro Preview
    "d1122eff-ee85-4fdc-8a9f-23bee6590667": "google/gemini-3-pro-preview",  # Gemini 3 Pro high
    "b2f3191f-77d6-4155-8be6-330f0baa1ae5": "google/gemini-3-pro-preview",  # Gemini 3 Pro low
    "7c73c3be-7f51-4d14-bec8-d5789488df25": "google/gemini-3-flash-preview",# Gemini 3 Flash Reasoning

    # ── DeepSeek ───────────────────────────────────────────────────────────────
    "d621247c-d47e-458c-82cb-a166bc3b37e5": "deepseek/deepseek-v3.2",       # DeepSeek V3.2 Reasoning

    # ── xAI Grok ───────────────────────────────────────────────────────────────
    "5ea94a4a-55ac-4ea1-8898-2b3971e94af6": "x-ai/grok-4",                  # Grok 4

    # ── Qwen ───────────────────────────────────────────────────────────────────
    "0e66bae9-41f1-42fc-9276-ce8cb6f72919": "qwen/qwen3.5-397b-a17b",       # Qwen3.5 397B Reasoning
    "30ef2a79-e800-4165-9f13-2a338f120db7": "qwen/qwen3.5-397b-a17b",       # Qwen3.5 397B Non-reasoning
    "806032ff-6252-4c22-ba99-a126e411b7a4": "qwen/qwen3-max-thinking",       # Qwen3 Max Thinking

    # ── Kimi (Moonshot) — not yet on OR, skip ──────────────────────────────────
    # "a550ffca-...": None,   # Kimi K2.5
    # "bddebfd3-...": None,   # Kimi K2 Thinking

    # ── ZhipuAI GLM — not on OR ────────────────────────────────────────────────
    # "40663ad2-...": None,   # GLM-5
    # "6fc35842-...": None,   # GLM-4.7
}

# Which quality score to use for QPD calculation per task type
TASK_QPD_DIMENSION: dict[str, str] = {
    "coding":    "coding_index",
    "math":      "math_index",
    "reasoning": "intelligence_index",
    "knowledge": "intelligence_index",
    "simple":    "intelligence_index",
    "general":   "intelligence_index",
}


# ── Data classes ───────────────────────────────────────────────────────────────
@dataclass
class Candidate:
    aa_model_id:        str
    model_name:         str
    or_model_id:        str
    provider:           str
    intelligence_index: float | None
    coding_index:       float | None
    math_index:         float | None
    mmlu_pro:           float | None
    input_cost_mtok:    float
    output_cost_mtok:   float
    blended_cost_mtok:  float
    quality_score:      float | None   # dimension relevant to task
    quality_per_dollar: float | None   # quality_score / blended_cost


# ── Database queries ───────────────────────────────────────────────────────────
def load_candidates(conn: sqlite3.Connection) -> list[Candidate]:
    """
    Join latest quality scores with latest OR pricing.
    Only returns models that are in AA_TO_OR mapping.
    """
    # Get latest snapshot timestamps
    latest_aa = conn.execute(
        "SELECT MAX(snapshot_ts) FROM quality_snapshots"
    ).fetchone()[0]
    latest_or = conn.execute(
        "SELECT MAX(snapshot_ts) FROM price_snapshots WHERE source='openrouter'"
    ).fetchone()[0]

    if not latest_aa or not latest_or:
        raise RuntimeError("No data in database. Run agent_mr_price.py first.")

    # Pull quality scores
    quality_rows = conn.execute("""
        SELECT aa_model_id, model_name, intelligence_index, coding_index,
               math_index, mmlu_pro, aa_price_blended_mtok
        FROM quality_snapshots
        WHERE snapshot_ts = ?
    """, (latest_aa,)).fetchall()

    # Pull OR pricing
    price_rows = conn.execute("""
        SELECT model_id, input_cost_mtok, output_cost_mtok
        FROM price_snapshots
        WHERE source = 'openrouter' AND snapshot_ts = ? AND is_free = 0
    """, (latest_or,)).fetchall()
    or_prices = {r[0]: (r[1], r[2]) for r in price_rows}

    candidates = []
    for row in quality_rows:
        aa_id = row[0]
        or_id = AA_TO_OR.get(aa_id)
        if not or_id:
            continue   # not in our mapping

        pricing = or_prices.get(or_id)
        if not pricing:
            continue   # not available on OR right now

        inp, out = pricing
        blended = round((inp + out * 3) / 4, 4)   # 1:3 input:output ratio

        candidates.append(Candidate(
            aa_model_id        = aa_id,
            model_name         = row[1] or aa_id,
            or_model_id        = or_id,
            provider           = or_id.split("/")[0],
            intelligence_index = row[2],
            coding_index       = row[3],
            math_index         = row[4],
            mmlu_pro           = row[5],
            input_cost_mtok    = inp,
            output_cost_mtok   = out,
            blended_cost_mtok  = blended,
            quality_score      = None,
            quality_per_dollar = None,
        ))

    return candidates


def is_accessible(c: Candidate) -> bool:
    """Check if model's provider is unlocked by active keys."""
    if ACTIVE_KEYS.get("openrouter"):
        return True   # OpenRouter covers everything in our mapping
    return ACTIVE_KEYS.get(c.provider, False)


def passes_floor(c: Candidate, task_type: str) -> tuple[bool, str]:
    """Returns (passes, reason_if_failed)."""
    floors = TASK_FLOORS.get(task_type, TASK_FLOORS["general"])
    score_map = {
        "intelligence_index": c.intelligence_index,
        "coding_index":       c.coding_index,
        "math_index":         c.math_index,
        "mmlu_pro":           c.mmlu_pro,
    }
    for dimension, floor in floors.items():
        if floor is None:
            continue
        score = score_map.get(dimension)
        if score is None:
            return False, f"no {dimension} score available"
        if score < floor:
            return False, f"{dimension}={score:.1f} < floor {floor}"
    return True, ""


# ── Core routing logic ─────────────────────────────────────────────────────────
def route(
    prompt:    str,
    task_type: str = "general",
    top_n:     int = 3,
    verbose:   bool = False,
) -> tuple[str, float, str]:
    """
    Returns (or_model_id, blended_cost_mtok, reasoning_summary).

    task_type options: coding | math | reasoning | knowledge | simple | general
    """
    if task_type not in TASK_FLOORS:
        raise ValueError(f"Unknown task type '{task_type}'. Options: {list(TASK_FLOORS)}")

    conn = sqlite3.connect(DB_FILE)
    all_candidates = load_candidates(conn)
    conn.close()

    dimension = TASK_QPD_DIMENSION[task_type]

    # Stage 1 — availability filter
    accessible = [c for c in all_candidates if is_accessible(c)]
    if verbose:
        print(f"\n  Stage 1 — Accessibility: {len(accessible)}/{len(all_candidates)} models available")

    # Stage 2 — quality floor filter
    qualified = []
    disqualified = []
    for c in accessible:
        ok, reason = passes_floor(c, task_type)
        if ok:
            qualified.append(c)
        else:
            disqualified.append((c, reason))

    if verbose:
        print(f"  Stage 2 — Quality floor ({task_type}): {len(qualified)}/{len(accessible)} pass")
        if disqualified[:3]:
            for c, reason in disqualified[:3]:
                print(f"             ✗ {c.model_name[:40]:<40} ({reason})")

    if not qualified:
        return "", 0.0, "No models passed quality floors for this task type."

    # Stage 3 — compute quality-per-dollar and rank
    def get_score(c: Candidate) -> float:
        return getattr(c, dimension) or 0.0

    for c in qualified:
        score = get_score(c)
        c.quality_score = score
        c.quality_per_dollar = round(score / c.blended_cost_mtok, 4) if c.blended_cost_mtok > 0 else 0.0

    # Deduplicate OR model IDs — keep best quality_per_dollar per OR model
    seen_or: dict[str, Candidate] = {}
    for c in qualified:
        existing = seen_or.get(c.or_model_id)
        if not existing or (c.quality_per_dollar or 0) > (existing.quality_per_dollar or 0):
            seen_or[c.or_model_id] = c
    qualified = list(seen_or.values())

    ranked = sorted(qualified, key=lambda c: c.quality_per_dollar or 0, reverse=True)
    winner = ranked[0]

    reasoning = (
        f"Task: {task_type} | Dimension: {dimension} | "
        f"{len(qualified)} models qualified | "
        f"Winner: {winner.or_model_id} | "
        f"Score: {winner.quality_score:.1f} | "
        f"QPD: {winner.quality_per_dollar:.1f} | "
        f"Cost: ${winner.blended_cost_mtok:.3f}/MTok"
    )

    if verbose:
        print(f"\n  Stage 3 — Top {min(top_n, len(ranked))} by quality-per-dollar ({dimension}):")
        print(f"  {'Model':<45} {'Score':>6} {'$/MTok':>8} {'QPD':>8}")
        print(f"  {'─'*45} {'─'*6} {'─'*8} {'─'*8}")
        for c in ranked[:top_n]:
            marker = " ◄ WINNER" if c is winner else ""
            print(f"  {c.or_model_id:<45} {c.quality_score:>6.1f} "
                  f"  ${c.blended_cost_mtok:>6.3f} {c.quality_per_dollar:>8.1f}{marker}")
        print()

    return winner.or_model_id, winner.blended_cost_mtok, reasoning


# ── List all available models ──────────────────────────────────────────────────
def list_models():
    conn = sqlite3.connect(DB_FILE)
    candidates = load_candidates(conn)
    conn.close()

    accessible = [c for c in candidates if is_accessible(c)]

    # Deduplicate by OR model ID
    seen: dict[str, Candidate] = {}
    for c in accessible:
        if c.or_model_id not in seen:
            seen[c.or_model_id] = c
    unique = sorted(seen.values(), key=lambda c: c.intelligence_index or 0, reverse=True)

    print(f"\n  {'OR Model ID':<45} {'Intel':>6} {'Code':>6} {'Math':>6} {'$/MTok':>8}")
    print(f"  {'─'*45} {'─'*6} {'─'*6} {'─'*6} {'─'*8}")
    for c in unique:
        intel = f"{c.intelligence_index:.1f}" if c.intelligence_index else "  — "
        code  = f"{c.coding_index:.1f}"       if c.coding_index       else "  — "
        math  = f"{c.math_index:.1f}"         if c.math_index         else "  — "
        print(f"  {c.or_model_id:<45} {intel:>6} {code:>6} {math:>6}   ${c.blended_cost_mtok:>6.3f}")
    print(f"\n  {len(unique)} models available via active keys\n")


# ── CLI ────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Cost Router")
    parser.add_argument("--task",  type=str, help="Task prompt to route")
    parser.add_argument("--type",  type=str, default="general",
                        help=f"Task type: {list(TASK_FLOORS.keys())}")
    parser.add_argument("--top",   type=int, default=5, help="Show top N candidates")
    parser.add_argument("--list",  action="store_true", help="List all available models")
    args = parser.parse_args()

    if args.list:
        list_models()
    elif args.task:
        model, cost, reason = route(args.task, task_type=args.type,
                                    top_n=args.top, verbose=True)
        print(f"  → Route to: {model}")
        print(f"  → Est cost: ${cost:.3f}/MTok (blended)")
        print(f"  → {reason}\n")
    else:
        # Demo all task types
        demos = [
            ("Write a binary search tree in Python", "coding"),
            ("Solve: integrate x^2 * sin(x) dx",    "math"),
            ("Summarize this paragraph in 2 sentences", "simple"),
            ("Analyze the geopolitical implications of AI chip controls", "reasoning"),
            ("What are the symptoms of appendicitis?", "knowledge"),
        ]
        for prompt, ttype in demos:
            print(f"\n  Task ({ttype}): {prompt[:60]}")
            model, cost, reason = route(prompt, task_type=ttype, verbose=True)