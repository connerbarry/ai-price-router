"""
Agent Mr. Price v3
------------------
Fetches model pricing AND quality scores from three sources:

  1. OpenRouter API         — aggregated/intermediary pricing (337+ models)
  2. LiteLLM GitHub         — direct provider pricing (Anthropic, OpenAI, Google, etc.)
  3. Artificial Analysis    — quality/intelligence scores per model per category

Also calculates:
  - OpenRouter markup over direct pricing
  - Quality-per-dollar score (intelligence index / blended price)

Schedule this script to run daily via Windows Task Scheduler.
"""

import os
import sys
import sqlite3
import logging
import requests
from datetime import datetime, timezone
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
from config import (
    DB_FILE, LOG_FILE,
    OPENROUTER_MODELS_URL, LITELLM_PRICES_URL, AA_MODELS_URL,
    get_key,
)

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("agent_mr_price")

# ── Database ──────────────────────────────────────────────────────────────────
def init_db(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS price_snapshots (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts      TEXT    NOT NULL,
            source           TEXT    NOT NULL,
            model_id         TEXT    NOT NULL,
            model_name       TEXT,
            provider         TEXT,
            context_length   INTEGER,
            input_cost       REAL,
            output_cost      REAL,
            input_cost_mtok  REAL,
            output_cost_mtok REAL,
            is_free          INTEGER DEFAULT 0
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS quality_snapshots (
            id                          INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts                 TEXT    NOT NULL,
            aa_model_id                 TEXT    NOT NULL,
            model_name                  TEXT,
            provider                    TEXT,
            -- Overall intelligence
            intelligence_index          REAL,
            -- Category scores
            coding_index                REAL,
            math_index                  REAL,
            -- Individual benchmarks
            mmlu_pro                    REAL,   -- general knowledge
            gpqa                        REAL,   -- science reasoning
            hle                         REAL,   -- hard logic evals
            livecodebench               REAL,   -- coding
            scicode                     REAL,   -- scientific coding
            math_500                    REAL,   -- math
            aime                        REAL,   -- advanced math
            -- Performance
            output_tokens_per_sec       REAL,
            time_to_first_token_sec     REAL,
            -- AA pricing (for quality-per-dollar)
            aa_price_blended_mtok       REAL,
            aa_price_input_mtok         REAL,
            aa_price_output_mtok        REAL,
            -- Derived
            quality_per_dollar          REAL    -- intelligence_index / aa_price_blended_mtok
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS markup_snapshots (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_ts             TEXT    NOT NULL,
            model_id                TEXT    NOT NULL,
            provider                TEXT,
            direct_input_mtok       REAL,
            openrouter_input_mtok   REAL,
            input_markup_pct        REAL,
            direct_output_mtok      REAL,
            openrouter_output_mtok  REAL,
            output_markup_pct       REAL
        )
    """)
    for idx in [
        "CREATE INDEX IF NOT EXISTS idx_ps_ts     ON price_snapshots(snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_ps_model  ON price_snapshots(model_id)",
        "CREATE INDEX IF NOT EXISTS idx_ps_source ON price_snapshots(source)",
        "CREATE INDEX IF NOT EXISTS idx_qs_ts     ON quality_snapshots(snapshot_ts)",
        "CREATE INDEX IF NOT EXISTS idx_qs_model  ON quality_snapshots(aa_model_id)",
        "CREATE INDEX IF NOT EXISTS idx_mu_ts     ON markup_snapshots(snapshot_ts)",
    ]:
        conn.execute(idx)
    conn.commit()
    log.info("Database ready: %s", DB_FILE)


# ── Fetch OpenRouter ──────────────────────────────────────────────────────────
def fetch_openrouter(api_key: str) -> list[dict]:
    log.info("Fetching from OpenRouter...")
    resp = requests.get(OPENROUTER_MODELS_URL,
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    log.info("OpenRouter: %d models received", len(models))
    return models


def parse_openrouter(models: list[dict], snapshot_ts: str) -> list[dict]:
    rows = []
    for m in models:
        pricing = m.get("pricing", {})
        inp = float(pricing.get("prompt",     0) or 0)
        out = float(pricing.get("completion", 0) or 0)
        if inp < 0 or out < 0:
            continue
        model_id = m.get("id", "")
        provider = model_id.split("/")[0] if "/" in model_id else ""
        rows.append({
            "snapshot_ts":      snapshot_ts,
            "source":           "openrouter",
            "model_id":         model_id,
            "model_name":       m.get("name", ""),
            "provider":         provider,
            "context_length":   m.get("context_length"),
            "input_cost":       inp,
            "output_cost":      out,
            "input_cost_mtok":  round(inp * 1_000_000, 6),
            "output_cost_mtok": round(out * 1_000_000, 6),
            "is_free":          1 if (inp == 0 and out == 0) else 0,
        })
    return rows


# ── Fetch LiteLLM ─────────────────────────────────────────────────────────────
def fetch_litellm() -> dict:
    log.info("Fetching from LiteLLM GitHub...")
    resp = requests.get(LITELLM_PRICES_URL, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    log.info("LiteLLM: %d entries received", len(data))
    return data


def parse_litellm(data: dict, snapshot_ts: str) -> list[dict]:
    rows = []
    for model_id, info in data.items():
        if not isinstance(info, dict):
            continue
        inp = float(info.get("input_cost_per_token",  0) or 0)
        out = float(info.get("output_cost_per_token", 0) or 0)
        if inp < 0 or out < 0:
            continue
        mode = info.get("mode", "")
        if mode not in ("chat", "completion", ""):
            continue
        rows.append({
            "snapshot_ts":      snapshot_ts,
            "source":           "litellm",
            "model_id":         model_id,
            "model_name":       model_id,
            "provider":         info.get("litellm_provider", ""),
            "context_length":   info.get("max_tokens") or info.get("max_input_tokens"),
            "input_cost":       inp,
            "output_cost":      out,
            "input_cost_mtok":  round(inp * 1_000_000, 6),
            "output_cost_mtok": round(out * 1_000_000, 6),
            "is_free":          1 if (inp == 0 and out == 0) else 0,
        })
    return rows


# ── Fetch Artificial Analysis ─────────────────────────────────────────────────
def fetch_aa(api_key: str) -> list[dict]:
    log.info("Fetching from Artificial Analysis...")
    resp = requests.get(AA_MODELS_URL,
                        headers={"x-api-key": api_key},
                        timeout=30)
    resp.raise_for_status()
    models = resp.json().get("data", [])
    log.info("Artificial Analysis: %d models received", len(models))
    return models


def parse_aa(models: list[dict], snapshot_ts: str) -> list[dict]:
    rows = []
    for m in models:
        evals   = m.get("evaluations", {}) or {}
        pricing = m.get("pricing",     {}) or {}

        intel   = evals.get("artificial_analysis_intelligence_index")
        price_b = pricing.get("price_1m_blended_3_to_1")

        # quality per dollar — only when both exist and price > 0
        qpd = None
        if intel is not None and price_b and price_b > 0:
            qpd = round(intel / price_b, 4)

        rows.append({
            "snapshot_ts":              snapshot_ts,
            "aa_model_id":              m.get("id", ""),
            "model_name":               m.get("name", ""),
            "provider":                 (m.get("model_creator") or {}).get("name", ""),
            "intelligence_index":       intel,
            "coding_index":             evals.get("artificial_analysis_coding_index"),
            "math_index":               evals.get("artificial_analysis_math_index"),
            "mmlu_pro":                 evals.get("mmlu_pro"),
            "gpqa":                     evals.get("gpqa"),
            "hle":                      evals.get("hle"),
            "livecodebench":            evals.get("livecodebench"),
            "scicode":                  evals.get("scicode"),
            "math_500":                 evals.get("math_500"),
            "aime":                     evals.get("aime"),
            "output_tokens_per_sec":    m.get("median_output_tokens_per_second"),
            "time_to_first_token_sec":  m.get("median_time_to_first_token_seconds"),
            "aa_price_blended_mtok":    price_b,
            "aa_price_input_mtok":      pricing.get("price_1m_input_tokens"),
            "aa_price_output_mtok":     pricing.get("price_1m_output_tokens"),
            "quality_per_dollar":       qpd,
        })
    return rows


# ── Markup Calculator ─────────────────────────────────────────────────────────
def calc_markups(or_rows: list[dict], ll_rows: list[dict], snapshot_ts: str) -> list[dict]:
    ll_lookup: dict[str, dict] = {}
    for r in ll_rows:
        key = r["model_id"].lower()
        if key not in ll_lookup or r["input_cost_mtok"] < ll_lookup[key]["input_cost_mtok"]:
            ll_lookup[key] = r

    markups = []
    for or_row in or_rows:
        if or_row["is_free"]:
            continue
        base = or_row["model_id"].split("/", 1)[-1].lower()
        ll_match = ll_lookup.get(base) or next(
            (v for k, v in ll_lookup.items() if base in k or k in base), None
        )
        if not ll_match:
            continue
        d_inp, o_inp = ll_match["input_cost_mtok"],  or_row["input_cost_mtok"]
        d_out, o_out = ll_match["output_cost_mtok"], or_row["output_cost_mtok"]
        if d_inp <= 0 or d_out <= 0:
            continue
        markups.append({
            "snapshot_ts":            snapshot_ts,
            "model_id":               or_row["model_id"],
            "provider":               or_row["provider"],
            "direct_input_mtok":      d_inp,
            "openrouter_input_mtok":  o_inp,
            "input_markup_pct":       round((o_inp - d_inp) / d_inp * 100, 2),
            "direct_output_mtok":     d_out,
            "openrouter_output_mtok": o_out,
            "output_markup_pct":      round((o_out - d_out) / d_out * 100, 2),
        })
    return markups


# ── Store ─────────────────────────────────────────────────────────────────────
def store_prices(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany("""
        INSERT INTO price_snapshots
            (snapshot_ts, source, model_id, model_name, provider, context_length,
             input_cost, output_cost, input_cost_mtok, output_cost_mtok, is_free)
        VALUES
            (:snapshot_ts, :source, :model_id, :model_name, :provider, :context_length,
             :input_cost, :output_cost, :input_cost_mtok, :output_cost_mtok, :is_free)
    """, rows)
    conn.commit()


def store_quality(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany("""
        INSERT INTO quality_snapshots
            (snapshot_ts, aa_model_id, model_name, provider,
             intelligence_index, coding_index, math_index,
             mmlu_pro, gpqa, hle, livecodebench, scicode, math_500, aime,
             output_tokens_per_sec, time_to_first_token_sec,
             aa_price_blended_mtok, aa_price_input_mtok, aa_price_output_mtok,
             quality_per_dollar)
        VALUES
            (:snapshot_ts, :aa_model_id, :model_name, :provider,
             :intelligence_index, :coding_index, :math_index,
             :mmlu_pro, :gpqa, :hle, :livecodebench, :scicode, :math_500, :aime,
             :output_tokens_per_sec, :time_to_first_token_sec,
             :aa_price_blended_mtok, :aa_price_input_mtok, :aa_price_output_mtok,
             :quality_per_dollar)
    """, rows)
    conn.commit()


def store_markups(conn: sqlite3.Connection, rows: list[dict]):
    conn.executemany("""
        INSERT INTO markup_snapshots
            (snapshot_ts, model_id, provider,
             direct_input_mtok, openrouter_input_mtok, input_markup_pct,
             direct_output_mtok, openrouter_output_mtok, output_markup_pct)
        VALUES
            (:snapshot_ts, :model_id, :provider,
             :direct_input_mtok, :openrouter_input_mtok, :input_markup_pct,
             :direct_output_mtok, :openrouter_output_mtok, :output_markup_pct)
    """, rows)
    conn.commit()


# ── Report ────────────────────────────────────────────────────────────────────
def print_summary(or_rows, ll_rows, aa_rows, markups):
    paid_or  = [r for r in or_rows if not r["is_free"]]
    paid_ll  = [r for r in ll_rows if not r["is_free"]]
    scored   = [r for r in aa_rows if r["intelligence_index"] is not None]
    top_intel = sorted(scored, key=lambda r: r["intelligence_index"], reverse=True)[:5]
    top_qpd   = sorted(
        [r for r in scored if r["quality_per_dollar"] is not None],
        key=lambda r: r["quality_per_dollar"], reverse=True
    )[:5]
    top_code  = sorted(
        [r for r in scored if r.get("coding_index") is not None],
        key=lambda r: r["coding_index"], reverse=True
    )[:5]
    top_math  = sorted(
        [r for r in scored if r.get("math_index") is not None],
        key=lambda r: r["math_index"], reverse=True
    )[:5]
    top_markup = sorted(markups, key=lambda r: r["input_markup_pct"], reverse=True)[:3]

    print("\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  Agent Mr. Price v3  |  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
    print(f"  OpenRouter models   : {len(or_rows):>4}  ({len(paid_or)} paid)")
    print(f"  LiteLLM models      : {len(ll_rows):>4}  ({len(paid_ll)} paid)")
    print(f"  AA quality scores   : {len(aa_rows):>4}  ({len(scored)} with intel score)")
    print(f"  Markup comparisons  : {len(markups):>4}")

    if top_intel:
        print("\n  Top 5 by Intelligence Index:")
        for r in top_intel:
            price = f"${r['aa_price_blended_mtok']:.2f}/MTok" if r['aa_price_blended_mtok'] else "no price"
            print(f"    {r['model_name']:<45} {r['intelligence_index']:>6.1f}  ({price})")

    if top_qpd:
        print("\n  Top 5 Quality-per-Dollar (intel / blended price):")
        for r in top_qpd:
            print(f"    {r['model_name']:<45} {r['quality_per_dollar']:>8.2f}  "
                  f"(intel={r['intelligence_index']:.1f}, ${r['aa_price_blended_mtok']:.2f}/MTok)")

    if top_code:
        print("\n  Top 5 Coding Index:")
        for r in top_code:
            print(f"    {r['model_name']:<45} {r['coding_index']:>6.1f}")

    if top_math:
        print("\n  Top 5 Math Index:")
        for r in top_math:
            print(f"    {r['model_name']:<45} {r['math_index']:>6.1f}")

    if top_markup:
        print("\n  Highest OpenRouter markups:")
        for r in top_markup:
            print(f"    {r['model_id']:<45} +{r['input_markup_pct']:.0f}%")

    print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    log.info("=== Agent Mr. Price v3 starting ===")

    or_key = get_key("OPENROUTER_API_KEY")
    aa_key = get_key("ARTIFICIAL_ANALYSIS_API_KEY")

    if not or_key:
        log.error("OPENROUTER_API_KEY not set — add it to .env")
        sys.exit(1)
    if not aa_key:
        log.warning("ARTIFICIAL_ANALYSIS_API_KEY not set — quality scores will be skipped")

    snapshot_ts = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(DB_FILE)
    init_db(conn)

    # OpenRouter pricing
    or_raw  = fetch_openrouter(or_key)
    or_rows = parse_openrouter(or_raw, snapshot_ts)
    store_prices(conn, or_rows)
    log.info("Stored %d OpenRouter price records", len(or_rows))

    # LiteLLM direct pricing
    ll_raw  = fetch_litellm()
    ll_rows = parse_litellm(ll_raw, snapshot_ts)
    store_prices(conn, ll_rows)
    log.info("Stored %d LiteLLM price records", len(ll_rows))

    # Markup comparisons
    markups = calc_markups(or_rows, ll_rows, snapshot_ts)
    store_markups(conn, markups)
    log.info("Stored %d markup comparisons", len(markups))

    # Artificial Analysis quality scores
    aa_rows = []
    if aa_key:
        try:
            aa_raw  = fetch_aa(aa_key)
            aa_rows = parse_aa(aa_raw, snapshot_ts)
            store_quality(conn, aa_rows)
            log.info("Stored %d AA quality records", len(aa_rows))
        except Exception as e:
            log.error("Artificial Analysis fetch failed: %s", e)

    conn.close()
    print_summary(or_rows, ll_rows, aa_rows, markups)

    # ── Daily backup ───────────────────────────────────────────────────────────
    try:
        import shutil
        backup_dir = DB_FILE.parent / "backups"
        backup_dir.mkdir(exist_ok=True)
        date_str    = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        backup_file = backup_dir / f"mr_price_{date_str}.db"
        shutil.copy2(DB_FILE, backup_file)
        # Keep last 30 days only
        backups = sorted(backup_dir.glob("mr_price_*.db"))
        for old in backups[:-30]:
            old.unlink()
        log.info("DB backed up to %s", backup_file.name)
    except Exception as e:
        log.warning("DB backup failed: %s", e)

    log.info("=== Agent Mr. Price v3 done ===")


if __name__ == "__main__":
    main()