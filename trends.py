"""
Mr. Price Trends
----------------
Query Agent Mr. Price's historical database for pricing trends.

Usage:
    python trends.py                          # interactive menu
    python trends.py --model "claude-sonnet"  # trend for a specific model
    python trends.py --top 10                 # top 10 cheapest right now
    python trends.py --markup                 # biggest OpenRouter markups today
    python trends.py --movers                 # biggest price changes since yesterday
"""

import sqlite3
import argparse
from pathlib import Path
from datetime import datetime, timezone

from config import DB_FILE

# ── Helpers ───────────────────────────────────────────────────────────────────
def connect() -> sqlite3.Connection:
    if not DB_FILE.exists():
        print(f"Database not found: {DB_FILE}")
        print("Run agent_mr_price.py first to collect data.")
        exit(1)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def latest_snapshot(conn: sqlite3.Connection, source="openrouter") -> str | None:
    row = conn.execute(
        "SELECT MAX(snapshot_ts) as ts FROM price_snapshots WHERE source = ?",
        (source,)
    ).fetchone()
    return row["ts"] if row else None


def all_snapshots(conn: sqlite3.Connection, source="openrouter") -> list[str]:
    rows = conn.execute(
        "SELECT DISTINCT snapshot_ts FROM price_snapshots WHERE source = ? ORDER BY snapshot_ts",
        (source,)
    ).fetchall()
    return [r["snapshot_ts"] for r in rows]


def fmt_ts(ts: str) -> str:
    try:
        dt = datetime.fromisoformat(ts)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return ts


# ── Queries ───────────────────────────────────────────────────────────────────
def show_model_trend(conn: sqlite3.Connection, search: str):
    """Show price history for a model matching the search string."""
    rows = conn.execute("""
        SELECT snapshot_ts, source, model_id, input_cost_mtok, output_cost_mtok
        FROM price_snapshots
        WHERE LOWER(model_id) LIKE LOWER(?)
        ORDER BY model_id, snapshot_ts
    """, (f"%{search}%",)).fetchall()

    if not rows:
        print(f"\n  No models found matching '{search}'")
        return

    # Group by model_id
    models: dict[str, list] = {}
    for r in rows:
        models.setdefault(r["model_id"], []).append(r)

    for model_id, history in models.items():
        print(f"\n  {'─'*60}")
        print(f"  {model_id}")
        print(f"  {'─'*60}")
        print(f"  {'Date':<20} {'Source':<12} {'Input $/MTok':>14} {'Output $/MTok':>14}  {'Change':>8}")
        print(f"  {'─'*60}")
        prev_inp = None
        for r in history:
            inp = r["input_cost_mtok"]
            change = ""
            if prev_inp is not None and prev_inp > 0:
                pct = (inp - prev_inp) / prev_inp * 100
                change = f"{'▲' if pct > 0 else '▼'} {abs(pct):.1f}%"
            print(f"  {fmt_ts(r['snapshot_ts']):<20} {r['source']:<12} {inp:>14.4f} {r['output_cost_mtok']:>14.4f}  {change:>8}")
            prev_inp = inp


def show_top_cheapest(conn: sqlite3.Connection, n: int = 10, source="openrouter"):
    """Show the N cheapest paid models in the latest snapshot."""
    ts = latest_snapshot(conn, source)
    if not ts:
        print("No data yet.")
        return

    rows = conn.execute("""
        SELECT model_id, provider, input_cost_mtok, output_cost_mtok, context_length
        FROM price_snapshots
        WHERE snapshot_ts = ? AND source = ? AND is_free = 0 AND input_cost_mtok > 0
        ORDER BY input_cost_mtok ASC
        LIMIT ?
    """, (ts, source, n)).fetchall()

    print(f"\n  Top {n} cheapest ({source}) — snapshot {fmt_ts(ts)}")
    print(f"  {'─'*70}")
    print(f"  {'Model':<50} {'Input':>10} {'Output':>10} {'Ctx':>8}")
    print(f"  {'─'*70}")
    for r in rows:
        ctx = f"{r['context_length']//1000}k" if r['context_length'] else "?"
        print(f"  {r['model_id']:<50} {r['input_cost_mtok']:>10.4f} {r['output_cost_mtok']:>10.4f} {ctx:>8}")


def show_markups(conn: sqlite3.Connection, n: int = 15):
    """Show biggest OpenRouter markups over direct pricing."""
    ts = conn.execute("SELECT MAX(snapshot_ts) FROM markup_snapshots").fetchone()[0]
    if not ts:
        print("No markup data yet.")
        return

    rows = conn.execute("""
        SELECT model_id, provider,
               direct_input_mtok, openrouter_input_mtok, input_markup_pct,
               direct_output_mtok, openrouter_output_mtok, output_markup_pct
        FROM markup_snapshots
        WHERE snapshot_ts = ?
        ORDER BY input_markup_pct DESC
        LIMIT ?
    """, (ts, n)).fetchall()

    print(f"\n  OpenRouter markups over direct — {fmt_ts(ts)}")
    print(f"  {'─'*80}")
    print(f"  {'Model':<45} {'Direct':>10} {'OR Price':>10} {'Markup':>8}")
    print(f"  {'─'*80}")
    for r in rows:
        arrow = "▲" if r["input_markup_pct"] > 0 else "▼"
        print(f"  {r['model_id']:<45} ${r['direct_input_mtok']:>9.4f} ${r['openrouter_input_mtok']:>9.4f} "
              f"  {arrow}{abs(r['input_markup_pct']):>6.1f}%")


def show_movers(conn: sqlite3.Connection):
    """Show models with the biggest price changes between the last two snapshots."""
    snapshots = all_snapshots(conn)
    if len(snapshots) < 2:
        print("\n  Need at least 2 snapshots to show movers. Check back tomorrow!")
        return

    ts_new = snapshots[-1]
    ts_old = snapshots[-2]

    rows = conn.execute("""
        SELECT n.model_id, n.source,
               o.input_cost_mtok AS old_inp, n.input_cost_mtok AS new_inp,
               ROUND((n.input_cost_mtok - o.input_cost_mtok) / o.input_cost_mtok * 100, 2) AS pct_change
        FROM price_snapshots n
        JOIN price_snapshots o
          ON n.model_id = o.model_id AND n.source = o.source
        WHERE n.snapshot_ts = ? AND o.snapshot_ts = ?
          AND o.input_cost_mtok > 0 AND n.input_cost_mtok > 0
          AND pct_change != 0
        ORDER BY ABS(pct_change) DESC
        LIMIT 20
    """, (ts_new, ts_old)).fetchall()

    if not rows:
        print(f"\n  No price changes between {fmt_ts(ts_old)} and {fmt_ts(ts_new)}")
        return

    print(f"\n  Biggest movers: {fmt_ts(ts_old)} → {fmt_ts(ts_new)}")
    print(f"  {'─'*70}")
    print(f"  {'Model':<48} {'Old':>10} {'New':>10} {'Change':>8}")
    print(f"  {'─'*70}")
    for r in rows:
        arrow = "▲" if r["pct_change"] > 0 else "▼"
        print(f"  {r['model_id']:<48} ${r['old_inp']:>9.4f} ${r['new_inp']:>9.4f}  "
              f"{arrow}{abs(r['pct_change']):>6.1f}%")


def show_summary(conn: sqlite3.Connection):
    """High-level overview of what's in the database."""
    snapshots = all_snapshots(conn)
    total = conn.execute("SELECT COUNT(*) FROM price_snapshots").fetchone()[0]
    markup_count = conn.execute("SELECT COUNT(*) FROM markup_snapshots").fetchone()[0]

    print(f"\n  {'─'*50}")
    print(f"  Mr. Price Database Summary")
    print(f"  {'─'*50}")
    print(f"  Snapshots collected : {len(snapshots)}")
    if snapshots:
        print(f"  First snapshot      : {fmt_ts(snapshots[0])}")
        print(f"  Latest snapshot     : {fmt_ts(snapshots[-1])}")
    print(f"  Total price records : {total:,}")
    print(f"  Markup comparisons  : {markup_count:,}")
    print(f"  {'─'*50}")


def interactive_menu(conn: sqlite3.Connection):
    while True:
        print("\n  ┌─────────────────────────────────┐")
        print("  │       Mr. Price Trends          │")
        print("  ├─────────────────────────────────┤")
        print("  │  1. Search model price history  │")
        print("  │  2. Top cheapest models         │")
        print("  │  3. OpenRouter markups          │")
        print("  │  4. Price movers                │")
        print("  │  5. Database summary            │")
        print("  │  6. Exit                        │")
        print("  └─────────────────────────────────┘")
        choice = input("  Choice: ").strip()

        if choice == "1":
            q = input("  Search model name: ").strip()
            show_model_trend(conn, q)
        elif choice == "2":
            n = input("  How many? [10]: ").strip() or "10"
            show_top_cheapest(conn, int(n))
        elif choice == "3":
            show_markups(conn)
        elif choice == "4":
            show_movers(conn)
        elif choice == "5":
            show_summary(conn)
        elif choice == "6":
            print("\n  Goodbye.\n")
            break
        else:
            print("  Invalid choice.")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Mr. Price trend queries")
    parser.add_argument("--model",  type=str, help="Search model price history")
    parser.add_argument("--top",    type=int, help="Top N cheapest models")
    parser.add_argument("--markup", action="store_true", help="OpenRouter markups")
    parser.add_argument("--movers", action="store_true", help="Biggest price movers")
    parser.add_argument("--summary",action="store_true", help="Database summary")
    args = parser.parse_args()

    conn = connect()

    if args.model:
        show_model_trend(conn, args.model)
    elif args.top:
        show_top_cheapest(conn, args.top)
    elif args.markup:
        show_markups(conn)
    elif args.movers:
        show_movers(conn)
    elif args.summary:
        show_summary(conn)
    else:
        interactive_menu(conn)

    conn.close()


if __name__ == "__main__":
    main()