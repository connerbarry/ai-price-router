import sqlite3
conn = sqlite3.connect("mr_price.db")
rows = conn.execute("SELECT model_id, input_cost_mtok, output_cost_mtok FROM price_snapshots WHERE source='openrouter' AND model_id LIKE '%opus%' ORDER BY snapshot_ts DESC LIMIT 5").fetchall()
for r in rows: print(r)
conn.close()
