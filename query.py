import sqlite3
conn = sqlite3.connect("mr_price.db")
sql = "SELECT model_id, input_cost_mtok, output_cost_mtok FROM price_snapshots WHERE source='openrouter' AND is_free=0 AND provider IN ('anthropic','openai','google','deepseek','x-ai','qwen','mistralai','meta-llama') ORDER BY provider, model_id"
rows = conn.execute(sql).fetchall()
for r in rows:
    print(r[0], r[1], r[2])
conn.close()
