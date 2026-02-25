# ai-price-router

Automatically routes LLM requests to the optimal model based on live pricing and quality scores. Stop overpaying for simple tasks.

```
"Summarize this email"        →  gpt-oss-120b      $0.000053  (99.5% cheaper than Opus)
"Write a binary search tree"  →  gemini-3-flash    $0.000831  (91.7% cheaper than Opus)
"Solve this integral"         →  deepseek-v3.2     $0.000127  (98.7% cheaper than Opus)
```

## How it works

Three stages, fully automated:

**1. Classify** — A local Ollama model (Qwen 2.5 3B) reads the prompt and returns a task type: `coding`, `math`, `reasoning`, `knowledge`, `simple`, or `general`. Runs on your machine, zero API cost.

**2. Filter** — Quality floors remove models that aren't good enough for the task type. No cheap-but-useless picks. A model scoring below the coding floor won't handle your code, no matter how cheap it is.

**3. Route** — Among qualified models, rank by quality-per-dollar (QPD) using live pricing from OpenRouter and quality scores from Artificial Analysis. The winner is the best value for your specific task.

Pricing data is collected daily via a background scheduler and stored locally in SQLite.

---

## Architecture

```
ai-price-router/
├── agent_mr_price.py   # Daily data collector — OpenRouter + LiteLLM + Artificial Analysis
├── router.py           # Three-stage routing engine
├── classifier.py       # Local Ollama task classifier
├── trends.py           # CLI tool for querying historical price data
├── config.py           # All settings in one place
├── .env.example        # API key template
└── mr_price.db         # SQLite database (created on first run)
```

**Data sources:**
| Source | Models | What it provides |
|--------|--------|-----------------|
| OpenRouter API | 335+ | Intermediary pricing |
| LiteLLM GitHub | 2,000+ | Direct provider pricing |
| Artificial Analysis | 400+ | Intelligence, coding, math scores |

**Key insight:** OpenRouter charges +900–1,580% markup on frontier models (GPT, Claude, Gemini) but is 90–94% *cheaper* than direct pricing for open-source models (Llama, Qwen, Gemma). The router automatically picks the best source for each model.

---

## Requirements

- Python 3.11+
- [Ollama](https://ollama.ai) (for local task classification)
- OpenRouter API key (free tier available)
- Artificial Analysis API key (for quality scores)

---

## Setup

**1. Clone and install dependencies**
```bash
git clone https://github.com/connerbarry/ai-price-router
cd ai-price-router
pip install -r requirements.txt
```

**2. Configure API keys**
```bash
cp .env.example .env
```
Edit `.env` and add your keys:
```
OPENROUTER_API_KEY=your_key_here
ARTIFICIAL_ANALYSIS_API_KEY=your_key_here
```
Get keys at [openrouter.ai/settings/keys](https://openrouter.ai/settings/keys) and [artificialanalysis.ai](https://artificialanalysis.ai).

**3. Pull the local classifier model**
```bash
ollama pull qwen2.5:3b
```

**4. Run initial data collection**
```bash
python agent_mr_price.py
```
This populates `mr_price.db` with current pricing and quality scores (~2 minutes).

**5. Verify everything works**
```bash
python config.py       # check keys and DB
python router.py --list  # show available models with scores
```

---

## Usage

### As a Python module

```python
from router import route

model, blended_cost_mtok, reasoning = route("Write a binary search tree", task_type="coding")
print(model)    # google/gemini-3-flash-preview
print(reasoning)  # Task: coding | 14 models qualified | Winner: gemini-3-flash | QPD: 17.9
```

Auto-detect task type using the local classifier:

```python
from classifier import classify
from router import route

task = "Write a binary search tree"
task_type, method = classify(task)   # ("coding", "ollama")
model, cost, reason = route(task, task_type=task_type)
```

### CLI

```bash
# Route a single task
python router.py --task "Write a recursive fibonacci function" --type coding

# Auto-detect task type
python router.py --task "Summarize this document"

# List all available models with scores
python router.py --list

# Query historical pricing trends
python trends.py --model "claude-sonnet"
python trends.py --top 10
python trends.py --markup        # biggest OpenRouter markups today
python trends.py --movers        # biggest price changes since yesterday
```

---

## Scheduling daily price updates

**macOS / Linux (cron)**
```bash
# Run daily at 6:00 AM
0 6 * * * cd /path/to/ai-price-router && python agent_mr_price.py
```

**Windows (Task Scheduler)**
```powershell
$action  = New-ScheduledTaskAction -Execute "python" -Argument "C:\path\to\ai-price-router\agent_mr_price.py"
$trigger = New-ScheduledTaskTrigger -Daily -At "6:00AM"
Register-ScheduledTask -TaskName "ai-price-router" -Action $action -Trigger $trigger -RunLevel Highest
```

---

## Adjusting quality floors

Quality floors control the minimum acceptable model quality per task type. Edit `.env` to raise or lower them:

```env
# Coding: minimum coding_index score (0-100)
FLOOR_CODING_INDEX=38.0

# Math: minimum math benchmark score (0-100)
FLOOR_MATH_INDEX=85.0

# Reasoning: minimum intelligence index
FLOOR_REASONING_INTEL=35.0
```

Higher floors = better quality, fewer qualifying models, higher cost.
Lower floors = more models qualify, potentially lower cost.

---

## Adding direct provider keys

By default ai-price-router routes everything through OpenRouter. If you add direct provider keys to `.env`, it will automatically unlock those providers and factor in direct pricing:

```env
ANTHROPIC_API_KEY=your_key
OPENAI_API_KEY=your_key
GOOGLE_API_KEY=your_key
```

---

## OpenClaw plugin

An OpenClaw plugin that wraps ai-price-router is available at [openclaw-cost-optimizer](https://github.com/connerbarry/openclaw-cost-optimizer). It exposes routing as a tool your OpenClaw agent can invoke automatically on every LLM request.

---

## Data collected

ai-price-router stores all data locally in `mr_price.db` (SQLite). Nothing is sent to external services beyond the API calls needed to fetch pricing and quality data.

Tables:
- `price_snapshots` — daily pricing from OpenRouter and LiteLLM
- `quality_snapshots` — daily quality scores from Artificial Analysis
- `markup_snapshots` — calculated OpenRouter markups vs direct pricing

---

## License

MIT
