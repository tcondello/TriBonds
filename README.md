# TriBonds LLM Benchmark

Benchmarks LLMs on lateral word-connection puzzles. Given three words, models must identify the single connecting word that links all three (e.g. `[cake, swiss, roll]` → `cheese`).

Results across closed frontier providers (**Anthropic**, **OpenAI**, **Google**) and open-weight models served on **Fireworks AI** are asynchronously evaluated and streamed per-puzzle to CSV.

---

## 🏆 Benchmark Leaderboard (300 Puzzles)

Full detailed analysis and individual puzzle showdowns are available in [**`BENCHMARK_RESULTS.md`**](./BENCHMARK_RESULTS.md).

| Rank | Model | Provider | Accuracy (300 Puzzles) | Median Latency | Pricing (In / Out per 1M) |
| :---: | :--- | :--- | :---: | :---: | :---: |
| 🥇 | **`claude-opus-5-5`** | Anthropic | **85.7%** (257/300) | 1.60s | \$4.00 / \$20.00 |
| 🥈 | **`gpt-6-astra`** | OpenAI | **85.0%** (255/300) | 1.98s | \$10.00 / \$50.00 |
| 🥉 | **`gemini-3.8-flash`** | Google | **84.7%** (254/300) | 2.96s | **\$0.75 / \$3.75** |
| 4 | **`gemini-3.5-flash`** | Google | **84.3%** (253/300) | 1.93s | \$1.50 / \$9.00 |
| 4 | **`gpt-6.1-sol`** | OpenAI | **84.3%** (253/300) | 2.27s | \$2.00 / \$10.00 |
| 6 | **`gpt-6-luna`** | OpenAI | **83.7%** (251/300) | 1.59s | **\$0.10 / \$0.50** |
| 7 | **`ember-1`** | Fireworks | **83.3%** (250/300) | 2.36s | **\$0.30 / \$1.20** |
| 8 | **`glm-5p3-fast`** | Fireworks | **81.7%** (245/300) | **0.61s** | \$2.10 / \$6.60 |
| 8 | **`kimi-k3`** | Fireworks | **81.7%** (245/300) | 1.78s | \$3.00 / \$15.00 |
| 10 | **`glm-5p3-flash`** | Fireworks | **81.3%** (244/300) | **0.72s** | \$0.15 / \$0.50 |
| 11 | **`claude-haiku-5-5`** | Anthropic | **81.0%** (243/300) | 1.34s | \$0.10 / \$0.50 |
| 12 | **`glm-5p3`** | Fireworks | **80.7%** (242/300) | 0.92s | \$1.40 / \$4.40 |
| 13 | **`kimi-k3-fast`** | Fireworks | **80.0%** (240/300) | 1.69s | \$4.50 / \$22.50 |
| 14 | **`claude-sonnet-5-5`** | Anthropic | **79.3%** (238/300) | 1.40s | \$2.00 / \$10.00 |
| 15 | **`qwen3p8-2p4t-a95b`** | Fireworks | **78.3%** (235/300) | 1.87s | \$0.60 / \$2.00 |
| 16 | **`qwen3p8-max`** | Fireworks | **78.0%** (234/300) | 1.81s | \$1.40 / \$4.40 |
| 17 | **`deepseek-v4p1-flash`** | Fireworks | **77.0%** (231/300) | 0.74s | \$0.30 / \$1.20 |
| 17 | **`gpt-oss-120b`** | Fireworks | **77.0%** (231/300) | 0.78s | \$0.15 / \$0.60 |
| 19 | **`nemotron-3-ultra`** | Fireworks | **74.3%** (223/300) | 1.65s | \$1.40 / \$4.40 |
| 20 | **`minimax-m3`** | Fireworks | **70.3%** (211/300) | 0.78s | \$0.30 / \$1.20 |
| 21 | **`nemotron-lightning`** | Fireworks | **51.3%** (154/300) | 0.35s | \$0.15 / \$0.60 |

---

## How It Works

For each row in the challenge CSV, the pipeline:
1. Sends the three clue words to every enabled model concurrently (async)
2. Extracts a single-word answer from each response (automatically stripping `<think>` tags and reasoning blocks)
3. Grades against primary and secondary valid answers using lemmatization (e.g. `"Plane"` matches `"Planes"`)
4. Tracks latency (`_latency_s`), token counts, and USD cost, persisting row checkpoints immediately—safe to interrupt and resume at any time.

---

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/tcondello/TriBonds.git
cd TriBonds
uv sync
cp .env.example .env
# Add your API keys to .env
```

---

## Configuration

### 1. API Keys — `.env`

```env
# Fireworks AI (Open Source Models)
FIREWORKS_API_KEY=...
FIREWORKS_BASE_URL=https://api.fireworks.ai/inference/v1

# Frontier Closed Models
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

### 2. Models & Settings — `config.json`

Configure providers, models, reasoning effort levels, and execution settings:

```json
{
  "models": [
    { "provider": "fireworks", "model": "accounts/fireworks/models/ember-1", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/routers/glm-5p3-fast", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/glm-5p3", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/glm-5p3-flash", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/routers/kimi-k3-fast", "reasoning_effort": "none", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/kimi-k3", "reasoning_effort": "none", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/deepseek-v4p1-flash", "reasoning_effort": "none", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/gpt-oss-120b", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/qwen3p8-max", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/qwen3p8-2p4t-a95b", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/nemotron-3-ultra-nvfp4", "reasoning_effort": "low", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b", "reasoning_effort": "none", "enabled": true },
    { "provider": "fireworks", "model": "accounts/fireworks/models/minimax-m3", "reasoning_effort": "none", "enabled": true },
    { "provider": "openai",    "model": "gpt-6.1-sol",             "reasoning_effort": "low", "enabled": true },
    { "provider": "openai",    "model": "gpt-6-astra",             "reasoning_effort": "low", "enabled": true },
    { "provider": "openai",    "model": "gpt-6-luna",              "enabled": true },
    { "provider": "anthropic", "model": "claude-opus-5-5",         "enabled": true },
    { "provider": "anthropic", "model": "claude-sonnet-5-5",       "enabled": true },
    { "provider": "anthropic", "model": "claude-haiku-5-5",        "enabled": true },
    { "provider": "google",    "model": "gemini-3.8-flash",        "enabled": true },
    { "provider": "google",    "model": "gemini-4-argon",          "enabled": false },
    { "provider": "google",    "model": "gemini-3.5-flash",        "enabled": true }
  ],
  "model_timeout_seconds": 60,
  "max_retries": 3,
  "backoff_base_seconds": 3,
  "prompt_cache_affinity": true,
  "row_delay_seconds": 0.5,
  "csv": {
    "input_file": "data/TriBonds - Challenge.csv",
    "output_file": "data/TriBonds - results.csv",
    "word_columns": ["Word 1", "Word 2", "Word 3"],
    "answer_columns": ["Answer", "Answer 2"]
  }
}
```

* **Toggle Models:** Set `"enabled": false` to skip any model.
* **Reasoning Control:** `reasoning_effort` (`"none"`, `"low"`, `"medium"`, `"high"`) tunes reasoning depth for models that support it.
* **Session Cache Affinity:** `prompt_cache_affinity` sets routing headers (`x-session-affinity`) and cache keys (`prompt_cache_key`) on Fireworks serverless inference.
* **Resume Capability:** Models already completed for a given puzzle in the output CSV are skipped automatically.

### 3. Prompt Template — `prompt.json`

```json
{
  "system_prompt": "You are a word puzzle expert. Your sole task is to find the word that connects or links three given words. Rules you must follow without exception: respond with exactly ONE word phrase, no punctuation, no explanation, no sentence — just the raw connecting word itself.",
  "user_prompt": "What single word connects these three words: {word1}, {word2}, {word3}?\n\nRespond with one word phrase only."
}
```

---

## Running the Benchmark

```bash
uv run main.py
```

The script automatically resumes where it left off if interrupted.

---

## Supported Providers

| Provider | Env Variable | Capabilities |
|---|---|---|
| `fireworks` | `FIREWORKS_API_KEY` | Serverless open models (`ember-1`, `glm-5p3`, `kimi-k3`, `deepseek-v4p1`, `qwen`, `nemotron`) with session affinity |
| `anthropic` | `ANTHROPIC_API_KEY` | Claude 5.5 generation (`claude-opus-5-5`, `claude-sonnet-5-5`, `claude-haiku-5-5`) |
| `openai` | `OPENAI_API_KEY` | GPT-6 generation (`gpt-6.1-sol`, `gpt-6-astra`, `gpt-6-luna`) with reasoning effort controls |
| `google` | `GOOGLE_API_KEY` | Gemini 3.x generation (`gemini-3.8-flash`, `gemini-3.5-flash`) |
