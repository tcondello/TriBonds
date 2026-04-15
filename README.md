# TriBonds LLM Benchmark

Benchmarks LLMs on word-connection puzzles. Given three words, models must identify the single word that links all three. Results across providers are written to a CSV as they arrive.

## How it works

For each row in a challenge CSV, the script:
1. Sends the three words to every enabled model simultaneously (async)
2. Extracts a single-word answer from each response
3. Grades it against one or two valid answers using lemmatization (so "Plane" matches "Planes")
4. Writes the row to the output CSV immediately — safe to interrupt and resume

## Setup

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/YOUR_USERNAME/TriBonds.git
cd TriBonds
uv sync
cp .env.example .env
# add your API keys to .env
```

## Configuration

### API keys — `.env`

```
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
```

### Models — `config.json`

```json
{
  "models": [
    { "provider": "openai",    "model": "gpt-5.4",              "enabled": true },
    { "provider": "anthropic", "model": "claude-opus-4-6",       "enabled": true },
    { "provider": "google",    "model": "gemini-3-flash-preview", "enabled": false }
  ],
  "model_timeout_seconds": 60,
  "max_retries": 3,
  "backoff_base_seconds": 5,
  "csv": {
    "input_file": "data/TriBonds - Challenge.csv",
    "output_file": "data/TriBonds - results.csv",
    "word_columns": ["Word 1", "Word 2", "Word 3"],
    "answer_columns": ["Answer", "Answer 2"]
  }
}
```

- Set `"enabled": false` to skip a model without removing it
- `answer_columns` accepts one or two columns — either counts as a correct answer
- Timeout and backoff apply per model call; backoff doubles each retry (`base * 2^attempt`)

### Prompt — `prompt.json`

```json
{
  "system_prompt": "...",
  "user_prompt": "... {word1}, {word2}, {word3} ..."
}
```

Edit either field freely. The `{word1}`, `{word2}`, `{word3}` placeholders are required in `user_prompt`.

## Challenge CSV format

| Column | Description |
|---|---|
| `Word 1`, `Word 2`, `Word 3` | The three words to connect |
| `Answer` | The correct connecting word |
| `Answer 2` *(optional)* | A second valid answer |

Answers are cleaned before grading — anything in parentheses is stripped, e.g. `Bank (financial)` becomes `Bank`.

## Running

```bash
uv run main.py
```

The script resumes automatically if interrupted — rows already in the output CSV are skipped.

## Output CSV format

| Column | Description |
|---|---|
| `words` | The three input words joined by `, ` |
| `answer` | Cleaned primary answer |
| `answer 2` | Cleaned secondary answer |
| `{provider}_{model}_response` | What the model returned |
| `{provider}_{model}_grade` | `pass` or `fail` |

A summary table is printed at the end of each run.

## Supported providers

| Provider | Env var |
|---|---|
| `openai` | `OPENAI_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` |
| `google` | `GOOGLE_API_KEY` |
