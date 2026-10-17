import asyncio
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

from pricing import cost_usd

load_dotenv()

# Download required NLTK data on first run (no-ops if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_lemmatizer = WordNetLemmatizer()
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


@dataclass
class QueryResult:
    word: str
    latency_s: float
    prompt_tokens: int = 0
    cached_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0


def new_session_key(prefix: str = "tribonds") -> str:
    """Sticky routing ID so Fireworks pins requests to replica GPU cache."""
    return f"{prefix}-{uuid.uuid4().hex}"


def clean_answer(value) -> str:
    """Strip parenthetical content (e.g. '(brand name)') and whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s*\([^)]*\)", "", str(value)).strip()


def extract_word(text: str) -> str:
    """Pull the first word out of a model response, stripping thinking blocks and punctuation."""
    # Strip <think>...</think> reasoning blocks
    text = _THINK.sub("", text).strip()
    if "<|close|>think" in text:
        text = text.split("<|close|>think")[-1].strip()
    if not text:
        return ""
    # Strip markdown bolding / code ticks if present
    first = text.split()[0].replace("*", "").replace("`", "")
    return re.sub(r"[^\w'-]", "", first)


def lemma(word: str) -> str:
    """Reduce a word to its base lemma (noun form) for comparison."""
    return _lemmatizer.lemmatize(word.lower(), pos="n")


def grade(response: str, answer1: str, answer2: str) -> str:
    r = lemma(response)
    if answer1 and r == lemma(answer1):
        return "pass"
    if answer2 and r == lemma(answer2):
        return "pass"
    return "fail"


def build_llm(model_cfg: dict, prompt_cache_affinity: bool = True):
    """Instantiate the appropriate LangChain chat model."""
    provider = model_cfg["provider"]
    model = model_cfg["model"]
    reasoning_effort = model_cfg.get("reasoning_effort")

    if provider == "fireworks":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("FIREWORKS_API_KEY")
        if not api_key:
            raise ValueError(f"Provider 'fireworks' is enabled for model '{model}', but FIREWORKS_API_KEY is not set in environment or .env")
        base_url = os.getenv("FIREWORKS_BASE_URL", "https://api.fireworks.ai/inference/v1")
        extra_body = {}
        default_headers = {}
        if prompt_cache_affinity:
            session_key = new_session_key()
            extra_body["prompt_cache_key"] = session_key
            default_headers["x-session-affinity"] = session_key
        if reasoning_effort:
            extra_body["reasoning_effort"] = reasoning_effort

        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            temperature=0,
            max_tokens=300,
            extra_body=extra_body if extra_body else None,
            default_headers=default_headers if default_headers else None,
        )
    elif provider == "openai":
        from langchain_openai import ChatOpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(f"Provider 'openai' is enabled for model '{model}', but OPENAI_API_KEY is not set in environment or .env")
        kwargs = {"model": model, "api_key": api_key, "max_tokens": 1000}
        extra_body = {}
        if reasoning_effort and reasoning_effort != "none":
            extra_body["reasoning_effort"] = reasoning_effort
        # GPT-6 and o-series reasoning models reject temperature=0
        if not (model.startswith("gpt-6") or model.startswith("o")):
            kwargs["temperature"] = 0
        if extra_body:
            kwargs["extra_body"] = extra_body
        return ChatOpenAI(**kwargs)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(f"Provider 'anthropic' is enabled for model '{model}', but ANTHROPIC_API_KEY is not set in environment or .env")
        kwargs = {"model": model, "api_key": api_key, "max_tokens": 1000}
        # Claude 5.5 models deprecate temperature
        if not ("-5-5" in model or "-5.5" in model or "-5-" in model):
            kwargs["temperature"] = 0
        return ChatAnthropic(**kwargs)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(f"Provider 'google' is enabled for model '{model}', but GOOGLE_API_KEY is not set in environment or .env")
        return ChatGoogleGenerativeAI(model=model, google_api_key=api_key, temperature=0, max_output_tokens=1000)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


async def query_model_async(
    llm,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    model_name: str = "",
    label: str = "",
) -> QueryResult:
    """Query a model asynchronously with latency tracking, token accounting, and exponential backoff."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    for attempt in range(max_retries + 1):
        try:
            start = time.perf_counter()
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            latency = time.perf_counter() - start

            content = response.content
            # Flatten lists of content blocks (Anthropic thinking/text parts, Gemini parts, etc.)
            if isinstance(content, list):
                text_parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "thinking":
                            continue
                        text_parts.append(part.get("text", str(part)))
                    elif hasattr(part, "text"):
                        text_parts.append(part.text)
                    else:
                        text_parts.append(str(part))
                content = " ".join(text_parts)

            # Robust token extraction across OpenAI, Anthropic, Google, and Fireworks
            meta = getattr(response, "response_metadata", {}) or {}
            usage = getattr(response, "usage_metadata", {}) or meta.get("token_usage", {}) or meta.get("usage", {}) or {}

            prompt_tok = (
                usage.get("prompt_tokens")
                or usage.get("input_tokens")
                or usage.get("prompt_token_count")
                or 0
            )
            comp_tok = (
                usage.get("completion_tokens")
                or usage.get("output_tokens")
                or usage.get("candidates_token_count")
                or 0
            )
            cached_details = usage.get("prompt_tokens_details") or usage.get("input_token_details") or {}
            cached_tok = (
                cached_details.get("cached_tokens")
                or cached_details.get("cache_read")
                or usage.get("cache_read_input_tokens")
                or usage.get("cached_content_token_count")
                or 0
            )
            call_cost = cost_usd(model_name, prompt_tok, cached_tok, comp_tok)

            return QueryResult(
                word=extract_word(content),
                latency_s=latency,
                prompt_tokens=prompt_tok,
                cached_tokens=cached_tok,
                completion_tokens=comp_tok,
                cost_usd=call_cost,
            )
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                print(f"    [{label}] timeout — retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"    [{label}] all {max_retries} retries exhausted — giving up")
                return QueryResult(word="TIMEOUT", latency_s=float(timeout))
        except Exception as e:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                print(f"    [{label}] error ({e}) — retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"    [{label}] all {max_retries} retries exhausted — giving up")
                return QueryResult(word=f"ERROR: {e}", latency_s=0.0)
    return QueryResult(word="TIMEOUT", latency_s=float(timeout))


def col_prefix(provider: str, model: str) -> str:
    short_model = model.split("/")[-1] if "/" in model else model
    safe = re.sub(r"[^a-zA-Z0-9]", "_", short_model)
    return f"{provider}_{safe}"


async def main():
    with open("config.json") as f:
        config = json.load(f)
    with open("prompt.json") as f:
        prompt_cfg = json.load(f)

    system_prompt = prompt_cfg["system_prompt"]
    user_template = prompt_cfg["user_prompt"]

    input_file = config["csv"]["input_file"]
    output_file = config["csv"]["output_file"]
    word_cols = config["csv"]["word_columns"]
    answer_cols = config["csv"]["answer_columns"]
    timeout = config.get("model_timeout_seconds", 60)
    max_retries = config.get("max_retries", 3)
    backoff_base = config.get("backoff_base_seconds", 3)
    prompt_cache_affinity = config.get("prompt_cache_affinity", True)
    row_delay = config.get("row_delay_seconds", 0.5)

    df = pd.read_csv(input_file)
    total = len(df)

    active_models = [m for m in config["models"] if m.get("enabled", True)]
    if not active_models:
        print("No enabled models in config.json — nothing to do.")
        sys.exit(1)

    prefixes = [col_prefix(m["provider"], m["model"]) for m in active_models]
    all_fieldnames = ["words", "answer", "answer 2"]
    for prefix in prefixes:
        all_fieldnames += [
            f"{prefix}_response",
            f"{prefix}_grade",
            f"{prefix}_latency_s",
        ]

    # Load existing output file if present
    existing_records = {}
    existing_cols = []
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        try:
            existing_df = pd.read_csv(output_file)
            existing_cols = list(existing_df.columns)
            for _, r in existing_df.iterrows():
                w = r["words"]
                if pd.notna(w):
                    existing_records[str(w).strip()] = r.to_dict()
            print(f"Loaded existing results: {len(existing_records)} rows found.")
        except Exception as e:
            print(f"Could not load existing {output_file}: {e}")

    # Ensure all existing columns are preserved plus active model columns
    all_fieldnames = list(existing_cols) if existing_cols else ["words", "answer", "answer 2"]
    for prefix in prefixes:
        for suffix in ["_response", "_grade", "_latency_s"]:
            col = f"{prefix}{suffix}"
            if col not in all_fieldnames:
                all_fieldnames.append(col)

    # Build LLMs up front
    llms = {}
    for m in active_models:
        short_id = m["model"].split("/")[-1]
        print(f"Loading model: {m['provider']}/{short_id}")
        llms[(m["provider"], m["model"])] = build_llm(m, prompt_cache_affinity=prompt_cache_affinity)

    new_count = 0
    results_rows = []

    for i, row in df.iterrows():
        word1 = str(row[word_cols[0]]).strip()
        word2 = str(row[word_cols[1]]).strip()
        word3 = str(row[word_cols[2]]).strip()
        words_key = f"{word1}, {word2}, {word3}"

        answer1 = clean_answer(row[answer_cols[0]])
        answer2 = clean_answer(row[answer_cols[1]]) if len(answer_cols) > 1 else ""

        existing_row = existing_records.get(words_key, {})
        result_row = dict(existing_row)
        result_row["words"] = words_key
        result_row["answer"] = answer1
        result_row["answer 2"] = answer2

        # Check which models need to be queried
        models_to_query = []
        for m in active_models:
            prefix = col_prefix(m["provider"], m["model"])
            resp_val = existing_row.get(f"{prefix}_response")
            grade_val = existing_row.get(f"{prefix}_grade")
            lat_val = existing_row.get(f"{prefix}_latency_s")

            # Check if this model is already cleanly completed
            is_done = (
                pd.notna(resp_val)
                and str(resp_val).strip() != ""
                and not str(resp_val).startswith("ERROR:")
                and pd.notna(grade_val)
            )

            if is_done:
                result_row[f"{prefix}_response"] = resp_val
                result_row[f"{prefix}_grade"] = grade_val
                result_row[f"{prefix}_latency_s"] = lat_val
            else:
                models_to_query.append(m)

        if not models_to_query:
            results_rows.append(result_row)
            continue

        user_prompt = user_template.format(word1=word1, word2=word2, word3=word3)
        to_query_names = ", ".join(m["model"].split("/")[-1] for m in models_to_query)
        print(f"\n[{i + 1}/{total}] {word1} | {word2} | {word3}  →  expected: {answer1 or answer2} (querying: {to_query_names})")

        # Models requiring query fire concurrently
        tasks = [
            query_model_async(
                llms[(m["provider"], m["model"])],
                system_prompt,
                user_prompt,
                timeout,
                max_retries,
                backoff_base,
                model_name=m["model"],
                label=f"{m['provider']}/{m['model'].split('/')[-1]}",
            )
            for m in models_to_query
        ]
        responses = await asyncio.gather(*tasks)

        for m, res in zip(models_to_query, responses):
            prefix = col_prefix(m["provider"], m["model"])
            verdict = grade(res.word, answer1, answer2)
            result_row[f"{prefix}_response"] = res.word
            result_row[f"{prefix}_grade"] = verdict
            result_row[f"{prefix}_latency_s"] = f"{res.latency_s:.2f}"
            short_id = m["model"].split("/")[-1]
            print(f"  {m['provider']}/{short_id}: {res.word!r} → {verdict} ({res.latency_s:.2f}s)")

        results_rows.append(result_row)
        new_count += 1

        # Checkpoint: save DataFrame immediately
        checkpoint_df = pd.DataFrame(results_rows)
        for col in all_fieldnames:
            if col not in checkpoint_df.columns:
                checkpoint_df[col] = None
        checkpoint_df[all_fieldnames].to_csv(output_file, index=False)

        if row_delay > 0:
            await asyncio.sleep(row_delay)

    # Save final DataFrame ensuring full row count
    final_df = pd.DataFrame(results_rows)
    for col in all_fieldnames:
        if col not in final_df.columns:
            final_df[col] = None
    final_df[all_fieldnames].to_csv(output_file, index=False)

    # Final summary over the full output file
    print(f"\n{'=' * 75}")
    print(f"{'RESULTS SUMMARY':^75}")
    print(f"{'=' * 75}")
    print(f"New / updated rows this run: {new_count}")
    print(f"{'-' * 75}")
    print(f"{'Model':<42} | {'Accuracy':<14} | {'Avg Latency':<12}")
    print(f"{'-' * 75}")

    results_df = pd.read_csv(output_file)
    count = len(results_df)
    grade_cols = [c for c in results_df.columns if c.endswith("_grade")]
    for grade_col in grade_cols:
        prefix = grade_col[:-6]
        lat_col = f"{prefix}_latency_s"
        passes = (results_df[grade_col] == "pass").sum()
        pct = passes / count * 100 if count else 0
        lat_mean = (
            pd.to_numeric(results_df[lat_col], errors="coerce").mean()
            if lat_col in results_df.columns
            else None
        )
        lat_str = f"{lat_mean:.2f}s" if pd.notna(lat_mean) else "N/A"
        print(f"{prefix:<42} | {passes}/{count} ({pct:5.1f}%) | {lat_str:<12}")

    print(f"{'=' * 75}")
    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
