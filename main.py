import asyncio
import csv
import json
import re
import os
import sys
import nltk
import pandas as pd
from nltk.stem import WordNetLemmatizer
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

# Download required NLTK data on first run (no-ops if already present)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)

_lemmatizer = WordNetLemmatizer()


def clean_answer(value) -> str:
    """Strip parenthetical content (e.g. '(brand name)') and whitespace."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s*\([^)]*\)", "", str(value)).strip()


def extract_word(text: str) -> str:
    """Pull the first word out of a model response and strip punctuation."""
    text = text.strip()
    if not text:
        return ""
    first = text.split()[0]
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


def build_llm(provider: str, model: str):
    """Instantiate the appropriate LangChain chat model."""
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model, api_key=os.getenv("OPENAI_API_KEY"), temperature=0)
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model, api_key=os.getenv("ANTHROPIC_API_KEY"), temperature=0)
    elif provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=model, google_api_key=os.getenv("GOOGLE_API_KEY"), temperature=0)
    else:
        raise ValueError(f"Unknown provider: {provider!r}")


async def query_model_async(
    llm,
    system_prompt: str,
    user_prompt: str,
    timeout: int,
    max_retries: int,
    backoff_base: float,
    label: str = "",
) -> str:
    """Query a model asynchronously with timeout and exponential backoff on retry."""
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ]
    for attempt in range(max_retries + 1):
        try:
            response = await asyncio.wait_for(llm.ainvoke(messages), timeout=timeout)
            content = response.content
            # Gemini returns a list of content parts; flatten to string
            if isinstance(content, list):
                content = " ".join(
                    part["text"] if isinstance(part, dict) and "text" in part else str(part)
                    for part in content
                )
            return extract_word(content)
        except asyncio.TimeoutError:
            if attempt < max_retries:
                wait = backoff_base * (2 ** attempt)
                print(f"    [{label}] timeout — retrying in {wait:.0f}s (attempt {attempt + 1}/{max_retries})")
                await asyncio.sleep(wait)
            else:
                print(f"    [{label}] all {max_retries} retries exhausted — giving up")
                return "TIMEOUT"
        except Exception as e:
            return f"ERROR: {e}"
    return "TIMEOUT"


def col_prefix(provider: str, model: str) -> str:
    safe = re.sub(r"[^a-zA-Z0-9]", "_", model)
    return f"{provider}_{safe}"


def load_completed_words(output_file: str) -> set[str]:
    """Return the set of 'words' values already written to the output CSV."""
    if not os.path.exists(output_file):
        return set()
    try:
        df = pd.read_csv(output_file)
        return set(df["words"].dropna().tolist())
    except Exception:
        return set()


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
    backoff_base = config.get("backoff_base_seconds", 5)

    df = pd.read_csv(input_file)
    total = len(df)

    active_models = [m for m in config["models"] if m.get("enabled", True)]
    if not active_models:
        print("No enabled models in config.json — nothing to do.")
        sys.exit(1)

    prefixes = [col_prefix(m["provider"], m["model"]) for m in active_models]
    fieldnames = ["words", "answer", "answer 2"]
    for prefix in prefixes:
        fieldnames += [f"{prefix}_response", f"{prefix}_grade"]

    # Resume: skip rows already in the output file
    completed = load_completed_words(output_file)
    if completed:
        print(f"Resuming — {len(completed)} rows already done, skipping.")

    file_exists = os.path.exists(output_file) and os.path.getsize(output_file) > 0

    # Build LLMs up front
    llms = {}
    for m in active_models:
        print(f"Loading model: {m['provider']}/{m['model']}")
        llms[(m["provider"], m["model"])] = build_llm(m["provider"], m["model"])

    new_count = 0

    with open(output_file, "a", newline="", encoding="utf-8") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for i, row in df.iterrows():
            word1 = str(row[word_cols[0]]).strip()
            word2 = str(row[word_cols[1]]).strip()
            word3 = str(row[word_cols[2]]).strip()
            words_key = f"{word1}, {word2}, {word3}"

            if words_key in completed:
                continue

            answer1 = clean_answer(row[answer_cols[0]])
            answer2 = clean_answer(row[answer_cols[1]]) if len(answer_cols) > 1 else ""
            user_prompt = user_template.format(word1=word1, word2=word2, word3=word3)

            print(f"\n[{i + 1}/{total}] {word1} | {word2} | {word3}  →  expected: {answer1 or answer2}")

            # All models fire concurrently for this row
            tasks = [
                query_model_async(
                    llms[(m["provider"], m["model"])],
                    system_prompt,
                    user_prompt,
                    timeout,
                    max_retries,
                    backoff_base,
                    label=f"{m['provider']}/{m['model']}",
                )
                for m in active_models
            ]
            responses = await asyncio.gather(*tasks)

            result_row: dict = {"words": words_key, "answer": answer1, "answer 2": answer2}
            for m, response in zip(active_models, responses):
                prefix = col_prefix(m["provider"], m["model"])
                verdict = grade(response, answer1, answer2)
                result_row[f"{prefix}_response"] = response
                result_row[f"{prefix}_grade"] = verdict
                print(f"  {m['provider']}/{m['model']}: {response!r} → {verdict}")

            writer.writerow(result_row)
            csvfile.flush()  # persist immediately
            new_count += 1
            await asyncio.sleep(3)

    # Final summary over the full output file
    print(f"\n{'=' * 50}")
    print("RESULTS SUMMARY")
    print(f"{'=' * 50}")
    print(f"New rows this run: {new_count}  |  Previously completed: {len(completed)}")

    results_df = pd.read_csv(output_file)
    count = len(results_df)
    for m in active_models:
        prefix = col_prefix(m["provider"], m["model"])
        grade_col = f"{prefix}_grade"
        if grade_col in results_df.columns:
            passes = (results_df[grade_col] == "pass").sum()
            pct = passes / count * 100 if count else 0
            print(f"  {m['provider']}/{m['model']}: {passes}/{count} ({pct:.1f}%)")

    print(f"\nFull results saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
