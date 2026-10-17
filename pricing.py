"""Token pricing schedules, USD per 1M tokens: (input, cached input, output).

Sources:
- Fireworks AI serverless pricing
- OpenAI API pricing (GPT-6 series & GPT-5.4)
- Anthropic Claude API pricing (Claude 5.5 series & Claude 4.6)
- Google Gemini API pricing (Gemini 3.8 / 4 series & preview)
"""

from typing import Dict, Tuple

PRICES: Dict[str, Tuple[float, float, float]] = {
    # Fireworks AI Models
    "accounts/fireworks/models/kimi-k3": (3.00, 0.30, 15.00),
    "accounts/fireworks/routers/kimi-k3-fast": (4.50, 0.45, 22.50),
    "accounts/fireworks/models/glm-5p3": (1.40, 0.26, 4.40),
    "accounts/fireworks/routers/glm-5p3-fast": (2.10, 0.39, 6.60),
    "accounts/fireworks/models/glm-5p3-flash": (0.15, 0.03, 0.50),
    "accounts/fireworks/models/glm-5p2": (1.40, 0.14, 4.40),
    "accounts/fireworks/models/deepseek-v4p1-flash": (0.30, 0.006, 1.20),
    "accounts/fireworks/models/minimax-m3": (0.30, 0.06, 1.20),
    "accounts/fireworks/models/gpt-oss-120b": (0.15, 0.015, 0.60),
    "accounts/fireworks/models/ember-1": (0.30, 0.03, 1.20),
    "accounts/fireworks/models/nemotron-3-ultra-nvfp4": (1.40, 0.14, 4.40),
    "accounts/fireworks/models/nemotron-lightning-3p5-30b-a3b": (0.15, 0.015, 0.60),
    "accounts/fireworks/models/qwen3p8-max": (1.40, 0.26, 4.40),
    "accounts/fireworks/models/qwen3p8-2p4t-a95b": (0.60, 0.06, 2.00),
    "accounts/fireworks/models/inkling": (0.50, 0.05, 2.00),

    # OpenAI Models (Latest GPT-6 generation + GPT-5.4 baseline)
    "gpt-6.1-sol": (2.00, 0.10, 10.00),
    "gpt-6-sol": (2.00, 0.10, 10.00),
    "gpt-6-astra": (10.00, 1.00, 50.00),
    "gpt-6-luna": (0.10, 0.01, 0.50),
    "gpt-5.4": (5.00, 1.25, 15.00),

    # Anthropic Claude Models (Latest Claude 5.5 generation + Claude 4.6 baseline)
    "claude-opus-5-5": (4.00, 0.20, 20.00),
    "claude-sonnet-5-5": (2.00, 0.10, 10.00),
    "claude-haiku-5-5": (0.10, 0.01, 0.50),
    "claude-opus-4-6": (15.00, 1.50, 75.00),

    # Google Gemini Models (Latest Gemini 3.8 / 4 generation + preview baseline)
    "gemini-4-argon": (2.00, 0.20, 10.00),
    "gemini-3.8-flash": (0.75, 0.075, 3.75),
    "gemini-3.5-flash": (1.50, 0.15, 9.00),
    "gemini-3.5-flash-lite": (0.075, 0.015, 0.30),
    "gemini-2.5-flash": (0.075, 0.015, 0.30),
    "gemini-3-flash-preview": (0.15, 0.0375, 0.60),
}


def cost_usd(model: str, prompt_tokens: int, cached_tokens: int, completion_tokens: int) -> float:
    """Calculate token cost in USD based on provider pricing schedules."""
    price = PRICES.get(model)
    if price is None and "/" in model:
        price = PRICES.get(model.split("/")[-1])
    if price is None:
        return 0.0
    p_in, p_cached, p_out = price
    uncached = max(prompt_tokens - cached_tokens, 0)
    return (uncached * p_in + cached_tokens * p_cached + completion_tokens * p_out) / 1_000_000
