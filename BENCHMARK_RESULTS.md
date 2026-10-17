# TriBonds LLM Benchmark Report: Frontier Closed (Google, OpenAI, Claude) vs. Fireworks Open Models

**Dataset:** [`data/TriBonds - Challenge.csv`](./data/TriBonds%20-%20Challenge.csv) (300 lateral association puzzles)  
**Output Results:** [`data/TriBonds - results.csv`](./data/TriBonds%20-%20results.csv)  
**Total Models Evaluated:** 21 models across Google Gemini, OpenAI GPT-6, Anthropic Claude 5.5, and Fireworks AI open-weight models  
**Total Queries Evaluated:** 6,300 benchmark queries  

---

## 1. Executive Summary & Key Highlights

All **21 models** across commercial closed frontier leaders (**Google**, **OpenAI**, **Anthropic**) and open-weight models (**Fireworks AI**) were evaluated across all 300 puzzles in the TriBonds lateral reasoning benchmark.

### Major Findings:
1. **The Frontier Podium (84.7% – 85.7%)**:
   * 🥇 **`claude-opus-5-5`** (Anthropic) took #1 with **257/300 (85.7%)** strict accuracy.
   * 🥈 **`gpt-6-astra`** (OpenAI) took #2 with **255/300 (85.0%)** strict accuracy.
   * 🥉 **`gemini-3.8-flash`** (Google) captured #3 with **254/300 (84.7%)** strict accuracy, establishing itself as the highest-accuracy model under \$1/1M tokens (\$0.75 / \$3.75).
2. **Workhorse Parity (84.3%)**:
   * **`gemini-3.5-flash`** and **`gpt-6.1-sol`** tied for 4th place at **253/300 (84.3%)**.
3. **Open-Weight Frontier Closeness (`ember-1`)**:
   * Fireworks' **`ember-1`** achieved **250/300 (83.3%)**, trailing the top frontier models by just 4–7 puzzles overall, while costing only **\$0.30 / \$1.20 per 1M tokens** (over **40x cheaper** than GPT-6 Astra and **16x cheaper** than Claude Opus).
   * It outperformed Anthropic's **`claude-sonnet-5-5` (79.3%)** and **`claude-haiku-5-5` (81.0%)**.
4. **Latency Champion: `glm-5p3-fast`**:
   * Achieved **81.7% accuracy** (245/300) with a blazing **0.61s median latency**—3x to 5x faster than closed frontier models.
5. **Value Front**:
   * **`gemini-3.8-flash`** (84.7% at \$0.75/\$3.75) and **`gpt-6-luna`** (83.7% at \$0.10/\$0.50) provide the best price-to-accuracy ratio among closed providers.
   * **`glm-5p3-flash`** (81.3% at 0.72s for \$0.15/\$0.50) and **`ember-1`** (83.3% for \$0.30/\$1.20) lead the open-weight value front.

---

## 2. Complete 21-Model Leaderboard

| Rank | Model | Provider | Strict Passes | Strict (%) | Median Latency | Mean Latency | Pricing (In / Out per 1M) |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| 🥇 | **`claude-opus-5-5`** | Anthropic | **257 / 300** | **85.7%** | **1.60s** | 1.99s | \$4.00 / \$20.00 |
| 🥈 | **`gpt-6-astra`** | OpenAI | **255 / 300** | **85.0%** | **1.98s** | 2.60s | \$10.00 / \$50.00 |
| 🥉 | **`gemini-3.8-flash`** | Google | **254 / 300** | **84.7%** | **2.96s** | 3.98s | **\$0.75 / \$3.75** |
| 4 | **`gemini-3.5-flash`** | Google | **253 / 300** | **84.3%** | **1.93s** | 2.37s | \$1.50 / \$9.00 |
| 4 | **`gpt-6.1-sol`** | OpenAI | **253 / 300** | **84.3%** | **2.27s** | 2.95s | \$2.00 / \$10.00 |
| 6 | **`gpt-6-luna`** | OpenAI | **251 / 300** | **83.7%** | **1.59s** | 2.02s | **\$0.10 / \$0.50** |
| 7 | **`ember-1`** | Fireworks | **250 / 300** | **83.3%** | **2.36s** | 2.70s | **\$0.30 / \$1.20** |
| 8 | **`glm-5p3-fast`** | Fireworks | **245 / 300** | **81.7%** | **0.61s** | 0.86s | \$2.10 / \$6.60 |
| 8 | **`kimi-k3`** | Fireworks | **245 / 300** | **81.7%** | **1.78s** | 2.16s | \$3.00 / \$15.00 |
| 10 | **`glm-5p3-flash`** | Fireworks | **244 / 300** | **81.3%** | **0.72s** | 1.23s | \$0.15 / \$0.50 |
| 11 | **`claude-haiku-5-5`** | Anthropic | **243 / 300** | **81.0%** | **1.34s** | 1.60s | \$0.10 / \$0.50 |
| 12 | **`glm-5p3`** | Fireworks | **242 / 300** | **80.7%** | **0.92s** | 2.67s | \$1.40 / \$4.40 |
| 13 | **`kimi-k3-fast`** | Fireworks | **240 / 300** | **80.0%** | **1.69s** | 3.95s | \$4.50 / \$22.50 |
| 14 | **`claude-sonnet-5-5`** | Anthropic | **238 / 300** | **79.3%** | **1.40s** | 1.78s | \$2.00 / \$10.00 |
| 15 | **`qwen3p8-2p4t-a95b`** | Fireworks | **235 / 300** | **78.3%** | **1.87s** | 2.21s | \$0.60 / \$2.00 |
| 16 | **`qwen3p8-max`** | Fireworks | **234 / 300** | **78.0%** | **1.81s** | 2.20s | \$1.40 / \$4.40 |
| 17 | **`deepseek-v4p1-flash`** | Fireworks | **231 / 300** | **77.0%** | **0.74s** | 0.98s | \$0.30 / \$1.20 |
| 17 | **`gpt-oss-120b`** | Fireworks | **231 / 300** | **77.0%** | **0.78s** | 1.22s | \$0.15 / \$0.60 |
| 19 | **`nemotron-3-ultra`** | Fireworks | **223 / 300** | **74.3%** | **1.65s** | 1.82s | \$1.40 / \$4.40 |
| 20 | **`minimax-m3`** | Fireworks | **211 / 300** | **70.3%** | **0.78s** | 1.11s | \$0.30 / \$1.20 |
| 21 | **`nemotron-lightning`** | Fireworks | **154 / 300** | **51.3%** | **0.35s** | 0.42s | \$0.15 / \$0.60 |

---

## 3. Provider Head-to-Head Comparison

| Metric | Anthropic | OpenAI | Google | Fireworks |
| :--- | :--- | :--- | :--- | :--- |
| **Top Model** | `claude-opus-5-5` | `gpt-6-astra` | `gemini-3.8-flash` | `ember-1` |
| **Top Accuracy** | **85.7%** | 85.0% | 84.7% | 83.3% |
| **Median Latency** | 1.60s | 1.98s | 2.96s | 2.36s (or **0.61s** on GLM) |
| **Flagship Token Price (In / Out)** | \$4.00 / \$20.00 | \$10.00 / \$50.00 | **\$0.75 / \$3.75** | **\$0.30 / \$1.20** |
| **Price Ratio vs GPT-6 Astra** | 2.5x cheaper | Baseline | **13x cheaper** | **41x cheaper** |

---

## 4. Architectural & Benchmark Insights

1. **Google Gemini 3.8 Flash is an Outstanding Reasoning Value**:
   * At 84.7% accuracy, `gemini-3.8-flash` essentially matched the \$50/1M `gpt-6-astra` (85.0%) while costing only \$0.75 / \$3.75 per 1M tokens.
2. **Open-Weight Parity Holds Strong**:
   * Fireworks' `ember-1` (83.3%) sits within 2.4 percentage points of the world #1 (`claude-opus-5-5` at 85.7%), demonstrating that open-weight architectures combined with optimized reasoning loops are competitive with multi-billion-dollar proprietary frontier models.
3. **Sub-Second Real-Time Workloads**:
   * For applications requiring real-time answers (<1 second), **`glm-5p3-fast` (0.61s, 81.7%)** and **`glm-5p3-flash` (0.72s, 81.3%)** outperform any commercial frontier model in latency by 2x to 4x while maintaining >81% accuracy.
