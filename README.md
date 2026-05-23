# CollapseRadar

**Detect LLM collapse before it happens — using output structure alone.**

```
  token output:   ████████████████████ ░░░░░░░░░ │░░░│░░░░░░░░░░░
                  normal (bank full)  │ drift   │collapse

  γ_fast:         ▁▁▁▁▂▂▃▃▃▃▃▃▃▃▂▂▂▁ │▃▃▃▃▃▃▃▃ │▁▁▁▁▁▁▁▁▁▁▁▁▁▁
  γ_slow:         ▁▁▁▁▁▁▁▁▂▂▂▂▂▂▂▂▂▂ │▂▂▂▂▂▃▃▃ │▁▁▁▁▁▁▁▁▁▁▁▁▁▁
  Δγ (alert!):                        │▼▼▼▼▼▼▼▼ │
  token_rep:      ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁ │         │▃▃▃▃▃▃▃▃▃▃▃▃▃▃
                                       ↑         ↑
                                  CollapseRadar  conventional
                                  fires (t=12)   fires (t=16)
```

LLM output has internal structure — a self-similarity pattern CollapseRadar tracks as **γ** (gamma).
When the fast and slow timescales of γ diverge (**Δγ**), collapse is imminent — before any word
becomes repeated, before entropy drops, before the model produces garbage.

---

## Key Insight

> LLM collapse behaves like a phase transition.
> The critical signal is not *what* the model says, but *how its output structure self-organizes*.

**γ** = "output self-reuse rate": what fraction of the current output pattern already appeared
in the model's recent output bank. During normal generation, γ stabilizes (bank saturates).
During pre-collapse, fast-timescale γ diverges from slow-timescale γ — a hidden drift invisible
to surface metrics.

---

## Does Δγ Really Lead?

Tested on **llama3.2** (Ollama) across 5 adversarial collapse scenarios:

| Scenario | Δγ fires | token\_rep fires | entropy fires | **Lead vs rep** |
|---|---|---|---|---|
| S1 長文推論ループ | t=15 | t=19 | t=20 | **+4 steps** |
| S2 再帰タスク崩壊 | t=13 | t=17 | t=18 | **+4 steps** |
| S3 矛盾文脈ループ | t=12 | t=16 | t=9  | **+4 steps** |
| S4 過負荷プロンプト | t=22 | t=22 | t=21 | +0 steps |
| S5 敵対的推論崩壊 | t=12 | t=19 | t=20 | **+7 steps** |
| **Average** | — | — | — | **+3.8 steps** |

Each "step" ≈ 15 words of output. **+3.8 steps ≈ 57 words of early warning** before repetition
becomes visible. For long-context agents, this is the difference between catching a runaway loop
and letting it flood the context.

---

## Honest FP Analysis

Tested on 5 normal-generation prompts (creative writing, math explanation, code walkthrough,
essay, brainstorming):

| Prompt type | YELLOW fired? | Streak≥2? | Final |
|---|---|---|---|
| Creative fiction | 1 step | No | GREEN |
| Mathematical induction | 5 steps | **Yes** | GREEN |
| Hash table explanation | 6 steps | **Yes** | GREEN |
| Exploratory essay | 1 step | No | YELLOW |
| AI brainstorm list | 1 step | No | GREEN |

**FP rate at streak≥1: 100%. At streak≥2: 40%.**

Why: math proofs and structured code naturally re-use vocabulary (inductive step, base case,
hash function) — this looks like pre-collapse to a token-level detector. This is a fundamental
limit of surface features.

**Practical filter**: in CI/CD mode, use `streak≥3` or ORANGE/RED as the hard signal.
For monitoring, Δγ is best treated as an early-warning heatmap, not a binary alarm.

---

## Scenarios (10 adversarial collapse types)

| ID | Name | Collapse pattern |
|---|---|---|
| S1 | 長文推論ループ | Circular reference snowball |
| S2 | 再帰タスク崩壊 | Recursive summarization → content erasure |
| S3 | 矛盾文脈ループ | Simultaneous contradictory claims |
| S4 | 過負荷プロンプト | Mutually exclusive constraint overload |
| S5 | 敵対的推論崩壊 | Each step refutes the previous |
| S6 | コード生成崩壊 | Confident algorithm → "may or may not work" |
| S7 | 数学証明崩壊 | Proof → doubt → circular loop |
| S8 | 多言語混在崩壊 | Formal English → language mixing → chaos |
| S9 | ロールプレイ崩壊 | In-character → frame break → identity loop |
| S10 | CoT推論崩壊 | Clear chain-of-thought → hedged steps → lost answer |

---

## CRS: Collapse Resistance Score

A single number for CI/CD pipeline integration:

```
CRS ∈ [0, 1]
  1.0 = fully resistant (no collapse signal fired)
  0.0 = collapsed at first output segment
```

Per scenario: `score = first_alert_step / total_segments` (1.0 if no alert).

```bash
# Fail CI if model CRS < 0.4
python collapse_radar.py --model llama3.2 --threshold 0.4

# Compare two models
python collapse_radar.py --compare llama3.2,mistral --scenario S1,S5,S7
```

---

## Quick Start

```bash
# Requires: ollama pull llama3.2
python collapse_radar.py                          # all 10 scenarios
python collapse_radar.py --scenario S1,S5         # specific scenarios
python collapse_radar.py --fp-test                # false positive audit
python collapse_radar.py --no-llm                 # simulated (no Ollama)
python collapse_radar.py --model mistral          # any Ollama model
python collapse_radar.py --threshold 0.4          # CI/CD exit code
```

```
pip install numpy
# No other dependencies. No API keys. No external models.
```

---

## How γ Works (Technical)

```python
# Linguistic feature vector (6 dims per 15-word segment):
vec = [type_token_ratio, hedge_density, long_word_ratio,
       bigram_repetition, adverb_density, sentence_length_norm]

# Quantize → state bank
key = tuple((vec / eps).round().astype(int))  # eps=0.25
hit = key in bank
bank[key] += 1

# Multi-timescale EMA
γ_fast = 0.40 * hit + 0.60 * γ_fast   # reacts in ~3 segments
γ_slow = 0.04 * hit + 0.96 * γ_slow   # reacts in ~25 segments
Δγ = γ_fast - γ_slow                   # divergence = precursor signal
```

Bank freezes once stable (MACD-style): subsequent text is scored against the frozen
"normal" reference. Anomaly = anything that doesn't match the initial stable pattern.

**No external model. No API. No training. Fully online. Unsupervised.**

---

## What This Is (and Isn't)

**Is:** A lightweight unsupervised precursor detector for structural drift in LLM output.
Works on any model, any language, any framework, without access to logits or weights.

**Isn't:** A semantic quality judge. CollapseRadar cannot tell if the *content* is correct —
only if the *output structure* is drifting toward known collapse patterns.

---

## Contact

Research questions / collaboration: yoiyoicarnival@gmail.com

---

## Citation

```bibtex
@software{collapseradar2026,
  title  = {CollapseRadar: LLM Collapse Precursor Detection via Multi-Scale Gamma Divergence},
  author = {B126},
  year   = {2026},
  url    = {https://github.com/yoiyoicarnival/collapse-radar}
}
```
