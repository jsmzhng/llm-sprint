# Parts V–VIII Implementation Plan

Status as of 2026-04-13. Parts I–IV notebooks are built and merged. The four reviews
have been incorporated into the curriculum outline, including several new modules
(0, 4.5, 11.5, 17.5, 22.5, 25.5) and Module 25 expansion. This document plans the
remaining notebook work and the sequencing for V–VIII.

## Guiding principles

1. **One notebook per module by default.** Parts V–VIII modules are dense; the
   experience in Part IV confirmed that per-module notebooks are the right grain.
2. **Dependencies drive order.** Notebooks that depend on earlier ones must be
   built after them (e.g. Module 21 GRPO uses the nanoGPT from Part III; Module 25
   speculative decoding uses Module 13's KV cache infrastructure).
3. **"Break things on purpose" mandate.** Every new notebook needs at least one
   experiment where a critical component is disabled and the failure is made visible.
4. **Currency.** Web-search for the latest numbers on every frontier claim
   (TurboQuant-KV/RotorQuant were moving targets as of April 2026; so are
   Medusa/EAGLE/Lookahead benchmarks).
5. **Style.** Same conversational, no-AI-tone, narrative-markdown voice as Parts I–IV.
   No bullet-point lectures. LaTeX math with symbol-by-symbol walkthrough.

## Revised module list for V–VIII (post-review)

### Part V — Generation & Alignment
| # | Module | Status | Notes |
|---|---|---|---|
| 18 | Next-Token Prediction | new | Short notebook; re-uses nanoGPT from Part III |
| 19 | Sampling Strategies | new | Temperature, top-k, top-p, min-p with live sliders |
| 20 | RLHF & DPO (+SimPO/KTO/ORPO) | new | Can be a single notebook if kept tight |
| 21 | GRPO & RLVR (+DAPO) | new | Headline module; rich interactive demo |
| 22 | Prompt Eng. as Bayesian Conditioning | new | Short conceptual notebook |
| 22.5 | **Alignment Stack & Red Teaming** | new (added in review) | Constitutional AI, PyRIT, HarmBench |

### Part VI — Reasoning & Test-Time Compute
| # | Module | Status | Notes |
|---|---|---|---|
| 23 | Reasoning Models & CoT | new | Qualitative + inspection of R1-style traces |
| 24 | Test-Time Compute Scaling | new | Thinking-budget plots; key empirical module |
| 25 | Speculative Decoding (+ Medusa/EAGLE/Lookahead) | new | Reuses Part IV KV cache |
| 25.5 | **Sparse Autoencoders & Mech Interp** | new (added in review) | Train tiny SAE on GPT-2 |

### Part VII — Serving Infrastructure & Tooling
| # | Module | Status | Notes |
|---|---|---|---|
| 26 | vLLM & Serving Engines | new | Needs Docker or a cloud runtime; may need simulation fallback |
| 27 | LiteLLM & Model Routing | new | Live API calls; keep budget cap obvious |
| 28 | Continuous Batching & Scheduling | new | Animated scheduling grid |
| 29 | Distributed Inference (TP/PP) | new | Pure simulation; no multi-GPU required |

### Part VIII — Agents & Evaluation
| # | Module | Status | Notes |
|---|---|---|---|
| 30 | Tool Use & Function Calling | new | Small live model; ~80 line harness |
| 31 | Agent Loops & Harnesses | new | State-machine visualization |
| 32 | Multi-Agent & Swarm (OpenAI Swarm → KIMI K2.5) | new | Parallel vs sequential timing |
| 33 | Eval Frameworks (lm-eval, Inspect AI) | new | Run a real eval end-to-end |

Total new notebooks for V–VIII: **18** (including the 2 review-added modules
that belong in V and VI).

## Dependency graph (build order)

```
Part III nanoGPT ─┬─> 18 ──> 19 ──> 20 ──> 21
                  │
Part IV M13 KV ───┼─> 25 ──> 25.5 (SAE uses trained model)
                  │
                  └─> 26 ──> 28 ──> 29
                            
22 ──> 22.5            23 ──> 24           30 ──> 31 ──> 32 ──> 33
```

Modules with no cross-part dependency (18, 19, 22, 22.5, 23, 30, 31) can run
in parallel batches. The heavier-dependency modules (21, 25, 25.5, 26) should
run after their prerequisites.

## Proposed implementation batches

Given we had ~10 notebooks running in parallel in Parts I–IV with ~5–10 min
wall-clock per notebook, this is the recommended sequence:

**Batch A — independent, quick wins** (run in parallel):
- 18 Next-Token Prediction
- 19 Sampling Strategies
- 22 Prompt Engineering as Bayesian Conditioning
- 23 Reasoning Models & CoT
- 30 Tool Use & Function Calling
- 31 Agent Loops & Harnesses

**Batch B — alignment stack** (run in parallel):
- 20 RLHF & DPO
- 21 GRPO & RLVR (depends on 18; nanoGPT ready)
- 22.5 Alignment Stack & Red Teaming

**Batch C — reasoning & test-time** (run in parallel):
- 24 Test-Time Compute Scaling
- 25 Speculative Decoding + Medusa/EAGLE/Lookahead (depends on Part IV M13)
- 25.5 Sparse Autoencoders (depends on any trained model, can reuse nanoGPT)

**Batch D — serving** (run in parallel, mostly simulation):
- 26 vLLM & Serving Engines
- 27 LiteLLM & Model Routing
- 28 Continuous Batching
- 29 Distributed Inference

**Batch E — agents & eval** (run in parallel):
- 32 Multi-Agent & Swarm
- 33 Eval Frameworks

Wall-clock estimate: ~5 batches × 10 min each ≈ **50 min of agent wall-clock**
for all 18 notebooks, plus review of each batch. Realistically plan for a half-day
session including debugging rebuilds.

## Resource / infrastructure concerns

| Module | Concern | Mitigation |
|---|---|---|
| 21 GRPO | Real RL training is slow | Use a toy verifier task (digit multiplication); 500 steps max |
| 25 Medusa/EAGLE | Need working draft+target models | Fall back to simulation if model weights unavailable |
| 26 vLLM | vLLM needs GPU in practice | Simulate continuous batching + KV cache in numpy; note that vLLM is the real production system |
| 27 LiteLLM | Requires API keys | Use a free-tier provider or mock; emphasize cost tracking |
| 30/31/32 | Real agent loops need tool runtimes | Use local Python tools (calculator, search over a small corpus) |

## Open questions for the user before launching Batch A

1. **Real API access?** For Module 27 (LiteLLM) and Modules 30–32 (agents),
   do you have an API key you're comfortable using, or should the notebooks
   use mock providers?
2. **GPU availability?** Module 26 (vLLM) is genuinely hard to teach without
   a GPU demo. OK to fall back to simulation + a "try this on a real GPU" appendix?
3. **Review-integration pass first?** Before launching Batch A, do you want to
   first rebuild the affected Part I–IV notebooks to match the revised outline
   (e.g. patch Module 10 notebook to add the YaRN/LongRoPE sidebar)?
4. **Module order within a batch?** The batches above run in parallel; if
   you'd rather see them sequentially (to review each before the next
   launches), say so and I'll serialize.

## Success criteria per notebook

Every V–VIII notebook must:
- [ ] Load cleanly with nbformat (no JSON errors)
- [ ] Compile all code cells without syntax errors
- [ ] Contain at least one "break it on purpose" experiment
- [ ] Have LaTeX math with a plain-English symbol walkthrough before each equation
- [ ] Have at least 3 plotted figures (histogram / heatmap / curve / bar)
- [ ] Run end-to-end on CPU in < 5 minutes (no hidden cloud dependencies)
- [ ] Close with a 5-question checkpoint quiz + bridge to the next module
- [ ] Avoid AI-sounding formatting (no "In this notebook we will", no bullet lectures)

## Non-goals

- **No full rewrite of existing notebooks.** Parts I–IV stay as-is unless a
  specific review issue mandates a fix (we've already done Module 13 KV cache
  and Module 25 math; anything else is a nice-to-have).
- **No building of real vLLM clusters or multi-GPU training.** We simulate.
- **No producing a PDF/HTML static site** until all 33+ notebooks are done.

## Next concrete action

Unless the user answers the open questions differently, start with **Batch A**
(6 independent, fast notebooks) as a single parallel launch.
