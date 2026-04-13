# Novelness Review: LLM Interactive Curriculum Outline

**Reviewer lens:** How current is this curriculum against the 2025-2026 LLM landscape? What's cutting-edge, what's missing, and what's outdated?
**Source document:** `/Users/jsmz97/Desktop/llm-into/LLM_Interactive_Curriculum_Outline.md` (dated April 2026)
**Review date:** April 11, 2026

---

## Executive Summary

**Overall grade: B+ / A-.** This curriculum is genuinely current in several places where most 2026 curricula are still running 2023 content: it covers GRPO/RLVR, TurboQuant/RotorQuant KV quantization, KIMI K2.5 swarm, test-time compute scaling, and modern agent SDKs. The post-training and serving-infra sections are particularly strong and would not look out of place in a research lab onboarding doc.

However, the curriculum has a **transformer-monoculture blind spot**. It treats decoder-only transformers as the only architecture worth teaching — a defensible bet in 2023, but in 2026 it misses Mamba-3, hybrid Jamba/Griffin models, and the diffusion-LM wave (LLaDA, Mercury). The biggest strategic misses are:

1. **Synthetic data is relegated to a bonus** despite being arguably the single most important trend of 2025-2026.
2. **Mechanistic interpretability / sparse autoencoders are entirely absent** — the field's most important interpretability story of the decade.
3. **Multimodal is a one-line bonus** when omni-modal (text/audio/video) natively-trained models are the frontier in Q1 2026.
4. **No post-DAPO coverage** (Dr-GRPO, NorMuon training, base-model RL) in the otherwise excellent Module 21.
5. **No coverage of EAGLE-3, Medusa, or lookahead decoding** — Module 25 stops at vanilla speculative decoding (2023 tech).
6. **Long-context extension techniques (YaRN, LongRoPE, LongRoPE2)** get zero treatment even though Module 10 covers RoPE.

Recommendation: add 4-6 new modules (listed at the end) and promote 2-3 bonuses to mainline. Keep the cutting-edge 2025-2026 topics that are already there — they're the curriculum's differentiator.

---

## 1. What's Covered Well (Genuinely Cutting-Edge)

These are topics the curriculum handles at or near the 2026 frontier. Most curricula in circulation don't even mention them.

| Topic | Module | Why it's current |
|---|---|---|
| **GRPO + RLVR** | 21 | Correctly frames GRPO as eliminating the value network, covers RLVR's verifier-based reward, name-drops DAPO with the right techniques (Clip-Higher, Dynamic Sampling, Overlong Reward Shaping). This is post-training content you'd expect from a DeepSeek/Moonshot engineer. |
| **TurboQuant / RotorQuant** | 16b, 16c | Extremely current. Random rotation + QJL + Clifford algebra for KV quantization is 2025-2026 frontier material. Most curricula stop at GPTQ/AWQ (2023). |
| **Test-time compute scaling** | 24 | Correctly separates inference-time vs train-time scaling as a new axis. Name-drops the "Densing Law." |
| **KIMI K2.5 Agent Swarm + PARL** | 32b | This is a January 2026 development. Including it with the credit-assignment nuance (serial collapse, staged reward shaping) shows the author tracks frontier research. |
| **Paged Attention / GQA / sliding window** | 14 | Correctly identifies KV cache as the modern memory bottleneck and covers all three algorithmic families. Good. |
| **Modern MoE landscape** | 15 | Name-checks DeepSeek-V3, Llama-4 Maverick, CMoE, KIMI K2.5. Accurate as of Q1 2026. |
| **DPO variants** | 20 | SimPO, KTO, ORPO all mentioned — reflects the actual 2025 preference-optimization zoo. |
| **Modern agent SDKs** | 31 | Anthropic Agent SDK, OpenAI Agents SDK (March 2025), Google ADK (April 2025), LangGraph, AutoGen v0.4 — this is an accurate 2025-2026 snapshot. |
| **Reasoning models (o1/R1/QwQ)** | 23 | Correctly distinguishes prompt-based vs training-based CoT. |

---

## 2. What's Missing Entirely (Ranked by Importance)

### Tier 1 — Critical omissions

#### 2.1 Sparse Autoencoders & Mechanistic Interpretability
**Zero coverage.** This is arguably the most significant interpretability breakthrough of the last three years. Anthropic's "Scaling Monosemanticity" (Claude 3 Sonnet SAEs, May 2024) and subsequent 2025 work discovered millions of interpretable features (Golden Gate Bridge feature, deception features, bias features). OpenAI followed with their own scaled SAE work. A curriculum that explains *how LLMs work* but ignores the primary tool for peering inside them is leaving a gaping hole.

- Key references: Anthropic "Scaling Monosemanticity" (transformer-circuits.pub 2024); "A Survey on Sparse Autoencoders" (arXiv:2503.05613).
- **Recommended new module:** "Module 9.5 or Bonus F: Opening the Black Box — Sparse Autoencoders and the Residual Stream." This naturally extends Module 9's "residual stream view."

#### 2.2 Synthetic Data (currently Bonus D)
The curriculum treats this as an appendix. In 2025-2026 it is arguably *the* dominant training story:
- Phi-4-Reasoning (Apr 2025): 14B model beats 70B+ models after distillation from DeepSeek-R1 traces.
- DeepSeek-R1 itself is a self-play synthetic data story (RLVR generates its own training signal).
- EMNLP 2025 systematic study ("Demystifying Synthetic Data in LLM Pre-training," arXiv:2510.01631) shows 1/3 synthetic + 2/3 web gives 5-10× speedup to reach equal validation loss.
- Magpie, Source2Synth, DataGen pipelines.

**Recommendation:** Promote to mainline as a full module in Part V (between modules 20 and 21), titled "Synthetic Data & Self-Improvement Loops." It directly motivates GRPO/RLVR in Module 21.

#### 2.3 Architecture Pluralism: Mamba / SSMs / Hybrids / Diffusion LMs
Module 11 ("Encoder vs Decoder vs Encoder-Decoder") frames the architectural landscape as three transformer variants and declares decoder-only the winner. That's a 2023 framing. As of 2026:
- **Mamba-3** (ICLR 2026, CMU/Princeton/Together/Cartesia): new Pareto frontier at 1.5B, MIMO variant for hardware efficiency.
- **Jamba** (AI21): Hybrid Transformer-Mamba MoE, 256K effective context, strong throughput.
- **Gated DeltaNet**, linear attention families are now competitive at small scale.
- **LLaDA** (Large Language Diffusion Models, Feb 2025): 8B diffusion LM competitive with LLaMA3 8B, +22% on Winograd (bidirectional advantage).
- **Mercury / Mercury Coder** (Inception Labs, June 2025): commercial-scale diffusion LM, 1109 tok/s — 5-10× faster than speed-optimized AR models.
- **RWKV-7**, **Hyena** — linear alternatives still alive, especially for long context.

**Recommendation:** Add a new Module 11.5 or Bonus F: "Beyond Transformers — SSMs, Hybrids, and Diffusion Language Models." Even as a single module it prevents the curriculum from reading as a transformer-only worldview.

#### 2.4 Long-Context Extension (YaRN, NTK-aware, LongRoPE/LongRoPE2)
Module 10 teaches RoPE but stops there. In 2026, every serious 128K+ context model uses YaRN, NTK-aware interpolation, or LongRoPE. Gemini 2.5 Pro runs 1M+ context. Grok-4-fast runs 2M. Research models now hit 10M tokens. The techniques to *get there* from a 4K-trained base are well-understood and teachable:
- YaRN (arXiv:2309.00071): 10× fewer tokens and 2.5× fewer steps than NTK-aware.
- LongRoPE (arXiv:2402.13753): 2048K tokens via evolutionary search over per-dimension rescaling.
- LongRoPE2: 128K with >98.5% of short-context accuracy preserved.

**Recommendation:** Extend Module 10 with a "10b: Length Extrapolation" sub-section, or add Bonus G: "How to Reach 1M Context."

#### 2.5 Modern Speculative Decoding (EAGLE-3, Medusa, Lookahead)
Module 25 covers vanilla speculative decoding — circa 2023 material. The 2024-2025 landscape has moved decisively:
- **Medusa**: parallel extra heads predict k tokens.
- **EAGLE-1/2/3** (ICML'24 / EMNLP'24 / NeurIPS'25): EAGLE-3 removes feature-prediction constraint, fuses low/mid/high features → **2-6× speedups**, production-ready.
- **Lookahead decoding**: two parallel branches in the same model (n-gram generation + verification).
- **Hydra, PEARL, Mirror Speculative Decoding**: more variants.

**Recommendation:** Expand Module 25 into 25a (baseline spec decoding), 25b (Medusa/Hydra/EAGLE family), 25c (lookahead). This is core serving content — EAGLE-3 is shipping in TensorRT-LLM and vLLM.

### Tier 2 — Important but second priority

#### 2.6 Mainline Multimodal (Omni Models)
Currently one line in Bonus E naming Gemini 2.5, Qwen3-VL, KIMI-VL, UI-TARS. But in 2026, **natively multimodal** is the default for frontier models:
- **Gemini 3.1 Flash Live** (Mar 2026): real-time A2A audio model, "collapses the voice-AI stack."
- **Qwen3.5-Omni** (Mar 2026): unified text/image/audio/video in one pipeline, Thinker-Talker architecture.
- **Gemini Embedding 2**: natively multimodal embeddings over Text/Image/Video/Audio/PDF.

Teaching LLMs as text-only in 2026 is like teaching computers as punch-card machines. Minimum, a text-only curriculum should explain how vision encoders get stitched in (CLIP → ViT → cross-attention vs token interleaving) and what "native" multimodal training means.

**Recommendation:** Promote Bonus E to a full mainline Module 34 or add a Part IX. Include vision tokenization (patchify + linear projection), audio tokenization (HuBERT, Whisper encoder, codec tokens), and late-fusion vs early-fusion architectures.

#### 2.7 Alignment & Safety Beyond DPO
The curriculum covers RLHF/DPO and GRPO. Missing entirely:
- **Constitutional AI / RLAIF** (the Anthropic lineage). Claude 4/4.5 ships with "character training" derived from CAI; 2025 work includes IterAlign (automated red-team + constitution proposal + revision loops).
- **Red teaming methodology** (automated + human).
- **Jailbreaks and adversarial robustness** (prompt injection, many-shot jailbreaks, universal adversarial suffixes).
- **Anthropic's 2025 "Alignment Science" reports** on misalignment risk.
- **Superego agents / Creed Constitutions** — 2025-2026 personalized alignment architecture.

**Recommendation:** Add Module 20b or new Module 22.5: "Constitutional AI, Red Teaming, and the Alignment Stack." Without this, the curriculum implicitly defines "alignment = RLHF/DPO/GRPO," which is incomplete.

#### 2.8 Modern Optimizers (Muon, SOAP, Shampoo)
Module 3 covers backprop but the curriculum never revisits optimizer choice. In 2025-2026:
- **Muon** (Keller Jordan 2024): orthogonalizes matrix updates via Newton-Schulz. Used in production by **Kimi K2, GLM-4.5, INTELLECT-3**. Set GPT-2 speedrun records.
- **SOAP** (Oct 2024): Shampoo-on-Adam. Connection: SOAP and Shampoo reduce to Muon under simplifying assumptions.
- **NorMuon** (Oct 2025): more efficient Muon.

This is directly relevant to any student running pretraining experiments, and the math is tractable (just Newton-Schulz iteration for matrix orthogonalization).

**Recommendation:** Add as a sub-section in a training-focused module or as Bonus H: "Optimizers for LLMs — Why AdamW Is Not the Last Word."

#### 2.9 Modern Evaluation Benchmarks
Module 33 lists lm-eval-harness, HELM, and older benchmarks. Missing:
- **SWE-bench / SWE-bench Verified** — the premier agent/code benchmark in 2025-2026 (AI went from 4.4% in 2023 to 71%+ by 2024).
- **GPQA Diamond** — PhD-level science.
- **FrontierMath** — undergraduate-through-research math, human-graded.
- **HLE (Humanity's Last Exam)** — 2500 hard multi-modal questions, 1000+ expert contributors.
- **LiveBench** — contamination-limited, frequently updated.
- **BrowseComp** (already referenced in Module 32b for KIMI — good!).
- **Chatbot Arena / Elo** — already mentioned but could use a discussion of its biases.

**Recommendation:** Expand Module 33a explicitly to name these benchmarks and discuss contamination, saturation, and how "hard benchmarks" have a half-life of ~18 months now.

#### 2.10 Edge / On-Device Deployment
The curriculum mentions GGUF, llama-cpp in the tech stack but has no module on edge deployment. In 2026 this is a major practitioner track:
- **llama.cpp + GGUF** as the cross-platform CPU/Metal standard.
- **Apple MLX** (and MLX Swift) for Apple Silicon.
- **Core ML** + coremltools for on-device iOS.
- **1-bit quantization** (PrismML Bonsai 8B, BitNet-derived) now commercially viable.
- **Liquid AI LFM2.5** and other purpose-built on-device models.
- **Phone-class models**: Gemini Nano, Apple Intelligence on-device.

**Recommendation:** Add Module 29.5 or Bonus I: "On-Device LLMs — llama.cpp, MLX, Core ML, and the 1-bit Frontier." Fits naturally after distributed inference (Module 29).

### Tier 3 — Worth mentioning

#### 2.11 Open-Source Frontier Model Landscape
Various modules name-drop DeepSeek, Llama-4, Qwen, KIMI — but there's no module that gives students a map of the open-source ecosystem: licensing, architecture family, strengths, deployment options. By 2026, **five open families are at frontier**: DeepSeek, Qwen, Kimi, GLM, Mistral. A short "landscape" module (or appendix) would help.

#### 2.12 Pretraining Data Pipelines
The curriculum skips pretraining data curation entirely. Data > architecture is a 2024-2026 consensus. At minimum, a bonus on: FineWeb/FineWeb-Edu, DCLM, Nemotron-CC, deduplication, quality filters, contamination scrubbing.

#### 2.13 Tokenizer-Free & Byte-Level
Module 1 teaches BPE but doesn't touch byte-level (ByT5) or tokenizer-free approaches (MambaByte, Meta's BLT — Byte Latent Transformer, Dec 2024). BLT is particularly notable: dynamic patching based on entropy beats BPE tokenizers on compute-matched settings. Worth a paragraph.

---

## 3. What's Present But Outdated / Needs Updating

| Module | Issue | Fix |
|---|---|---|
| **Module 11: Encoder vs Decoder** | Declares decoder-only "won the scaling race" as final answer. In 2026 with SSM/hybrid/diffusion comeback, this framing is stale. | Add a caveat: "Decoder-only won 2020-2024; 2025-2026 is seeing renewed architectural diversity — see Module 11.5." |
| **Module 20: RLHF & DPO** | Treats these as "classics." Good framing, but the mention of SimPO/KTO/ORPO is one line. | Expand to give each a 2-sentence treatment with the intuition of how they differ. |
| **Module 21: GRPO** | DAPO mentioned but only with its original four techniques. Missing: Dr-GRPO, base-model RL (no SFT warmup), data curation findings from late-2025. | Add a "Post-DAPO" paragraph citing Dr-GRPO and RLVR domain expansion (chemistry, biology, etc.). |
| **Module 25: Speculative Decoding** | Stops at vanilla draft-verify. Explicitly mentions "typical: 2-3× speedup" — but EAGLE-3 gets 2-6× in production. | See section 2.5. |
| **Module 10: Positional Encoding** | RoPE/ALiBi/sinusoidal coverage good, but nothing on extension. | See section 2.4. |
| **Module 16: Quantization** | Great coverage of TurboQuant/RotorQuant but oddly silent on 1-bit quantization (BitNet, Bonsai) and integer-only inference on edge hardware. | Add a "Bonus: The 1-bit Frontier" sub-section. |
| **Module 33: Eval** | Missing modern benchmarks (see 2.9). Mention of "LLM-as-judge" is there but no discussion of judge bias, position bias, or the 2025 findings on judge reliability. | Add a note on judge limitations. |
| **Bonus E: Multimodal** | One-line mentions. Should be mainline. | See section 2.6. |
| **Bonus D: Synthetic Data** | One-line mention. Should be mainline. | See section 2.2. |

---

## 4. Specific Papers / Developments to Incorporate

Grouped by theme, with arXiv IDs or primary references.

**Architecture**
- Mamba-3 (ICLR 2026, openreview HwCvaJOiCj)
- Jamba hybrid Transformer-Mamba-MoE (openreview JFPaD7lpBD)
- LLaDA: Large Language Diffusion Models (arXiv:2502.09992, Feb 2025)
- Mercury: diffusion LM (arXiv:2506.17298, June 2025)
- Byte Latent Transformer (Meta, Dec 2024) — tokenizer-free
- Gated DeltaNet, RWKV-7, Hyena

**Post-training / RL**
- DAPO (arXiv:2503.14476, ByteDance Seed, Mar 2025)
- Dr-GRPO (mentioned in Red Hat / Interconnects 2025 surveys)
- "Post-Training in 2026: GRPO, DAPO, RLVR & Beyond" (llm-stats.com blog)
- IterAlign — automated red-team + constitution refinement

**Long context**
- YaRN (arXiv:2309.00071)
- LongRoPE (arXiv:2402.13753)
- LongRoPE2 (OpenReview jwMjzGpzi4)

**Inference**
- EAGLE-1 (ICML'24), EAGLE-2 (EMNLP'24), EAGLE-3 (NeurIPS'25) — github.com/SafeAILab/EAGLE
- Medusa, Hydra
- Lookahead decoding
- Mirror Speculative Decoding (arXiv:2510.13161)

**Interpretability**
- Anthropic "Scaling Monosemanticity" (transformer-circuits.pub, 2024)
- "A Survey on Sparse Autoencoders" (arXiv:2503.05613)
- "Use SAEs to Discover Unknown Concepts" (arXiv:2506.23845)

**Synthetic data**
- "Demystifying Synthetic Data in LLM Pre-training" (arXiv:2510.01631, EMNLP 2025)
- Phi-4-Reasoning (Microsoft, Apr 2025)
- Self-Play Fine-Tuning (arXiv:2401.01335)

**Optimizers**
- Muon (Keller Jordan, kellerjordan.github.io/posts/muon/)
- "Practical Efficiency of Muon for Pretraining" (arXiv:2505.02222, Essential AI)
- NorMuon (arXiv:2510.05491)
- SOAP (Vyas et al., 2024)

**Multimodal**
- Gemini 3.1 Flash Live (Mar 2026)
- Qwen3.5-Omni (Mar 2026)
- Gemini Embedding 2 (Mar 2026)

**Evaluation**
- SWE-bench, SWE-bench Verified (swebench.com)
- FrontierMath, GPQA Diamond, HLE, LiveBench

**Edge**
- Apple MLX / MLX Swift
- Liquid AI LFM2.5
- PrismML Bonsai 8B (1-bit)
- "On-Device LLMs: State of the Union, 2026" (v-chandra.github.io)

---

## 5. Recommended New Modules / Expansions

Here's a concrete proposal — 5 new mainline modules and 2 promoted bonuses, plus small in-place additions.

### New mainline modules

1. **Module 11.5: Beyond Transformers** (after Module 11)
   State space models, Mamba-3, Jamba hybrids, diffusion LMs (LLaDA, Mercury). Interactive demo: train a tiny Mamba block vs a tiny attention block on the same data, plot compute/memory vs seq length.

2. **Module 17.5: Synthetic Data & Self-Improvement** (between 17 and 18, or between 20 and 21)
   Covers Phi pipeline, Magpie, distillation from reasoning traces, the EMNLP 2025 scaling-law findings, teacher-student loops. Interactive demo: generate 100 synthetic Q/A pairs from a small model and fine-tune on them.

3. **Module 22.5: Alignment Stack & Red Teaming** (after Module 22)
   Constitutional AI, RLAIF, IterAlign, jailbreak taxonomy, Anthropic's alignment science reports. Interactive demo: attempt several jailbreak techniques on a safety-tuned open model and measure refusal rates.

4. **Module 25.5: Sparse Autoencoders & Mechanistic Interpretability** (after 25, or as a Part VI addendum)
   Train a tiny SAE on GPT-2 residual stream activations. Surface interpretable features. Show the "residual stream view" from Module 9 in action.

5. **Module 34: Native Multimodal Models** (promote Bonus E, expand)
   Vision tokenization, audio tokenization, native omni architectures (Gemini Flash Live, Qwen3.5-Omni), Thinker-Talker designs. Interactive demo: stitch a ViT into a tiny LM and train joint.

### Expansions of existing modules

- **Module 10** → add 10b on YaRN/LongRoPE length extrapolation.
- **Module 21** → add post-DAPO paragraph (Dr-GRPO, base-model RL, RLVR domain expansion).
- **Module 25** → split into 25a (baseline), 25b (Medusa/EAGLE/Hydra), 25c (lookahead).
- **Module 16** → add 16d on 1-bit quantization (BitNet, Bonsai).
- **Module 29** → add a 29b on edge / on-device (MLX, Core ML, llama.cpp on phones).
- **Module 33** → refresh benchmark list (SWE-bench, FrontierMath, HLE, LiveBench, GPQA Diamond), add judge bias discussion.

### New bonus modules

- **Bonus F: Optimizers Beyond AdamW** — Muon, SOAP, Shampoo, when and why.
- **Bonus G: Pretraining Data Pipelines** — FineWeb-Edu, DCLM, dedup, contamination scrubbing.
- **Bonus H: The Open-Source Frontier Landscape** — licensing, model families, deployment tradeoffs for DeepSeek / Qwen / Kimi / GLM / Mistral / Llama.

---

## 6. Trend Forecast: What's Likely to Change by End of 2026

These are bets — use them to plan curriculum refresh cadence.

**High confidence by end of 2026:**
1. **Hybrid architectures become default at the frontier.** Pure-transformer and pure-SSM both lose to hybrids like Jamba successors at 70B+. Module 11.5 will need updating every 6 months.
2. **EAGLE-4 or successor** — speculative decoding continues compounding 1.5-2× per year. Module 25 should be designed to accommodate a new technique per refresh.
3. **1M+ context becomes standard** for all frontier models. Long-context extension becomes a "required skill" rather than a specialty.
4. **Agentic RL (RLVR in tool-use settings)** becomes the dominant post-training paradigm. Expect RLVR-for-agents papers in every major conference. Module 21 will need an "agentic RL" sub-section.
5. **Synthetic data dominates pretraining mixes.** The 2025 EMNLP finding (1/3 synthetic optimal) will become 1/2 by end of 2026.
6. **Native omni becomes the default** for frontier proprietary models. Text-only is legacy.
7. **On-device 1-bit models become production-ready** on phones. Apple Intelligence, Google Gemini Nano, and third-party 1-bit models hit the mainstream.

**Medium confidence:**
8. **Diffusion LMs break into production** for latency-critical workloads (Mercury-style). Not a transformer replacement, but a significant niche.
9. **Mechanistic interpretability influences safety testing.** Regulatory frameworks start citing feature-level audits.
10. **Evaluation crisis deepens.** HLE and FrontierMath will be saturated by late 2026. New "harder than humans" benchmarks will emerge.

**Lower confidence / watch list:**
11. **Muon-family optimizers become new default** for pretraining (currently used by Kimi K2, GLM-4.5, INTELLECT-3 — tipping point near).
12. **GRPO is replaced.** The post-DAPO successor may already exist; current leads are Dr-GRPO and related normalization variants.
13. **Reasoning compute hits a wall** at some point — diminishing returns beyond ~100K thinking tokens may reshape how Module 24 frames test-time compute.

**Curriculum refresh cadence recommendation:** This curriculum will need meaningful updates every 6 months to stay current. The architecture-agnostic parts (I-III, foundations) will age slowly. Parts IV-VIII should be treated as quarterly-refresh material.

---

## Sources

- [Mamba-3: Improved Sequence Modeling using State Space Principles (ICLR 2026)](https://openreview.net/pdf?id=HwCvaJOiCj)
- [Meet Mamba-3: A New State Space Model Frontier (MarkTechPost, Mar 2026)](https://www.marktechpost.com/2026/03/18/meet-mamba-3-a-new-state-space-model-frontier-with-2x-smaller-states-and-enhanced-mimo-decoding-hardware-efficiency/)
- [Jamba: Hybrid Transformer-Mamba Language Models](https://openreview.net/forum?id=JFPaD7lpBD)
- [Attention was never enough: Tracing the rise of hybrid LLMs (AI21)](https://www.ai21.com/blog/rise-of-hybrid-llms/)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476)
- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [Post-training methods for language models (Red Hat, Nov 2025)](https://developers.redhat.com/articles/2025/11/04/post-training-methods-language-models)
- [Recent reasoning research: GRPO tweaks, base model RL (Interconnects)](https://www.interconnects.ai/p/papers-im-reading-base-model-rl-grpo)
- [YaRN: Efficient Context Window Extension](https://arxiv.org/abs/2309.00071)
- [LongRoPE: Extending LLM Context Window Beyond 2M Tokens](https://arxiv.org/abs/2402.13753)
- [LongRoPE2: Near-Lossless LLM Context Window Scaling](https://openreview.net/forum?id=jwMjzGpzi4)
- [How LLMs Scaled from 512 to 2M Context (Technical Deep Dive)](https://amaarora.github.io/posts/2025-09-21-rope-context-extension.html)
- [EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/pdf/2401.15077)
- [EAGLE-1/2/3 Official Implementation](https://github.com/SafeAILab/EAGLE)
- [EAGLE-3 Speculative Decoding Guide](https://www.e2enetworks.com/blog/Accelerating_LLM_Inference_with_EAGLE)
- [Mirror Speculative Decoding](https://arxiv.org/html/2510.13161v1)
- [A Survey on Sparse Autoencoders](https://arxiv.org/html/2503.05613v3)
- [Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet](https://transformer-circuits.pub/2024/scaling-monosemanticity/)
- [LLM Interpretability and Sparse Autoencoders (Arize)](https://arize.com/blog/llm-interpretability-and-sparse-autoencoders-openai-anthropic/)
- [Demystifying Synthetic Data in LLM Pre-training (arXiv:2510.01631)](https://arxiv.org/abs/2510.01631)
- [Synthetic Pretraining (Vintage Data)](https://vintagedata.org/blog/posts/synthetic-pretraining)
- [Self-Play Fine-Tuning (arXiv:2401.01335)](https://arxiv.org/pdf/2401.01335)
- [Large Language Diffusion Models (LLaDA)](https://arxiv.org/abs/2502.09992)
- [Mercury: Ultra-Fast Language Models Based on Diffusion](https://arxiv.org/abs/2506.17298)
- [Introducing Mercury (Inception Labs)](https://www.inceptionlabs.ai/blog/introducing-mercury)
- [Muon optimizer blog (Keller Jordan)](https://kellerjordan.github.io/posts/muon/)
- [Practical Efficiency of Muon for Pretraining](https://arxiv.org/pdf/2505.02222)
- [NorMuon: Making Muon more efficient and scalable](https://arxiv.org/html/2510.05491v1)
- [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/pdf/2212.08073)
- [Alignment Science Blog (Anthropic)](https://alignment.anthropic.com/)
- [Aikipedia: Constitutional AI (2025)](https://champaignmagazine.com/2025/10/31/aikipedia-constitutional-ai/)
- [Google Gemini 3.1 Flash Live (Mar 2026)](https://www.marktechpost.com/2026/03/26/google-releases-gemini-3-1-flash-live-a-real-time-multimodal-voice-model-for-low-latency-audio-video-and-tool-use-for-ai-agents/)
- [Alibaba Qwen3.5 Omni (Mar 2026)](https://www.marktechpost.com/2026/03/30/alibaba-qwen-team-releases-qwen3-5-omni-a-native-multimodal-model-for-text-audio-video-and-realtime-interaction/)
- [Gemini Embedding 2 (Mar 2026)](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/)
- [10 Best Open-Source LLM Models 2025 (HuggingFace)](https://huggingface.co/blog/daya-shankar/open-source-llms)
- [Best Open Source LLMs 2026 (DeployBase)](https://deploybase.ai/articles/best-open-source-llm)
- [AI Model Benchmarks April 2026 (LM Council)](https://lmcouncil.ai/benchmarks)
- [SWE-Bench Verified Leaderboard](https://llm-stats.com/benchmarks/swe-bench-verified)
- [Epoch AI Benchmarks](https://epoch.ai/benchmarks/)
- [2025 AI Index Report (Stanford HAI)](https://hai.stanford.edu/ai-index/2025-ai-index-report/technical-performance)
- [On-Device LLMs: State of the Union 2026](https://v-chandra.github.io/on-device-llms/)
- [On-Device LLMs in 2026 (Edge AI Vision Alliance)](https://www.edge-ai-vision.com/2026/01/on-device-llms-in-2026-what-changed-what-matters-whats-next/)
- [Introducing LFM2.5 (Liquid AI)](https://www.liquid.ai/blog/introducing-lfm2-5-the-next-generation-of-on-device-ai)
- [Inside Transformers: Emerging Alternatives in 2025](https://www.gocodeo.com/post/inside-transformers-attention-scaling-tricks-emerging-alternatives-in-2025)
- [The End of Transformers? Sub-Quadratic Architectures](https://arxiv.org/html/2510.05364v1)
- [State of LLMs 2025 (Sebastian Raschka)](https://magazine.sebastianraschka.com/p/state-of-llms-2025)
