# Technical Accuracy Review
## `LLM_Interactive_Curriculum_Outline.md` (April 2026 edition)

---

## Executive Summary

**Overall accuracy grade: B+ (solid, with a few concrete fixes needed).**

The outline is impressively up-to-date and most technical claims hold up under verification. The core math (attention, softmax scaling, DPO, GRPO, LoRA, quantization basics, sinusoidal PE, RoPE, cross-attention) is correct. Most 2025–2026 model/framework claims (DeepSeek-R1, DAPO, KIMI K2.5, TurboQuant, RotorQuant, SimPO, Densing Law, inspect-ai, vLLM PagedAttention, GQA, continuous batching) also check out.

There are, however, **two critical errors** that should be fixed before a learner touches them:

1. The Llama-2 70B KV-cache numerical example in Module 13 is wrong — it silently assumes MHA when Llama-2 70B actually uses GQA. The "~10 GB" figure is off by roughly 8×.
2. The speculative-decoding acceptance rule in Module 25 is stated as a deterministic inequality, when the correct rule in the Leviathan et al. paper is a randomized Bernoulli test. As written, the module would teach the wrong algorithm.

A handful of medium/minor issues (DAPO name expansion, inspect-ai tagline, slightly over-confident speedup formula, a few hand-wavy "60%+" style statistics) are listed below with corrections.

---

## Critical issues (must fix)

### C1. Llama-2 70B KV cache calculation is wrong (Module 13)

**Outline says:**
> Llama-2 70B, 4K context, fp16: KV cache = `2 × 80 × 8192 × 4096 × 2 bytes ≈ 10 GB`

**Problem:** The formula uses `d_model = 8192`, which is only correct for vanilla multi-head attention (64 heads × head_dim 128). Llama-2 70B uses **Grouped-Query Attention with only 8 KV heads**, so the KV projection is `n_kv_heads × head_dim = 8 × 128 = 1024`, not 8192. This is the whole point of GQA — it's the reason Llama-2 70B is servable at long context at all.

**Correct calculation (per-request, fp16, 4K context):**
`2 (K+V) × 80 layers × (8 kv_heads × 128 head_dim) × 4096 tokens × 2 bytes ≈ 1.34 GB`

That's ~8× smaller than the outline's number, and it happens to be *the entire pedagogical payoff* of Module 14b (GQA). Teaching the ~10 GB figure first and then claiming GQA gives "8× KV cache reduction" in the very next module double-counts the savings and confuses the story.

**Suggested fix (one option):** present two numbers side by side —
- Llama-2 70B *if it used MHA*: 2 × 80 × 8192 × 4096 × 2 ≈ 10.7 GB (hypothetical)
- Llama-2 70B as it actually ships (GQA, 8 KV heads): ≈ 1.34 GB
- Then use that contrast to motivate Module 14b.

**Severity: Critical.** Wrong by ~8×, and it leaks into the pedagogical arc of the next module.

---

### C2. Speculative decoding acceptance rule is incorrect (Module 25)

**Outline says:**
> Draft: `x₁...xₖ ~ q(x)`. Target: compute `p(xᵢ)` for all i in parallel. Accept xᵢ if `p(xᵢ) ≥ q(xᵢ)`, else reject and resample.

**Problem:** That is not the Leviathan, Kalman & Matias (2023) algorithm. The correct rule is a **randomized** accept/reject test, essentially rejection sampling:

- If `p(x) ≥ q(x)`, accept with probability 1.
- If `p(x) < q(x)`, accept with probability `p(x) / q(x)` (i.e. draw `u ~ U(0,1)` and accept iff `u ≤ p(x)/q(x)`).
- On rejection, resample from the "residual" distribution `max(0, p(x) − q(x))` normalized.

This randomization is exactly what makes the output distribution *provably identical* to sampling from `p` — which is the "surprise moment" the module promises. The deterministic "`p ≥ q`" rule as written would bias samples toward the target's modes and violate that identity. Teaching it this way means the "mathematically identical" claim on the next line is false.

**Severity: Critical.** The module currently contradicts its own correctness claim.

---

## Medium issues (should fix)

### M1. Speculative decoding speedup formula is oversimplified (Module 25)

**Outline says:** `Expected speedup: 1/(1−α) where α = acceptance rate`.

**Correction:** The Leviathan et al. analysis gives expected tokens per target call `τ = (1 − α^(γ+1)) / (1 − α)` where γ is the number of draft tokens; the actual wall-clock speedup is `τ / (γc + 1)` where `c` is the draft-to-target latency ratio. `1/(1−α)` is the limit as γ → ∞ with `c = 0` — a useful intuition pump, but not the formula. Either present it explicitly as a "best case" bound or give the full expression. In practice speedups of 2–3× (which the outline also quotes) correspond to α in roughly the 0.6–0.8 range with γ around 4–8, not `1/(1−0.6) = 2.5` the way the outline implies.

**Severity: Medium** (right intuition, wrong formula).

---

### M2. DAPO acronym expansion is wrong (Module 21)

**Outline says:** "DAPO (Distributed Advantage PO)".

**Correction:** DAPO = **Decoupled Clip and Dynamic sAmpling Policy Optimization** (ByteDance Seed + Tsinghua AIR, March 2025). Nothing "distributed" about it as an algorithm — the decoupled-clip and dynamic-sampling pieces are the whole point, and Clip-Higher (which the outline does mention) is one of the four named techniques. The 50-on-AIME'24 / Qwen2.5-32B result and the Clip-Higher / Dynamic Sampling / Overlong Reward Shaping list are otherwise accurate.

**Severity: Medium** (factual naming error on a trendy method).

---

### M3. "60%+ of open-source frontier models use MoE" is an unsupported statistic (Module 15)

The claim is plausible-sounding for 2025–2026 and directionally true (DeepSeek-V3, Llama-4 Maverick, Mixtral line, Qwen-MoE, KIMI K2/K2.5, etc.), but there is no widely-cited source that puts a specific number like "60%+" on it. For a curriculum that promises "no 'just trust me'", this one-liner should either (a) be softened to "most" / "the majority of" with an example list, or (b) cite a specific tracker (e.g. Artificial Analysis, HuggingFace Open LLM Leaderboard) and pin the date.

**Severity: Medium** (stylistic — but the outline's own design principle forbids unverifiable claims).

---

### M4. "Inspect AI (UK AISI / π-framework)" — the π-framework tag is wrong (Module 33)

Inspect AI is just "Inspect" from the UK AI Security Institute (UKGovernmentBEIS/inspect_ai). It is not known as the "π-framework" anywhere I can find. Drop that parenthetical or replace it with a short accurate description like "(UK AISI, Python; used for AISI's own automated evals)".

**Severity: Medium** (will confuse anyone who tries to search for it).

---

### M5. KV cache memory formula omits n_kv_heads (Module 13, general formula)

**Outline says:** `Memory: 2 × n_layers × d_model × seq_len × bytes_per_param`.

**Problem:** This is only correct for MHA. For GQA/MQA (which every modern model uses — the outline itself says so in Module 14b), the correct formula is:

`Memory = 2 × n_layers × n_kv_heads × head_dim × seq_len × bytes_per_param`

Given that the very next module (14) teaches GQA and uses it to motivate serving efficiency, this formula should be introduced in its GQA-general form from the start, with MHA presented as the `n_kv_heads = n_heads` special case. Otherwise the numerical example and the general formula reinforce each other's mistake.

**Severity: Medium** (same root cause as C1; fix them together).

---

## Minor issues (nice to fix)

### m1. SwiGLU FFN formula is slightly off (Module 9)

**Outline says:** `FFN(x) = W₂ · SwiGLU(W₁x) + b₂`.

SwiGLU FFN is actually a gated form: `FFN(x) = W₂ · (SiLU(W_gate x) ⊙ (W_up x))`, i.e. *three* weight matrices (gate, up, down), no biases in Llama-style models. Writing it as a single "`SwiGLU(W₁x)`" hides the gating, which is what distinguishes SwiGLU from vanilla GLU and makes FFN_dim ≈ 8/3 × d_model (not 4× d_model as in GPT-2). Minor but worth tightening given how often the module uses Llama-2 numbers.

### m2. Module 10 sinusoidal formula is missing the `cos` companion

Only `PE(pos, 2i) = sin(...)` is shown. The original Vaswani formulation pairs it with `PE(pos, 2i+1) = cos(pos / 10000^(2i/d))`. Not wrong, just incomplete — and the even/odd interleaving is useful intuition for how the Fourier features actually work.

### m3. GRPO "~50% memory reduction vs PPO" (Module 21)

Directionally correct (dropping the critic nets roughly halves the model-state memory during RL training), but the exact savings depend on whether PPO uses a separate value head vs a separate value model, on whether the reward model shares weights with the reference, and on optimizer state. Call it "removes the value network, typically cutting training memory by ~40–50%" or cite a specific paper.

### m4. KV cache formula doesn't say per-request (Module 13)

The `2 × n_layers × d_model × seq_len × bytes` figure is *per sequence in the batch*. Learners who are thinking about serving (which is the stated frame for Part IV) will ask "per request or for the whole batch?" — a one-word fix.

### m5. "T=0 greedy → repetitive" (Module 19)

Greedy decoding isn't always repetitive — for short-form QA it's often the right default. The "greedy → degenerate loops" pathology is specifically a symptom of *unconditional long-form generation* with small models. Phrase it more precisely to avoid teaching a bad heuristic.

### m6. "Decoder-only won the scaling race because... Chinchilla" (Module 11)

Chinchilla is about the compute-optimal N/D trade-off; it doesn't directly argue that decoder-only beats encoder-decoder. The actual reason decoder-only won is more about (a) unified pretraining objective, (b) KV-cacheable autoregressive inference, (c) in-context learning emerging cleanly from next-token prediction. Worth rephrasing.

### m7. CMoE attribution (Module 15)

"CMoE (Carved MoE)" as a named technique for carving MoE from dense models is a little ambiguous — there are several recent papers in this space (LLaMA-MoE, MoE-from-dense, etc.). Either pin it to a specific paper or generalize to "upcycling dense models into MoE without full retraining".

### m8. KIMI K2.5 context length (Module 32)

Minor factoid worth pinning: public model cards list KIMI K2.5 at 256K context. The outline doesn't claim a number but if you add one, use 256K.

### m9. Densing Law doubling time (Module 24, Bonus C)

Sources vary between 3.3 and 3.5 months for the density doubling. 3.5 is defensible (it's in the Nature MI write-up) but 3.3 is in the original arXiv. Either is fine; just pick one and cite it.

---

## Verified claims (confirmed correct)

These all check out under verification and can be taught as-is:

- **Dot-product attention, QKV decomposition, matrix form** (Module 5) — standard.
- **`Var(q·k) = dₖ` → √dₖ scaling** argument (Module 6) — correct, including the saturation and vanishing-gradient story.
- **Softmax Jacobian** `∂softmax/∂x = diag(s) − ssᵀ` (Module 6) — correct.
- **Causal masking via -∞** (Module 7) — correct.
- **Multi-head concat with Wᴼ** (Module 8) — correct.
- **Pre-norm residual block** (Module 9) — correct (modulo the SwiGLU nitpick in m1).
- **Sinusoidal PE, RoPE rotation identity `R_θm q · R_θn k = q · R_θ(n-m) k`, ALiBi linear bias** (Module 10) — all correct.
- **Llama-2 7B specs:** `d_model=4096, n_heads=32, n_layers=32, vocab=32000, FFN_dim=11008` (Module 12) — verified against HuggingFace model card.
- **Llama-2 70B structural specs:** 80 layers, d=8192, 64 heads, 8 KV heads, head_dim 128 (relevant to C1) — verified.
- **GQA as middle ground between MHA and MQA, Llama-2 70B using 8 KV groups** (Module 14b) — correct.
- **PagedAttention OS/virtual-memory analogy, block-size-B internal-fragmentation bound** (Module 14a) — correct, matches the vLLM paper.
- **GPTQ layer-wise objective, AWQ salient-channel protection** (Module 16a) — correct.
- **TurboQuant (Google, ICLR 2026):** random rotation + QJL, data-oblivious, 3-bit KV without retraining, introduces PolarQuant — all verified against Google Research blog and secondary coverage. The "3+1=4 bits" accounting is consistent with the published claim of 1-bit QJL overhead.
- **RotorQuant:** Cl(3,0) rotors, sandwich product `RxR̃`, ~100 mul-adds per vector for d=128, 10–19× CUDA / 9–31× Metal speedup, 5× KV compression, 44× fewer params — all match the scrya-com writeup and linked PRs. Note the outline is correctly reporting "reimagining of TurboQuant", not claiming it's a peer-reviewed paper.
- **LoRA `W' = W + BA`, rank-r decomposition, QLoRA 4-bit base + LoRA adapters, 65B on single GPU** (Module 17) — correct.
- **Next-token prediction loss, perplexity `PPL = e^L`** (Module 18) — correct.
- **Temperature, top-k, top-p (nucleus), min-p** definitions (Module 19) — correct.
- **PPO objective with KL-to-reference penalty, three-model memory footprint** (Module 20) — correct.
- **DPO loss** `−log σ(β(log(π(y_w|x)/π_ref(y_w|x)) − log(π(y_l|x)/π_ref(y_l|x))))` (Module 20) — matches Rafailov et al. 2023 exactly.
- **"DPO eliminates the reward model"** — correct; that is literally the paper's thesis ("Your language model is secretly a reward model").
- **SimPO beating DPO by up to 6.4 points on AlpacaEval 2** (Module 20) — verified (Meng et al., NeurIPS 2024; 6.4 is the Mistral-Base number, 4.4 for Llama-3-Instruct).
- **KTO works with binary feedback, ORPO merges SFT + preference** (Module 20) — correct.
- **GRPO group-relative advantage** `Aᵢ = (rᵢ − mean(r)) / (std(r) + ε)` **and clipped ratio objective** (Module 21) — correct, matches DeepSeekMath paper.
- **RLVR framing** (Module 21) — correct.
- **DeepSeek-R1-Zero: pure GRPO/RLVR, no human CoT annotations, emergent self-reflection / "aha moment"** (Module 21, 23) — correct per the R1 tech report / Nature paper (AIME pass@1 15.6% → 71.0%).
- **DAPO result: Qwen2.5-32B → 50 on AIME 2024** (Module 21) — correct (outperforms DeepSeek-R1-Zero-Qwen-32B's 47 with 50% training steps). The list of techniques (Clip-Higher, Dynamic Sampling, Overlong Reward Shaping) is correct. Only the acronym expansion is wrong (M2).
- **Test-time compute scaling, thinking-budget trade-off, diminishing returns** (Module 24) — correct framing.
- **Densing Law doubling every ~3.5 months** (Module 24, Bonus C) — defensible (see m9).
- **vLLM (PagedAttention + continuous batching), SGLang, TGI, TensorRT-LLM, llama.cpp/GGUF** descriptions (Module 26) — all correct.
- **LiteLLM unified API + routing + fallbacks + proxy** (Module 27) — correct.
- **Continuous batching wins 2–8×** (Module 28) — within accepted range.
- **Tensor parallel vs pipeline parallel with micro-batching to hide bubbles** (Module 29) — correct.
- **Inspect AI solvers/scorers structure** (Module 33) — correct (modulo the "π-framework" nickname in M4).
- **Benchmark contamination demo (MMLU rephrasing)** (Module 33) — well-documented phenomenon.
- **KIMI K2.5: 1T total / 32B active, MoE, up to 100 sub-agents, PARL, ~4.5× speedup, 78.4% BrowseComp** (Module 32) — all verified against Moonshot's release materials and independent coverage (datacamp, marktechpost, digitalapplied).
- **Flash Attention tiling to stay in SRAM, IO-complexity framing** (Bonus A) — correct.
- **Chinchilla scaling law form** (Bonus C) — correct.

---

## Recommended research topics for deeper fact-checking

Things I did not fully verify and which a subject-matter expert should give a second pass:

1. **Exact per-request KV-cache numbers for Llama-2 70B at 4K/32K/128K** — once C1/M5 are fixed, double-check the numerical example lands correctly and the contrast with MHA is pedagogically useful.
2. **TurboQuant bit-count accounting.** The outline says "3 + 1 = 4 bits per value". Google's blog post also quotes "3.5 bits per value" as the break-even-with-fp16 point, and the QJL overhead is described as "just 1 bit". It is worth spending 10 minutes with the paper to nail which exact figure to put on the slide, because the marketing numbers (3, 3.5, 4) drift by audience.
3. **RotorQuant PPL / attention-fidelity numbers** (6.91 vs 7.07 PPL, 99.0% attention fidelity, 5× compression). RotorQuant is a GitHub-released independent re-implementation rather than a peer-reviewed paper, so the numbers are "as reported by the authors". Worth flagging that to learners — it is legitimate to teach, but the provenance is different from TurboQuant.
4. **DAPO four techniques** — Clip-Higher, Dynamic Sampling, Token-Level Policy Gradient Loss, Overlong Reward Shaping. The outline lists three of the four; consider whether to include the fourth for completeness.
5. **KIMI K2.5 "1,500 coordinated tool calls per session" and "up to 100 sub-agents"** — these come from Moonshot's own release materials. Worth a citation link in the module so learners can check the source.
6. **"60%+ open-source frontier MoE" (M3)** — if you want to keep the number, you need a tracker link.
7. **Speculative decoding: the exact Leviathan et al. acceptance rule and the speedup formula derivation** — once C2/M1 are fixed, it would be worth a 20-minute sanity pass by someone who has implemented it, to make sure the interactive demo actually samples from the residual distribution correctly.
8. **GPT-5.2** is referenced in the KIMI K2.5 "beating GPT-5.2 on BrowseComp" line. Worth pinning whether that was the current frontier at K2.5's release and citing the specific BrowseComp leaderboard snapshot.
9. **Claude Agent SDK / OpenAI Agents SDK / Google ADK launch dates** in Module 31 — the outline gives specific months (March 2025, April 2025). These should each have a one-line citation since agent-SDK launches are rapidly shifting and outline freshness matters here.
10. **SwiGLU gated formulation and the 8/3 × d_model FFN dimension** (m1) — worth getting exactly right since Module 12 quotes specific Llama-2 FFN numbers.

---

## Bottom line

Fix C1 and C2 before anyone runs the notebooks. Fix M2 (DAPO name) and M4 (inspect-ai tag) before anyone searches for them. Everything else is polish. The outline is otherwise a genuinely current and technically sound sketch of an April 2026 LLM curriculum.

---

### Sources consulted

- [meta-llama/Llama-2-70b · Hugging Face](https://huggingface.co/meta-llama/Llama-2-70b)
- [LLaMA Architecture: Design Philosophy and Training Efficiency (Brenndoerfer)](https://mbrenndoerfer.com/writing/llama-architecture-design-training-efficiency)
- [Fast Inference from Transformers via Speculative Decoding (Leviathan et al., arXiv 2211.17192)](https://arxiv.org/abs/2211.17192)
- [Looking back at speculative decoding — Google Research blog](https://research.google/blog/looking-back-at-speculative-decoding/)
- [Speculative Decoding — LLM Inference Handbook (BentoML)](https://bentoml.com/llm/inference-optimization/speculative-decoding)
- [TurboQuant: Redefining AI efficiency with extreme compression — Google Research](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/)
- [An Illustrated Deep Dive into TurboQuant: PolarQuant, QJL, and KV Cache Compression](https://darshanfofadiya.com/research-papers/turboquant/)
- [scrya-com/rotorquant on GitHub](https://github.com/scrya-com/rotorquant)
- [RotorQuant — Clifford Algebra Vector Quantization (Scrya)](https://www.scrya.com/rotorquant/)
- [Moonshot AI Releases Kimi K2.5 — MarkTechPost (Jan 2026)](https://www.marktechpost.com/2026/01/27/moonshot-ai-releases-kimi-k2-5-an-open-source-visual-agentic-intelligence-model-with-native-swarm-execution/)
- [Kimi K2.5 Agent Swarm Guide — DataCamp](https://www.datacamp.com/tutorial/kimi-k2-agent-swarm-guide)
- [Kimi K2.5: Agent Swarm Architecture Complete Guide — Digital Applied](https://www.digitalapplied.com/blog/kimi-k2-5-agent-swarm-open-source-guide)
- [DAPO: An Open-Source LLM Reinforcement Learning System at Scale (arXiv 2503.14476)](https://ar5iv.labs.arxiv.org/html/2503.14476)
- [DAPO project on GitHub (BytedTsinghua-SIA)](https://github.com/BytedTsinghua-SIA/DAPO)
- [Direct Preference Optimization (Rafailov et al., arXiv 2305.18290)](https://arxiv.org/abs/2305.18290)
- [SimPO: Simple Preference Optimization with a Reference-Free Reward (arXiv 2405.14734)](https://arxiv.org/abs/2405.14734)
- [Densing Law of LLMs — Nature Machine Intelligence](https://www.nature.com/articles/s42256-025-01137-0)
- [Densing Law of LLMs (arXiv 2412.04315)](https://arxiv.org/abs/2412.04315)
- [DeepSeekMath / GRPO (arXiv 2402.03300)](https://arxiv.org/abs/2402.03300)
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL (arXiv 2501.12948)](https://arxiv.org/pdf/2501.12948)
- [Inspect AI — UK AI Security Institute](https://inspect.aisi.org.uk/)
- [UKGovernmentBEIS/inspect_ai on GitHub](https://github.com/UKGovernmentBEIS/inspect_ai)
