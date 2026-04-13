# Interactive Introduction to LLMs
### A Visual, Code-First Curriculum for the Technically Curious
#### Last updated: April 2026

---

## Philosophy

Each module follows the same rhythm:

1. **Intuition first** — a real-world analogy or visual diagram
2. **The math** — one or two key equations, rendered in LaTeX, explained symbol-by-symbol
3. **Interactive code block** — a runnable Python cell that lets you *poke* the concept (change inputs, visualize outputs)
4. **Checkpoint** — a mini-challenge or quiz to test understanding

Target: smart generalists (engineers, designers, PMs, researchers from other fields) who want to understand *how* LLMs work — not just *that* they work.

---

## Curriculum Map

```
PART I: FOUNDATIONS              PART II: ATTENTION DEEP DIVE        PART III: THE TRANSFORMER
┌──────────────────────┐        ┌──────────────────────────────┐    ┌───────────────────────────┐
│ 1. Text → Numbers    │───────▶│ 5. Dot-Product Attention     │───▶│ 9.  The Transformer Block │
│ 2. Embeddings        │───────▶│ 6. Why √dₖ? (The Math)      │───▶│ 10. Positional Encoding   │
│ 3. Neural Net 101    │───────▶│ 7. Causal Masking            │    │ 11. Encoder vs Decoder    │
│ 4. Autoencoder       │        │ 8. Multi-Head Attention      │    │ 12. Full Architecture     │
└──────────────────────┘        └──────────────────────────────┘    └───────────────────────────┘

PART IV: SCALING & EFFICIENCY          PART V: GENERATION & ALIGNMENT
┌────────────────────────────────┐     ┌──────────────────────────────────┐
│ 13. KV Cache                   │     │ 18. Next-Token Prediction        │
│ 14. KV Cache Optimization:     │     │ 19. Sampling Strategies          │
│     PagedAttn, GQA, MQA        │     │ 20. RLHF & DPO (The Classics)   │
│ 15. Mixture of Experts (MoE)   │     │ 21. GRPO & RLVR (The New Wave)  │
│ 16. Quantization: GPTQ → AWQ   │     │ 22. Prompt Eng. as Bayes        │
│     → TurboQuant → RotorQuant  │     │                                  │
│ 17. LoRA & QLoRA               │     │                                  │
└────────────────────────────────┘     └──────────────────────────────────┘

PART VI: REASONING & TEST-TIME COMPUTE  PART VII: SERVING & INFRA
┌──────────────────────────────────┐    ┌──────────────────────────────┐
│ 23. Reasoning Models & CoT       │    │ 26. vLLM & Serving Engines   │
│ 24. Test-Time Compute Scaling    │    │ 27. LiteLLM & Model Routing  │
│ 25. Speculative Decoding         │    │ 28. Batching & Scheduling    │
│                                  │    │ 29. Distributed Inference    │
└──────────────────────────────────┘    └──────────────────────────────┘

PART VIII: AGENTS & EVAL
┌──────────────────────────────────┐
│ 30. Tool Use & Function Calling  │
│ 31. Agent Loops & Harnesses      │
│ 32. Multi-Agent & Swarm:         │
│     OpenAI Swarm → KIMI K2.5    │
│ 33. Eval Frameworks              │
└──────────────────────────────────┘
```

---

## PART I — Foundations

### Module 1: Text → Numbers (Tokenization)

| | |
|---|---|
| **Core idea** | Computers don't read words — they read integer IDs. Tokenizers split text into sub-word chunks and map them to numbers. |
| **Key math** | Byte-Pair Encoding (BPE): iteratively merge the most frequent adjacent pair. Frequency function `f(pair)`, vocabulary `V`. |
| **Interactive demo** | Load `tiktoken`, tokenize user-typed sentences, visualize token boundaries with color-coded spans, show vocab size trade-offs. |
| **Surprise moment** | "The word 'unhappiness' becomes 3 tokens. The word 'cat' is 1. Why?" |

### Module 2: Embeddings — Meaning as Geometry

| | |
|---|---|
| **Core idea** | Each token becomes a dense vector. Similar meanings = nearby vectors. |
| **Key math** | Embedding lookup: `E(token_id) → ℝ^d`. Cosine similarity: `cos(a,b) = (a·b) / (‖a‖·‖b‖)` |
| **Interactive demo** | Embed a set of words, project to 2D with PCA/t-SNE, drag words around to build intuition. Show king − man + woman ≈ queen. |
| **Surprise moment** | Plotting country-capital pairs reveals parallel geometric structure. |

### Module 3: Neural Networks 101

| | |
|---|---|
| **Core idea** | A neural net is a stack of matrix multiplications + nonlinearities. It learns by adjusting weights to minimize error. |
| **Key math** | Forward pass: `y = σ(Wx + b)`. Loss: `L = -Σ yᵢ log(ŷᵢ)`. Backprop: `∂L/∂W` via chain rule. |
| **Interactive demo** | Tiny 2-layer net in pure NumPy. Train it on XOR. Visualize decision boundary evolving in real time. |
| **Surprise moment** | Without the nonlinearity (ReLU/sigmoid), the net can only draw straight lines. Toggle it on/off. |

### Module 4: Autoencoders — Compress and Reconstruct

| | |
|---|---|
| **Core idea** | Squeeze data through a bottleneck, then reconstruct it. The bottleneck forces the model to learn what matters. |
| **Key math** | Encoder: `z = f(x)`, Decoder: `x̂ = g(z)`, Loss: `‖x − x̂‖²`. Variational: add `KL(q(z|x) ‖ p(z))`. |
| **Interactive demo** | Train a tiny autoencoder on MNIST digits. Slide through the latent space and watch digits morph. |
| **Surprise moment** | The "bottleneck" concept is *everywhere* in ML — this is the mental model for understanding why transformers compress so well. |

---

## PART II — Attention Deep Dive

### Module 5: Dot-Product Attention — The Core Mechanism

| | |
|---|---|
| **Core idea** | Each token asks every other token: "How relevant are you to me?" via a dot product, then creates a weighted mix. |
| **Key math** | **Step-by-step derivation:** (1) Compute similarity: `eᵢⱼ = qᵢ · kⱼ`. (2) Normalize: `αᵢⱼ = softmax(eᵢⱼ)`. (3) Weighted sum: `outᵢ = Σⱼ αᵢⱼ · vⱼ`. Matrix form: `Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V`. |
| **Math deep dive** | Why Q, K, V? Analogy to database retrieval: Q = "what am I looking for?", K = "what do I contain?", V = "what do I offer?". Without separate projections, the model can't independently control *what to attend to* vs *what to extract*. |
| **Interactive demo** | Type a sentence → build Q, K, V matrices by hand from random projections → compute attention step-by-step → visualize the heatmap. Modify Q/K/V weights and watch attention shift. |
| **Surprise moment** | Attention without V projections just gives you a weighted average of the *same* input — V lets the model extract *different information* than what it searched for. |

### Module 6: Why √dₖ? — Softmax, Gradients, and Stability

| | |
|---|---|
| **Core idea** | Without scaling, dot products grow with dimension, softmax saturates, and gradients vanish. √dₖ keeps things learnable. |
| **Key math** | If `q, k ~ N(0, 1)` and `dₖ`-dimensional, then `E[q·k] = 0` but `Var(q·k) = dₖ`. So `q·k / √dₖ` has unit variance. Show: `softmax([10, 1, 1]) ≈ [0.9998, 0.0001, 0.0001]` (one-hot!) vs `softmax([1.5, 0.15, 0.15]) ≈ [0.56, 0.22, 0.22]` (smooth). |
| **Gradient analysis** | `∂softmax/∂x = diag(s) - ssᵀ`. When softmax is saturated, `sᵢ ≈ 1` for one entry → gradient ≈ 0 everywhere. The model can't learn. |
| **Interactive demo** | Slider for `dₖ` from 1 to 512. Plot: (1) histogram of raw dot products, (2) softmax output distribution, (3) gradient magnitude. Watch gradients vanish as dₖ grows without scaling. |
| **Surprise moment** | Remove the √dₖ on a real model and watch training loss plateau. Add it back → loss drops. A single division saves everything. |

### Module 7: Causal Masking — "No Peeking at the Future"

| | |
|---|---|
| **Core idea** | In autoregressive generation, token 5 must not attend to tokens 6, 7, 8... Causal masking enforces this by setting future scores to -∞ before softmax. |
| **Key math** | `Mask(i,j) = 0 if j ≤ i, else -∞`. Applied: `Attention = softmax((QKᵀ / √dₖ) + Mask) · V`. The -∞ → `e^(-∞) = 0` in softmax, so future tokens contribute zero weight. |
| **Encoder vs decoder** | Encoders (BERT): full bidirectional attention, no mask. Decoders (GPT): causal mask. Encoder-decoders (T5): encoder is bidirectional, decoder is causal + cross-attends to encoder. |
| **Interactive demo** | Visualize the attention matrix as a heatmap. Toggle causal mask on/off. See how BERT's attention is a full square, while GPT's is a lower triangle. Show what happens if a decoder peeks at the future (loss collapses during training, but generation breaks). |
| **Surprise moment** | Prefix-LM (PaLM-style): the prompt is bidirectional, only the *generation* portion is causal. Show the hybrid mask shape. |

### Module 8: Multi-Head Attention — Parallel Perspectives

| | |
|---|---|
| **Core idea** | One attention head captures one type of relationship. Multiple heads capture syntax, semantics, coreference, etc. in parallel. |
| **Key math** | `MultiHead(Q,K,V) = Concat(head₁, ..., headₕ) Wᴼ` where `headᵢ = Attention(QWᵢQ, KWᵢK, VWᵢV)`. Per-head dimension: `dₖ = d_model / h`. Total compute stays the same. |
| **Interactive demo** | Visualize what each head in a pre-trained model "specializes" in using `bertviz`. Some track position, some track syntax, some track coreference. |
| **Surprise moment** | You can prune many heads with minimal accuracy loss — not all heads are equally important. Show the "lottery ticket" effect. |

---

## PART III — The Transformer

### Module 9: The Full Transformer Block

| | |
|---|---|
| **Core idea** | Attention + Feed-Forward + Residual Connections + Layer Norm = one block. Stack N of these. |
| **Key math** | Pre-norm: `x = x + MultiHead(LN(x))` then `x = x + FFN(LN(x))` where `FFN(x) = W₂ · SwiGLU(W₁x) + b₂`. Modern LLMs use SwiGLU/GELU, not ReLU. |
| **Residual stream view** | The residual connection means each layer *adds* to a running sum. Information flows through the "residual stream" and each layer contributes a delta. This is key to interpretability. |
| **Interactive demo** | Build a transformer block from scratch in PyTorch (~50 lines). Inspect tensor shapes at every stage. Remove residual connections → training collapses. |
| **Surprise moment** | Residual connections are the most important part — without them, deep transformers can't train. Visualize gradient norms with/without residuals. |

### Module 10: Positional Encoding — Order Matters

| | |
|---|---|
| **Core idea** | Attention is permutation-invariant. Without positional info, "dog bites man" = "man bites dog". |
| **Key math** | **Sinusoidal:** `PE(pos, 2i) = sin(pos / 10000^(2i/d))`. **RoPE:** rotation matrix `R(θ·pos)` applied to Q and K so `q·k` depends on relative position. **ALiBi:** linear bias `−m|i−j|` added to attention logits. |
| **Interactive demo** | Visualize sinusoidal encodings as a heatmap. Compare with RoPE and ALiBi. Show how RoPE enables length extrapolation while sinusoidal breaks. |
| **Surprise moment** | RoPE encodes *relative* distance via rotation: `Rθm q · Rθn k = q · Rθ(n-m) k`. Elegant. |

### Module 11: Encoder vs Decoder vs Encoder-Decoder

| | |
|---|---|
| **Core idea** | Three architectural families. The industry converged on decoder-only, but understanding all three reveals *why*. |
| **Key comparison** | **Encoder-only (BERT):** bidirectional, MLM. Best for classification/embeddings. **Decoder-only (GPT):** causal, next-token. Best for generation. **Encoder-decoder (T5):** cross-attention bridge. Best for translation/summarization. |
| **Key math** | Cross-attention: `Attention(Q_dec, K_enc, V_enc)` — decoder queries attend to encoder outputs. |
| **Interactive demo** | Same text processed by BERT vs GPT-2 vs T5. Show the different masking patterns and attention flows. |
| **Surprise moment** | Decoder-only won the scaling race because next-token prediction is a *simpler objective* that scales more predictably (Chinchilla scaling laws). |

### Module 12: Putting It All Together — Full Architecture Walkthrough

| | |
|---|---|
| **Core idea** | Trace a single prompt from raw text to output probability, through every layer of a real LLM. |
| **Architecture** | `Input → Tokenizer → Embedding + Position → [Transformer Block × N] → LayerNorm → Linear → Softmax → Next token`. Annotate every tensor shape. |
| **Key numbers** | For a 7B model: `d_model=4096, n_heads=32, n_layers=32, vocab=32000, FFN_dim=11008 (SwiGLU)`. Breakdown: ~65% params in FFN, ~30% in attention, ~5% in embeddings. |
| **Interactive demo** | `nanoGPT`-style: build a complete tiny transformer (4 layers, 4 heads, d=128). Train on Shakespeare. Generate text. Inspect every activation. |
| **Surprise moment** | FFN layers store most of the "knowledge" — attention is just the routing mechanism. |

---

## PART IV — Scaling & Efficiency

### Module 13: KV Cache — Don't Recompute the Past

| | |
|---|---|
| **Core idea** | During generation, previous tokens' Keys and Values don't change. Cache them to avoid redundant computation. |
| **Key math** | Without cache: `O(n²)` total for n tokens. With cache: `O(n)` per new token. Memory: `2 × n_layers × d_model × seq_len × bytes_per_param`. |
| **Concrete example** | Llama-2 70B, 4K context, fp16: KV cache = `2 × 80 × 8192 × 4096 × 2 bytes ≈ 10 GB` — just for the *cache*, separate from model weights. |
| **Interactive demo** | Generate 100 tokens with and without KV cache. Show wall-clock time, FLOPs counter, and memory growth. |
| **Surprise moment** | KV cache is often the memory bottleneck, not model weights. This is why context length is so expensive. |

### Module 14: KV Cache Optimization — PagedAttention, GQA, MQA

This module covers the *algorithmic* techniques for reducing KV cache cost. (Module 26 covers the *systems* that implement them.)

| | |
|---|---|
| **Core idea** | The KV cache is the #1 memory bottleneck for long-context serving. Three families of solutions attack it differently. |

**14a: Paged Attention (from vLLM)**

| | |
|---|---|
| **Concept** | Traditional KV cache allocates `max_seq_len × d` contiguous memory per request — massive waste for short sequences. Paged Attention borrows *virtual memory* from OS design: store KV in non-contiguous pages, use a page table for lookup. |
| **Key math** | Traditional: waste = `(max_len - actual_len) × d × 2 bytes` per request. Paged: waste ≤ `(B-1) × d × 2 bytes` where B = block size (typically 16 tokens). With B=16 and avg 60% utilization → 50-70% memory savings. |
| **OS analogy** | Physical pages ↔ GPU memory blocks. Page table ↔ block table. Page fault ↔ allocate new block on demand. Copy-on-write ↔ shared system prompt prefix stored once for all requests. |
| **Interactive demo** | Simulate 20 concurrent requests with varying lengths. Visualize GPU memory as a grid: traditional (lots of grey "wasted" blocks) vs paged (tightly packed). |

**14b: Grouped-Query Attention (GQA) & Multi-Query Attention (MQA)**

| | |
|---|---|
| **Concept** | Instead of each attention head having its own K and V, share them across groups of heads. MQA: all heads share one KV. GQA: groups of heads share KV (a middle ground). |
| **Key math** | Standard MHA: `n_heads × 2 × d_head × seq_len` KV memory. MQA: `1 × 2 × d_head × seq_len` (÷ n_heads reduction). GQA with g groups: `g × 2 × d_head × seq_len`. Llama-2 70B uses GQA with 8 groups → 8× KV cache reduction. |
| **Interactive demo** | Visualize the K/V sharing pattern: MHA (every head unique), GQA (groups share), MQA (all share). Plot memory savings vs quality degradation. |
| **Surprise moment** | GQA achieves 95%+ of MHA quality with 4-8× less KV memory. This is why every modern model uses it. |

**14c: Sliding Window Attention & Sparse Patterns**

| | |
|---|---|
| **Concept** | Not every token needs to attend to all previous tokens. Limit attention to a local window (Mistral: 4096 tokens), with optional global tokens. |
| **Key math** | Full attention: `O(n²)` memory and compute. Sliding window W: `O(n × W)`. With layer stacking, effective receptive field = `W × n_layers`. |
| **Interactive demo** | Visualize attention patterns: full causal triangle vs sliding window "band" vs Longformer's hybrid (local + global). |

### Module 15: Mixture of Experts (MoE)

| | |
|---|---|
| **Core idea** | Not every token needs every parameter. A router sends each token to only the top-k experts (sub-FFNs). |
| **Key math** | Router: `G(x) = TopK(softmax(Wg · x))`. Expert output: `y = Σᵢ G(x)ᵢ · Eᵢ(x)`. Load-balancing loss: `L_bal = α · n_experts · Σᵢ fᵢ · pᵢ`. |
| **Modern variants** | **DeepSeek-V3:** shared experts (always active) + routed experts. **Llama-4 Maverick:** alternates MoE and dense blocks every other layer. **CMoE (Carved MoE):** carves MoE from pretrained dense models without full retraining. **KIMI K2.5:** 1T total params, 32B active per token. |
| **Interactive demo** | Build a toy MoE with 8 experts, top-2 routing. Remove load-balancing → watch "expert collapse." Visualize expert specialization. |
| **Surprise moment** | 60%+ of open-source frontier models released in 2025-2026 use MoE. It's no longer exotic — it's the baseline. |

### Module 16: Quantization — From GPTQ to TurboQuant to RotorQuant

| | |
|---|---|
| **Core idea** | Shrink weights and/or KV cache from fp16 to int8/int4 or even 3-bit. The field has moved from "weight-only" to "KV cache quantization." |

**16a: Foundations (2023-2024)**

| | |
|---|---|
| **Key math** | Absmax: `q = round(x / max(|x|) × 127)`. GPTQ: layer-wise optimal quantization minimizing `‖WX - Q(W)X‖²`. AWQ: protect salient channels (1% of weights matter 100x more). |
| **Interactive demo** | Quantize a weight matrix to INT8 and INT4. Plot original vs quantized distributions. Measure perplexity degradation. |

**16b: TurboQuant (Google, ICLR 2026) — KV Cache at 3-bit**

| | |
|---|---|
| **Core idea** | Compresses KV cache to 3-bit *without training or fine-tuning*. Uses random rotation to simplify data geometry + Quantized Johnson-Lindenstrauss (QJL) for error correction — all with just 1 bit of overhead. |
| **Key innovations** | Data-oblivious (no offline calibration needed). Introduces PolarQuant. Operates near mathematical lower bounds with theoretical guarantees. Actually *speeds up* inference compared to full-precision. |
| **Key math** | Random rotation `R ∈ ℝ^{d×d}`: decorrelates vector components so uniform quantization works. QJL: project error into random subspace, quantize the correction. Total: `3 + 1 = 4 bits` per value with provable accuracy bounds. |
| **Interactive demo** | Apply TurboQuant to a KV cache matrix. Compare reconstruction error vs naive INT4 and INT8. Show: 3-bit TurboQuant ≈ INT8 quality at 2.6× less memory. |

**16c: RotorQuant — Clifford Algebra Meets Quantization**

| | |
|---|---|
| **Core idea** | A reimagining of TurboQuant that replaces rotation matrices with Clifford rotors from geometric algebra Cl(3,0). Result: 10-31× faster with 44× fewer parameters. |
| **Key math** | Instead of `d×d` rotation matrix (d²-parameter, O(d²) multiply), use a Clifford rotor `R` applied as sandwich product `RxR̃`. For d=128: rotation = 16,384 multiply-adds; rotor = ~100 per vector. |
| **Performance** | 10-19× faster on CUDA, 9-31× faster on Metal. 5× KV cache compression with 99.0% attention fidelity. |
| **Interactive demo** | Side-by-side: TurboQuant vs RotorQuant on the same KV cache. Benchmark throughput. Visualize the Clifford rotor as a geometric rotation in 3D. |
| **Surprise moment** | Abstract algebra (invented in the 1870s) is now a practical GPU optimization technique. Math never expires. |

### Module 17: LoRA & QLoRA — Parameter-Efficient Fine-Tuning

| | |
|---|---|
| **Core idea** | Instead of updating all weights, inject small trainable low-rank matrices. Train 0.1% of parameters, get 90%+ of full fine-tune quality. |
| **Key math** | `W' = W + BA` where `B ∈ ℝ^(d×r)`, `A ∈ ℝ^(r×d)`, `r ≪ d`. QLoRA: quantize base model to 4-bit, then add LoRA on top → fine-tune a 65B model on a single GPU. |
| **Interactive demo** | Fine-tune GPT-2 on a tiny dataset: full fine-tune vs LoRA. Compare memory, time, and quality. |
| **Surprise moment** | Visualize the rank-r update as a thin "lens" overlaid on the frozen weight matrix. |

---

## PART V — Generation & Alignment

### Module 18: Next-Token Prediction — The Core Objective

| | |
|---|---|
| **Core idea** | LLMs are trained on one objective: given all previous tokens, predict the next token's probability distribution. |
| **Key math** | `P(xₜ | x₁, ..., xₜ₋₁) = softmax(W_head · hₜ)`. Loss: `L = -(1/T) Σₜ log P(xₜ | x<t)`. Perplexity: `PPL = e^L`. |
| **Interactive demo** | Feed a prompt into a small model. Show the full distribution over vocabulary. Visualize how it sharpens as context grows. |
| **Surprise moment** | The model assigns non-zero probability to *every* token. "The cat sat on the ___" → "mat" isn't 100%. Show the long tail. |

### Module 19: Sampling Strategies — Temperature, Top-k, Top-p, Min-p

| | |
|---|---|
| **Core idea** | How you *choose* from the distribution dramatically changes the output. |
| **Key math** | Temperature: `P'(x) = softmax(logits / T)`. Top-k: keep top k, renormalize. Top-p (nucleus): smallest set where `Σ P ≥ p`. Min-p: keep tokens where `P(x) ≥ p_min · P(x_max)`. |
| **Interactive demo** | Interactive sliders for T, k, p, min-p. Generate text in real-time. Show the distribution reshaping live. |
| **Surprise moment** | T=0 (greedy) → repetitive. T=2.0 → gibberish. The sweet spot is narrow and task-dependent. |

### Module 20: RLHF & DPO — The Classics

| | |
|---|---|
| **Core idea** | Pre-training teaches *prediction*. Alignment teaches *helpfulness, harmlessness, honesty*. RLHF (2022) and DPO (2023) started the alignment era. |
| **RLHF math** | Train reward model `r(x, y)`. PPO objective: `max E[r(x,y) − β·KL(π‖π_ref)]`. Requires 3 models in memory: policy, reward, value (critic). |
| **DPO math** | Eliminates reward model. Direct optimization: `L_DPO = -log σ(β(log π(y_w|x)/π_ref(y_w|x) − log π(y_l|x)/π_ref(y_l|x)))`. Just 1 model + reference. |
| **Also mention** | **SimPO:** outperforms DPO by 6.4 points on AlpacaEval 2. **KTO (Kahneman-Tversky):** works with binary thumbs-up/down instead of pairwise preferences. **ORPO:** merges SFT + preference optimization into one stage. |
| **Interactive demo** | Given two responses, train a tiny reward model. Show how the policy shifts. Visualize the KL penalty. |
| **Surprise moment** | DPO is "offline" (fixed preference data). This limits it — the model can't improve beyond the quality of the preference pairs. This is exactly what GRPO fixes. |

### Module 21: GRPO & RLVR — The New Wave (DeepSeek-R1 and Beyond)

| | |
|---|---|
| **Core idea** | **GRPO** (Group Relative Policy Optimization) eliminates the value network from PPO by computing advantages *within a group of sampled outputs*. **RLVR** (RL with Verifiable Rewards) replaces human preferences with automatic verification (math correctness, code tests). Together, they enabled DeepSeek-R1's breakthrough: emergent reasoning without human-written chain-of-thought data. |
| **GRPO math** | For each prompt, sample G outputs. Score with reward model. Advantage: `Aᵢ = (rᵢ − mean(r)) / (std(r) + ε)`. Loss: `L = E[min(Aᵢ · ratioᵢ, Aᵢ · clip(ratioᵢ, 1−δ, 1+δ))]`. No critic network needed → ~50% memory reduction vs PPO. |
| **RLVR insight** | Instead of `r(x,y) = human preference`, use `r(x,y) = 1 if math_answer_correct else 0`. This is: (a) infinitely scalable, (b) perfectly accurate, (c) enables training beyond human-labeled data quality. |
| **Also cover** | **DAPO (Distributed Advantage PO):** designed for long chain-of-thought. Introduces Clip-Higher, Dynamic Sampling, Overlong Reward Shaping. Qwen2.5-32B hit 50 points on AIME 2024 with DAPO. |
| **Interactive demo** | Sample 16 responses to a math problem. Score each (correct/incorrect). Compute group advantages. Update a tiny model with GRPO. Compare: DPO (needs preference pairs) vs GRPO (just needs a verifier). |
| **Surprise moment** | DeepSeek-R1 trained with pure GRPO/RLVR (no human reasoning annotations) spontaneously learned to self-reflect and verify — the "aha moment" of 2025. |

### Module 22: Prompt Engineering as Bayesian Conditioning

| | |
|---|---|
| **Core idea** | A prompt conditions the probability distribution. System prompts, few-shot examples, and CoT all shape `P(output | context)`. |
| **Key math** | `P(answer | prompt) ∝ P(prompt | answer) · P(answer)`. Chain-of-thought: `P(answer | Q) = Σ P(answer | reasoning) · P(reasoning | Q)`. |
| **Interactive demo** | Same question, different prompts. Show token-level probability shifts. Compare zero-shot vs few-shot vs CoT. |
| **Surprise moment** | "Let's think step by step" literally reshapes the probability landscape. It's conditioning, not magic. |

---

## PART VI — Reasoning & Test-Time Compute

### Module 23: Reasoning Models & Chain-of-Thought

| | |
|---|---|
| **Core idea** | A new class of models (o1, DeepSeek-R1, QwQ) are trained to "think out loud" before answering. The chain-of-thought isn't just a prompt trick — it's baked into training via GRPO/RLVR. |
| **Key distinction** | **Prompting-based CoT** (Module 22): you ask the model to reason. **Training-based CoT** (this module): the model is *trained* to reason, with RL rewards for correct final answers. The reasoning traces emerge naturally. |
| **Key models** | OpenAI o1/o3, DeepSeek-R1, Qwen QwQ, Gemini "Deep Think". |
| **Interactive demo** | Compare the same math problem: GPT-4 (direct answer, often wrong) vs R1-style (extended reasoning trace, self-corrects mid-thought). Visualize token count and accuracy trade-off. |
| **Surprise moment** | R1 sometimes generates "Wait, let me reconsider..." and actually catches its own errors. This emerged from RL without being explicitly programmed. |

### Module 24: Test-Time Compute Scaling — A New Axis

| | |
|---|---|
| **Core idea** | Training-time scaling (more parameters, more data) was the only known axis. Now there's a second: *inference-time scaling* — letting the model spend more compute per query. |
| **Key insight** | More thinking tokens = better answers, up to a point. This is a fundamentally different trade-off: instead of a bigger model, use a smaller model that thinks longer. |
| **Key math** | Scaling law: `accuracy ≈ f(train_compute, test_compute)`. Both axes contribute. Optimal allocation depends on difficulty — easy questions don't benefit from extended reasoning, hard questions do. |
| **Interactive demo** | Same model, same question, vary the "thinking budget" (max reasoning tokens). Plot accuracy vs inference tokens. Show: diminishing returns on easy questions, big gains on hard ones. |
| **Surprise moment** | A 32B reasoning model with 10K thinking tokens can outperform a 405B model with direct answering. Size isn't everything anymore. |
| **Also cover** | The "Densing Law": capability density doubles every 3.5 months. By 2026, 8B models trained on trillions of tokens are shockingly capable for their size. |

### Module 25: Speculative Decoding — Draft, Then Verify

| | |
|---|---|
| **Core idea** | Use a fast small "draft" model to guess the next k tokens, then verify all k in a single forward pass of the large model. Accepted tokens are *provably identical in distribution* to the large model alone. |
| **Key math** | Draft: `x₁...xₖ ~ q(x)`. Target: compute `p(xᵢ)` for all i in parallel. Accept xᵢ if `p(xᵢ) ≥ q(xᵢ)`, else reject and resample. Expected speedup: `1/(1−α)` where α = acceptance rate. Typical: 2-3× speedup. |
| **Interactive demo** | Simulate speculative decoding: show the draft model's guesses, the target model's verification, and the accept/reject decisions. Vary draft model quality and show how acceptance rate changes. |
| **Surprise moment** | The output distribution is *mathematically identical* to running the large model alone — zero quality loss, pure speed gain. |

---

## PART VII — Serving Infrastructure & Tooling

### Module 26: vLLM & Serving Engines

| | |
|---|---|
| **Core idea** | Serving LLMs to many concurrent users requires optimized engines. vLLM pioneered Paged Attention (Module 14a) and continuous batching into one production-ready system. |
| **Key concepts** | **Continuous batching:** don't wait for the longest sequence — release completed requests, add new ones dynamically. **Speculative decoding** (Module 25) integrated. **Prefix caching** via copy-on-write. |
| **Metrics** | Throughput (tokens/sec), TTFT (time-to-first-token), TPOT (time-per-output-token), P99 latency. |
| **Interactive demo** | Benchmark a small model: single request vs 10 concurrent. Plot throughput vs latency. Compare naive sequential vs vLLM-style continuous batching. |
| **Landscape** | **vLLM** (PagedAttention, open-source standard). **SGLang** (co-designed frontend + runtime). **TGI** (HuggingFace). **TensorRT-LLM** (NVIDIA). **llama.cpp** (CPU/Metal inference, GGUF format). |

### Module 27: LiteLLM & Model Routing

| | |
|---|---|
| **Core idea** | In production, you call multiple LLM providers. LiteLLM provides a unified interface + routing + fallbacks + cost tracking across 100+ models. |
| **Key concepts** | **Unified API:** one interface, many providers. **Router:** load-balance (round-robin, least-busy, cost-optimized). **Fallbacks:** if provider A fails, try B. **Budget caps** per user/team. **Proxy server:** drop-in OpenAI-compatible endpoint. |
| **Interactive demo** | Route between 2 models. Send 20 requests, visualize routing decisions, latency, cost. Simulate provider failure → fallback. |
| **Also cover** | **OpenRouter:** model marketplace. **Portkey / Helicone:** observability + caching. The "LLM gateway" pattern is now standard infra. |

### Module 28: Continuous Batching & Request Scheduling

| | |
|---|---|
| **Core idea** | Naive batching wastes GPU cycles on padding. Continuous batching fills gaps dynamically. |
| **Key math** | Static: throughput ≤ `batch_size / max(seq_lengths)`. Continuous: throughput ≈ `total_tokens / time` regardless of length variance. |
| **Interactive demo** | Animate GPU utilization: static batching (grey idle blocks) vs continuous (nearly full). |
| **Surprise moment** | Continuous batching improves throughput 2-8× with zero accuracy change — pure scheduling win. |

### Module 29: Distributed Inference — Tensor & Pipeline Parallelism

| | |
|---|---|
| **Core idea** | When a model doesn't fit on one GPU, split it. Tensor parallel: slice each layer. Pipeline parallel: assign different layers to different GPUs. |
| **Key math** | **Tensor parallel:** split `W` into `[W₁|W₂]`, each GPU computes half, then all-reduce. Cost: `O(d)` communication per layer. **Pipeline parallel:** sequential dependency creates "bubbles." Micro-batching reduces them. |
| **Interactive demo** | Visualize 4-GPU setups. Animate the pipeline bubble and how micro-batching fills it. |
| **Also cover** | Expert parallelism for MoE. FSDP for training. |

---

## PART VIII — Agents & Evaluation

### Module 30: Tool Use & Function Calling

| | |
|---|---|
| **Core idea** | LLMs can't compute or access live data. Tool use lets them emit structured calls and incorporate results. |
| **Key math** | Decision as classification: `P(tool_i | context)`. The model generates valid JSON for tool arguments. |
| **Interactive demo** | Build a mini tool-use loop: model chooses between calculator, weather API, and "just answer." Full cycle: prompt → tool call → result injection → final answer. |
| **Surprise moment** | The model doesn't "know" it's calling a tool — it generates text that *happens* to be valid JSON. The harness does the real work. |

### Module 31: Agent Loops & Harnesses

| | |
|---|---|
| **Core idea** | An agent is an LLM in a loop: observe → think → act → observe. The harness manages state, memory, and dispatch. |
| **Key math** | Formalize as POMDP: `(S, A, T, R, O, Ω)`. In practice: `state = (system_prompt, history, tool_results)`. ReAct: interleave Reasoning + Acting. |
| **SDK landscape (2025-2026)** | **Anthropic Agent SDK:** Claude-native, tool-use focused. **OpenAI Agents SDK** (March 2025): replaced Swarm, explicit handoffs. **Google ADK** (April 2025): full agentic developer kit. **LangGraph:** state-machine approach with nodes/edges. **AutoGen v0.4:** async, event-driven redesign. |
| **Interactive demo** | Build a minimal agent harness in ~80 lines of Python. 3 tools. Solve a multi-step task. Log every loop iteration. |
| **Surprise moment** | Most agent failures are harness failures (bad prompts, missing error handling, infinite loops) — not model failures. |

### Module 32: Multi-Agent & Swarm — From Concept to KIMI K2.5

| | |
|---|---|
| **Core idea** | Multiple specialized agents collaborate. The field has evolved from hand-wired pipelines (2024) to self-directed swarms (2026). |

**32a: Foundations — Multi-Agent Patterns**

| | |
|---|---|
| **Patterns** | **Handoff:** agent A passes context + control to agent B. **Shared memory / blackboard:** agents read/write common state. **Agent-as-tool:** one agent is callable by another. **Debate:** agents argue to improve answers. |
| **Interactive demo** | Planner → Coder → Reviewer pipeline. Visualize message passing. Show how the Reviewer catches bugs. |

**32b: OpenAI Swarm → KIMI K2.5 Agent Swarm**

| | |
|---|---|
| **OpenAI Swarm** | Lightweight multi-agent with function-based handoffs. Simple but limited: predefined roles, sequential execution. |
| **KIMI K2.5 Agent Swarm** | Moonshot AI's breakthrough (January 2026). A single model dynamically spawns up to 100 sub-agents in parallel, *without predefined roles*. The model learns when to parallelize, how many agents to spawn, and how to merge results. |
| **Architecture** | 1T total params (MoE, 32B active). Four steps: Task Decomposition → Dynamic Sub-Agent Creation → Parallel Execution (up to 1,500 coordinated tool calls per session) → Result Aggregation. |
| **Key innovation: PARL** | Parallel-Agent Reinforcement Learning. Solves two problems: (1) credit assignment across parallel agents, (2) "serial collapse" (tendency to revert to sequential). Uses staged reward shaping. |
| **Interactive demo** | Compare: sequential agent (tool calls one at a time) vs swarm (parallel sub-agents). Same research task. Visualize timeline, speedup (up to 4.5×), and result quality. |
| **Surprise moment** | KIMI K2.5 scored 78.4% on BrowseComp (web search benchmark) — beating GPT-5.2. Self-directed parallelism is a genuine capability leap. |

### Module 33: Evaluation Frameworks — Measuring What Matters

| | |
|---|---|
| **Core idea** | You can't improve what you can't measure. The eval landscape has split into benchmark evals (test the model) and system evals (test the whole pipeline). |

**33a: Model Evals**

| | |
|---|---|
| **Frameworks** | **lm-evaluation-harness (EleutherAI):** the standard. Run any model against MMLU, HellaSwag, GSM8K, HumanEval, etc. **HELM (Stanford):** holistic eval across accuracy, calibration, robustness, fairness, toxicity. |
| **Key metrics** | Perplexity, pass@k (code), Elo rating (Chatbot Arena), exact-match, F1. |

**33b: System & Agent Evals**

| | |
|---|---|
| **Frameworks** | **Inspect AI (UK AISI / π-framework):** structured eval with solvers, scorers, sandboxed tool-use. **Braintrust / Langsmith:** production-grade eval + tracing for deployed apps. |
| **New approaches** | **LLM-as-judge:** use a strong model to grade a weaker model's outputs. **Revision distance:** how much a human must edit the output (human-centered quality metric). **Adversarial evals:** rephrased benchmarks to detect memorization. |

**33c: Running Your Own Eval**

| | |
|---|---|
| **Interactive demo** | Build a tiny eval harness: define 10 test cases, run through a model, score with (a) exact match, (b) LLM-as-judge. Compute pass rates with confidence intervals. |
| **Surprise moment** | Demo benchmark contamination: model scores 90% on original MMLU question, 40% on a trivially rephrased version. Scores can be gamed. |

---

## Bonus Modules (Appendix / Electives)

### Bonus A: Flash Attention

| | |
|---|---|
| **Core idea** | Standard attention is memory-bound (N×N matrix). Flash Attention tiles the computation to stay in fast SRAM. Same output, smarter compute. |
| **Key math** | IO complexity: standard `O(N²)` HBM reads → Flash `O(N²d/M)` where M = SRAM size. |

### Bonus B: Retrieval-Augmented Generation (RAG)

| | |
|---|---|
| **Core idea** | Retrieve relevant documents at inference time instead of stuffing all knowledge into weights. |
| **Pipeline** | Chunking → Embedding → Vector store → Retrieval → Reranking → Generation. |

### Bonus C: Scaling Laws & Chinchilla

| | |
|---|---|
| **Key math** | `L(N, D) ≈ (Nc/N)^αN + (Dc/D)^αD + L∞`. Chinchilla optimal: `N_opt ∝ C^0.5`, `D_opt ∝ C^0.5`. |
| **2026 update** | "Densing Law": capability density doubles every 3.5 months. 8B models in 2026 match 70B models from 2024. |

### Bonus D: Synthetic Data & Self-Improvement

| | |
|---|---|
| **Core idea** | Models generate their own training data. Magpie, Source2Synth, and DataGen enable creating alignment data from scratch. Humans validate rather than create. |

### Bonus E: Multimodal — Vision-Language Models

| | |
|---|---|
| **Key models** | Gemini 2.5 (1M+ context), Qwen3-VL (256K native, 1M expandable), KIMI-VL (MoE decoder + MoonViT), UI-TARS-1.5 (ByteDance, UI understanding). |

---

## Recommended Learning Paths

**Weekend crash course (8 hours):**
> Modules 1 → 2 → 5 → 6 → 9 → 18 → 19

**Week-long deep dive (25 hours):**
> All of Parts I–V, in order

**"I just want to understand agents":**
> Modules 1 → 2 → 5 → 9 → 18 → 30 → 31 → 32

**"I'm deploying models in production":**
> Modules 13 → 14 → 15 → 16 → 17 → 25 → 26 → 27 → 28 → 29 → 33

**"I want to understand the math deeply":**
> Modules 3 → 5 → 6 → 7 → 8 → 9 → 10 → 18 → 21 + Bonus A, C

**"What's new in 2025-2026?":**
> Modules 15 → 16b → 16c → 21 → 23 → 24 → 25 → 32b → Bonus D, E

---

## Technical Stack

| Component | Tool |
|---|---|
| **Notebooks** | Jupyter / Google Colab |
| **Math rendering** | LaTeX in Markdown cells |
| **Visualizations** | `matplotlib`, `plotly` (interactive), `bertviz` (attention) |
| **Models** | `tiktoken`, `transformers` (HuggingFace), `nanoGPT`, `trl` (for GRPO/DPO training) |
| **Serving** | `vllm`, `litellm`, `sglang`, `llama-cpp-python` |
| **Eval** | `lm-eval-harness`, `inspect-ai`, `braintrust` |
| **Interactive widgets** | `ipywidgets` for sliders, toggles, real-time parameter tuning |
| **Future HTML version** | `nbconvert` → standalone HTML, or Voilà, or custom React |

---

## Design Principles

1. **Every equation earns its place** — code cell right below implements it
2. **No "just trust me"** — every claim is verifiable by running a cell
3. **Break things on purpose** — remove √dₖ, disable residuals, kill load balancing, remove causal mask
4. **Progressive complexity** — NumPy (Part I) → PyTorch (Part II-III) → HuggingFace (Part IV-V) → Infra tools (Part VI-VII)
5. **Bite-sized** — each module is 20–30 minutes
6. **Production-aware** — theory + the tools engineers actually use
7. **Current as of April 2026** — GRPO, TurboQuant, RotorQuant, KIMI swarm, test-time compute scaling
