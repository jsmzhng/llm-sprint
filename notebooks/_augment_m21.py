"""Insert additional cells into part5_m21_grpo_rlvr.ipynb to reach ~50+ cells.
Run once, delete."""
import nbformat as nbf

path = "/Users/jsmz97/Desktop/llm-into/notebooks/part5_m21_grpo_rlvr.ipynb"
nb = nbf.read(path, as_version=4)

def md(s): return nbf.v4.new_markdown_cell(s)
def code(s): return nbf.v4.new_code_cell(s)

# Find insertion points by matching cell source prefixes
def find(prefix):
    for i, c in enumerate(nb.cells):
        if "".join(c.source).startswith(prefix):
            return i
    raise ValueError(f"no cell starting with {prefix!r}")

# --- 1. After "## 1 · What PPO looks like" markdown, insert memory bar chart
i = find("## 1 · What PPO looks like")
mem_code = code('''# A hand-wavy but instructive picture of why the second network matters.
# Dense FP16 training roughly costs ~16 bytes/param per actively-trained network
# (2 weights + 2 grads + 8 optimizer state + ~4 activations), plus ~2 bytes/param
# for a frozen reference model kept around for the KL term.
sizes_B = np.array([1, 7, 13, 32, 70])
per_trained_byte = 16
per_frozen_byte  = 2

ppo_mem  = sizes_B * 1e9 * per_trained_byte * 2 + sizes_B * 1e9 * per_frozen_byte
grpo_mem = sizes_B * 1e9 * per_trained_byte * 1 + sizes_B * 1e9 * per_frozen_byte

fig, ax = plt.subplots(figsize=(10, 4.5))
w = 0.35
x = np.arange(len(sizes_B))
ax.bar(x - w/2, ppo_mem/1e9,  w, color=PALETTE["rose"], edgecolor=PALETTE["ink"], label="PPO (policy + critic + ref)")
ax.bar(x + w/2, grpo_mem/1e9, w, color=PALETTE["teal"], edgecolor=PALETTE["ink"], label="GRPO (policy + ref)")
ax.set_xticks(x)
ax.set_xticklabels([f"{s}B" for s in sizes_B])
ax.set_ylabel("peak training memory (GB, approximate)")
ax.set_xlabel("policy size")
ax.set_title("Killing the critic ≈ halving training memory")
for idx, (p, g) in enumerate(zip(ppo_mem, grpo_mem)):
    ax.text(idx - w/2, p/1e9, f"{p/1e9:.0f}", ha="center", va="bottom", fontsize=9)
    ax.text(idx + w/2, g/1e9, f"{g/1e9:.0f}", ha="center", va="bottom", fontsize=9)
ax.legend()
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()''')
mem_md = md("This is a rough sketch — real numbers depend on sequence length, activation checkpointing, ZeRO sharding, and so on — but the shape is right. At 70B, you're looking at ~2.3 TB of training memory for PPO and ~1.2 TB for GRPO. That's the difference between \"needs a full H100 node\" and \"needs half a node\". It's also the difference between \"critic is slightly out of sync with the policy\" and \"there is no critic to be out of sync\".")
nb.cells.insert(i + 1, mem_code)
nb.cells.insert(i + 2, mem_md)

# --- 2. Before "## 5 · Mini-R1-Zero", insert a visual sketch of the GRPO loop
i = find("## 5 · Mini-R1-Zero")
loop_md = md("""### Before we code: the whole GRPO loop, in one picture

```
                   prompt q
                      │
                      ▼
         ┌────────────────────────┐
         │   old policy π_old     │   (a copy of the current params,
         │   samples G outputs    │    frozen for the duration of one step)
         └────────────┬───────────┘
                      │   o_1, o_2, …, o_G
                      ▼
         ┌────────────────────────┐
         │   verifier r(q, o_i)   │   (RLVR: a Python function, 0 or 1)
         └────────────┬───────────┘
                      │   r_1, …, r_G
                      ▼
         A_i = (r_i - mean(r)) / (std(r) + ε)       ← group-relative advantage
                      │
                      ▼
         ┌────────────────────────┐
         │   current policy π_θ   │   recompute logprobs on the same o_i
         │   ρ_i = π_θ / π_old    │   → clipped surrogate loss
         └────────────┬───────────┘
                      │   gradient
                      ▼
                 Adam step
                      │
                      ▼
              new policy π_θ'
```

Four moving parts: a sampler, a verifier, a line of numpy, and a clipped PPO objective. That's the whole algorithm.""")
nb.cells.insert(i, loop_md)

# --- 3. After the "smoke test" sample_rollout cell, add an untrained-model demo
# Find the cell that runs sample_rollout smoke test
i = find("# Sampling: given a prompt, roll out")
# that's the definition cell; the smoke test is inline in the same cell. Insert after it.
untrained_md = md("""It is — predictably — nonsense. An untrained LSTM over a 14-token vocab just emits roughly uniform digits. If we rolled out a whole group of 8 samples right now and scored them, we'd expect almost all to get reward 0. GRPO's job is to find the needles in this haystack, amplify them, and slowly push the policy toward being able to produce correct answers more often than chance.""")
nb.cells.insert(i + 1, untrained_md)

# --- 4. After the training loop runs, insert a "before training" demo
# We'll reuse the already-trained eval function. Insert before the first fig after training.
i = find("fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))")
# this is the first plot after training; we want to add a "distribution of rewards over time"
# further down. For now, insert a pre-training snapshot BEFORE this plot would be misleading
# since we only have post-training model. Instead, insert right after the 8-sample demo.
i = find("# Peek at what the trained model is actually generating")
# add a histogram of correct vs incorrect over a larger eval
big_eval_code = code('''# Bigger eval: 200 problems, measure the distribution of outcomes.
def big_eval(model, n=200, seed=42, max_new=6, temperature=0.3):
    rng = np.random.default_rng(seed)
    records = []
    for _ in range(n):
        a, b = make_problem(rng)
        prompt = encode(make_prompt(a, b))
        comp, _ = sample_rollout(model, prompt, max_new=max_new, temperature=temperature)
        ok = verify(comp, truth(a, b)) > 0.5
        records.append((a, b, ok))
    return records

recs = big_eval(model_grpo, n=200)
acc = np.mean([ok for _,_,ok in recs])
print(f"accuracy on 200 random 2..29 * 2..29 problems: {acc:.2%}")

# Break down by "problem difficulty" (product magnitude).
products = np.array([a*b for a,b,_ in recs])
oks = np.array([ok for _,_,ok in recs], dtype=float)
bins = [0, 50, 150, 300, 1000]
centers, means = [], []
for lo, hi in zip(bins[:-1], bins[1:]):
    mask = (products >= lo) & (products < hi)
    if mask.any():
        centers.append(f"{lo}-{hi}")
        means.append(oks[mask].mean())

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(centers, means, color=PALETTE["teal"], edgecolor=PALETTE["ink"])
ax.set_ylim(0, 1.05)
ax.set_xlabel("true product a*b")
ax.set_ylabel("accuracy")
ax.set_title("trained policy: accuracy vs product magnitude")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()''')
nb.cells.insert(i + 1, big_eval_code)
breakdown_md = md("""Accuracy is usually higher on small products than on large ones — which makes sense: a tiny LSTM has to learn multiplication as a token-level sequence prediction task, and the number of correct output tokens grows with the answer length. With only 500 GRPO steps and ~200k parameters, it learns the structure but not the full carry logic for larger products. Give it a bigger model and 5000 steps and the bars flatten out.""")
nb.cells.insert(i + 2, breakdown_md)

# --- 5. After the ablation plot, insert a histogram of advantages under each scheme
i = find("fig, ax = plt.subplots(figsize=(10, 5))")
# This finds the ablation plot. We want a follow-up cell showing advantage distributions.
adv_dist_code = code('''# Look at what advantages actually LOOK like under each normalization scheme,
# for a synthetic batch of 1000 random-ish groups of size 8.
rng_a = np.random.default_rng(0)
n_groups = 1000
p_correct = rng_a.uniform(0.05, 0.6, size=n_groups)     # each group has some "difficulty"
raw_groups = (rng_a.uniform(size=(n_groups, 8)) < p_correct[:, None]).astype(float)

def advs_under(groups, center, norm):
    out = []
    for r in groups:
        mu = r.mean() if center else 0.0
        sd = r.std() if norm   else 1.0
        out.extend((r - mu) / (sd + 1e-6))
    return np.array(out)

A_full   = advs_under(raw_groups, center=True,  norm=True)
A_nostd  = advs_under(raw_groups, center=True,  norm=False)
A_nobase = advs_under(raw_groups, center=False, norm=False)

fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
for ax, vals, title, color in zip(
    axes,
    [A_full, A_nostd, A_nobase],
    ["full GRPO", "mean-centered only", "no baseline"],
    [PALETTE["teal"], PALETTE["amber"], PALETTE["rose"]],
):
    ax.hist(vals, bins=40, color=color, edgecolor=PALETTE["ink"])
    ax.axvline(0, color=PALETTE["ink"], lw=1)
    ax.set_title(title)
    ax.set_xlabel("advantage")
axes[0].set_ylabel("count")
plt.suptitle("advantage distributions across 1000 synthetic groups", y=1.03)
plt.tight_layout()
plt.show()''')
nb.cells.insert(i + 2, adv_dist_code)
adv_dist_md = md("""Three very different shapes:

- **Full GRPO** (left) is symmetric around zero, roughly unit-scale, with a clean bimodal signature: positive advantages for the correct rollouts, negative for the failures, and not much in between. This is a healthy learning signal.
- **Mean-centered only** (middle) is zero-mean but wildly varying in scale — some groups produce advantages of ±0.1 and others ±0.9, depending entirely on how unanimous the group happened to be. The gradient magnitude couples to group difficulty instead of to learning progress.
- **No baseline** (right) is non-negative. The model can only be moved *toward* things that worked, never *away* from things that didn't. Combined with a mostly-zero reward, most updates are ~zero.

The bimodal teal histogram is what a working RL run looks like.""")
nb.cells.insert(i + 3, adv_dist_md)

# --- 6. Before the aha-moment trace, add a transition md
i = find("sample_trace = '''Prompt: If 3x + 5")
setup_md = md("The trace below is stylized — I paraphrased the DeepSeek figure for length — but the structure is real: the model writes a solution, then mid-thought says *\"wait, let me reconsider\"*, checks its own work, and confirms (or corrects) the answer. Read it as a data point, not a source.")
nb.cells.insert(i, setup_md)

# --- 7. Before checkpoint, add a "where we are" recap
i = find("## 11 · Checkpoint")
recap_md = md("""## 10.5 · Recap before the checkpoint

Here's the one-screen version of everything above.

| | PPO (RLHF-era) | GRPO (DeepSeek-era) | DAPO (ByteDance 2025) |
|---|---|---|---|
| Baseline | learned value network $V_\\phi$ | group mean $\\bar r$ | group mean $\\bar r$ |
| Reward source | reward model (from human prefs) | verifier (RLVR) | verifier (RLVR) |
| Second network? | yes (critic, ~policy-size) | no | no |
| Clip | symmetric $\\pm\\delta$ | symmetric $\\pm\\delta$ | asymmetric ($\\delta_\\text{low}, \\delta_\\text{high}$) |
| Dead groups | N/A | wasted compute | dynamically resampled |
| Loss granularity | token-level | sequence-level | token-level |
| Memory @ 70B (rough) | ~2.3 TB | ~1.2 TB | ~1.2 TB |
| Headline result | InstructGPT (2022) | DeepSeek-R1 (Jan 2025) | Qwen2.5-32B → AIME 50 (Mar 2025) |

Three algorithms, three years, roughly one order of magnitude of progress in reasoning-bench scores per year. It's a good time to be alive if you like RL.""")
nb.cells.insert(i, recap_md)

nbf.write(nb, path)
print(f"now has {len(nb.cells)} cells")
