# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an interactive, code-first curriculum teaching how LLMs work — targeting smart generalists (engineers, designers, PMs, researchers). The curriculum is currently in the **outline/planning phase** with a single document: `LLM_Interactive_Curriculum_Outline.md`.

## Curriculum Structure

33 modules across 8 parts, each following the pattern: intuition → math (LaTeX) → interactive Python code → checkpoint quiz. Parts progress from foundations (tokenization, embeddings) through attention, transformers, scaling/efficiency, generation/alignment, reasoning, serving infra, and agents/eval.

## Technical Stack (planned)

- **Notebooks:** Jupyter / Google Colab
- **Visualizations:** matplotlib, plotly, bertviz, ipywidgets
- **Models:** tiktoken, HuggingFace transformers, nanoGPT, trl (GRPO/DPO)
- **Serving:** vllm, litellm, sglang, llama-cpp-python
- **Eval:** lm-eval-harness, inspect-ai, braintrust
- **Math:** LaTeX in Markdown cells
- **Progressive complexity:** NumPy (Part I) → PyTorch (Part II-III) → HuggingFace (Part IV-V) → Infra tools (Part VI-VII)

## Key Design Principles

- Every equation must have a runnable code cell implementing it
- "Break things on purpose" — modules encourage disabling key mechanisms (√dₖ, residuals, causal mask) to build intuition
- Each module targets 20–30 minutes
- Content is current as of April 2026 (covers GRPO, TurboQuant, RotorQuant, KIMI swarm, test-time compute)
