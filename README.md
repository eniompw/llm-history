# LLM Architecture Timeline

A comprehensive reference for the key architectural innovations that shaped modern large language models from the Transformer (2017) through 1T-parameter models (2026), plus practical guides for building LLMs from scratch.

## Table of Contents

- [Overview](#overview)
- [Architectural Milestones](#architectural-milestones)
- [Building Your Own LLM](#building-your-own-llm)
  - [Simplified Hybrid GPT/LLaMA Example](#simplified-hybrid-gptllama-example-microgpt-style)
  - [Key Architectural Ideas](#key-architectural-ideas)
  - [Quick-Start Guide](#quick-start-llama-2-in-4-lines)
  - [Performance Enhancements](#enhancing-microgpt-for-modern-performance)

## Overview

This timeline documents key architectural milestones from 2017 to April 2026, tracking the evolution from the foundational Transformer architecture through the emergence of sparse mixture-of-experts (MoE) models, scaling laws, alignment techniques (RLHF), and optimization innovations like FlashAttention. Each entry highlights the core innovation and its impact on subsequent model development.

## Architectural Milestones

Key architectural innovations, 2017 to Apr 2026.

| Date | Model / Org | Params / Architecture | Key Innovation |
| --- | --- | --- | --- |
| Jun 2017 | [Transformer (Google Brain)](https://arxiv.org/abs/1706.03762) | —; Encoder-Decoder Attention | "Attention Is All You Need"; foundational architecture every LLM descends from. |
| Jun 2018 | [GPT-1 (OpenAI)](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) | 117M; Decoder-only Transformer | Introduced the decoder-only pretrain-then-finetune paradigm; showed unsupervised pretraining on text transfers well to downstream tasks. |
| Feb 2019 | [GPT-2 (OpenAI)](https://cdn.openai.com/better-language-models/language-models.pdf) | 1.5B; Decoder-only Transformer | Scaled GPT-1's recipe; sparked mainstream AI awareness through its staged release over safety concerns; open-sourced in full on GitHub. |
| Oct 2019 | [RMSNorm (Zhang & Sennrich)](https://arxiv.org/abs/1910.07467) | —; Normalization layer | Dropped LayerNorm's re-centering step while preserving scaling stability; later became standard in LLaMA-style decoder LLMs. |
| Feb 2020 | [SwiGLU (Shazeer, Google)](https://arxiv.org/abs/2002.05202) | —; Activation function | Replaced ReLU/GeLU in FFN layers with a gated variant; better gradient flow; later adopted by PaLM, LLaMA, and most modern decoder models. |
| May 2020 | [GPT-3 (OpenAI)](https://proceedings.neurips.cc/paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html) | 175B; Dense decoder + pre-norm | Demonstrated emergent few-shot in-context learning at scale; introduced pre-normalization as a training stability technique. |
| Jun 2020 | [GShard (Google)](https://arxiv.org/abs/2006.16668) | 600B; Coarse sparse MoE, top-k routing | First to scale MoE to 600B parameters via top-k routing; established a blueprint later refined by DeepSeek. _Note: Shazeer et al. Jan 2017 Sparse MoE introduced the expert routing concept this builds on._ |
| Apr 2021 | [RoPE (Su et al.)](https://arxiv.org/abs/2104.09864) | —; Positional encoding | Rotary positional embeddings; hybrid absolute/relative encoding that generalizes better to longer sequences than learned absolute positions. |
| Mar 2022 | [Chinchilla (DeepMind)](https://arxiv.org/abs/2203.15556) | 70B; Dense Transformer | Compute-optimal scaling: more tokens on a smaller model beats a bigger undertrained model at the same compute budget. |
| Mar 2022 | [InstructGPT / RLHF (OpenAI)](https://arxiv.org/abs/2203.02155) | 1.3B-175B; RLHF (SFT + RM + PPO) | Introduced the three-stage RLHF pipeline that turned raw pretrained LLMs into usable instruction-following assistants. |
| May 2022 | [FlashAttention (Dao et al.)](https://openreview.net/forum?id=H4DqfPSibmx) | —; IO-aware exact attention | Rewrote attention to be memory-efficient and fast via tiled IO; unlocked practical long-context training and became the de-facto attention kernel. |
| Nov 2022 | [ChatGPT (OpenAI)](https://openai.com/blog/chatgpt) | GPT-3.5 class; RLHF chat assistant | Wrapped an RLHF-tuned LLM in a conversational web interface; proved commercial viability of chat-aligned models and triggered the global generative AI boom. |
| Feb 2023 | [LLaMA 1 (Meta)](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) | 7B-65B; Dense + RoPE / SwiGLU / RMSNorm / FlashAttention | Synthesized the modern decoder recipe and sparked the open-source LLM explosion. |
| Mar 2023 | [GPT-4 (OpenAI, leak)](https://arxiv.org/abs/2303.08774) | ~1.8T total; ~280B active; reported coarse MoE top-2 of 16 | Major capability milestone; architecture figures are leak-reported (Jul 2023), not official. |
| Jul 2023 | [LLaMA 2 (Meta)](https://arxiv.org/abs/2307.09288) | 7B-70B; Dense + GQA | Added RLHF chat alignment; introduced Grouped-Query Attention for faster inference at larger scales. |
| Dec 2023 | [Mixtral 8x7B (Mistral AI, open)](https://arxiv.org/abs/2401.04088) | 46.7B total; 12.9B active; coarse MoE top-2 of 8 | First open-weight MoE with broad adoption; showed sparse routing could beat dense models at equal active-parameter cost. |
| May 2024 | [DeepSeek V2 (DeepSeek)](https://arxiv.org/abs/2405.04434) | 236B total; 21B active; fine-grained MoE + MLA | Introduced MLA (compressed KV cache) and DeepSeekMoE with fine-grained experts and shared expert isolation. |
| Jul 2024 | [LLaMA 3.1 (Meta)](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) | 405B; Dense + 128K context | First open model widely seen as competitive with top closed models at frontier scale. |
| Sep 2024 | [OpenAI o1 (OpenAI)](https://openai.com/o1/) | 671B total; 37B active; MoE + FP8 training + MTP | Shifted emphasis from pre-training scaling to test-time compute scaling; used RL for extended internal reasoning before answers. |
| Dec 2024 | [DeepSeek V3 (DeepSeek, open)](https://arxiv.org/abs/2412.19437) | 671B total; 37B active; MoE + FP8 training + MTP | Auxiliary-loss-free load balancing, FP8 training, and multi-token prediction; redefined efficiency expectations. |
| Jan 2025 | [DeepSeek R1 (DeepSeek, open)](https://arxiv.org/abs/2501.12948) | 671B total; 37B active; V3 arch + pure RL post-training | First open model reported to match o1-level reasoning using pure RL, without SFT warm-up. |
| Jul 2025 | [Kimi K2 (Moonshot AI, open)](https://arxiv.org/abs/2507.20534) | 1T total; 32B active; MoE + MLA + MuonClip | 1T-scale open MoE from Moonshot AI; used MuonClip for training stability at trillion-parameter scale. |
| Apr 2026 | [Kimi K2.6 (Moonshot AI, open)](https://huggingface.co/moonshotai/Kimi-K2.6) | 1T total; 32B active; MoE + MLA + MuonClip, 384 experts, 256K context | Expanded to 384 experts and 256K context; highlighted agent-swarm workflows with up to 300 parallel sub-agents across 4,000 coordinated steps. |

## Building Your Own LLM

### Simplified Hybrid GPT/LLaMA Example (MicroGPT-style)

This is a compact educational architecture that is GPT-2-inspired but includes a few modern LLaMA-like choices.

Reference implementation: [microgpt.py](https://gist.github.com/karpathy/8627fe009c40f57531cb18360106ce95)

### Key Architectural Ideas

- Decoder-only Transformer stack with causal self-attention.
- Token + position embeddings at input.
- Residual connections around attention and MLP sublayers.
- RMSNorm (LLaMA-style) instead of LayerNorm.
- Bias-free linear layers (LLaMA-style simplification).
- ReLU in the MLP for simplicity (instead of GPT-2's GeLU).

### Quick-Start: LLaMA 2 in 4 Lines

For end-to-end training and inference in minimal code:

Reference implementation: [modded-llama2.c](https://github.com/eniompw/modded-llama2.c)

```bash
git clone https://github.com/eniompw/modded-llama2.c
. ./modded-llama2.c/download_tinystories.sh
cd modded-llama2.c && python train.py --max_iters=1
./run out/model.bin -i "Once upon a time "
```

This approach:
- Uses pure C for fast inference with no dependencies.
- Trains on TinyStories dataset for rapid iteration.
- Demonstrates the full LLaMA 2 recipe (RoPE, SwiGLU, RMSNorm, grouped-query attention).
- Ideal for learning how modern LLMs train and infer end-to-end.

### Enhancing MicroGPT for Modern Performance

Reference implementation: [modded MicroGPT](https://github.com/eniompw/microgpt)

To make MicroGPT more performant and aligned with contemporary best practices:

- **FlashAttention 2**: Replace standard attention with tiled IO-aware kernels for 2–4× wall-clock speedup.
- **Grouped-Query Attention (GQA)**: Share KV heads across query heads to reduce memory and accelerate inference bandwidth.
- **Multi-Token Prediction (MTP)**: Train the model to predict multiple tokens per forward pass, increasing sample efficiency.
- **FP8 Training**: Use low-precision compute during training with proper scaling and accumulation for stability.
- **Rotary Embeddings (RoPE)**: Replace learned absolute positions with rotary positional encodings for better length generalization.

These enhancements keep the educational clarity of MicroGPT while moving it closer to production-grade LLaMA 2 / DeepSeek-style architectures.