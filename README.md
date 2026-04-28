# LLM Architecture Timeline

Key architectural milestones, 2017 to Apr 2026.

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
| Nov 2022 | ChatGPT (OpenAI) | GPT-3.5 class; RLHF chat assistant | Wrapped an RLHF-tuned LLM in a conversational web interface; proved commercial viability of chat-aligned models and triggered the global generative AI boom. |
| Feb 2023 | [LLaMA 1 (Meta)](https://research.facebook.com/publications/llama-open-and-efficient-foundation-language-models/) | 7B-65B; Dense + RoPE / SwiGLU / RMSNorm / FlashAttention | Synthesized the modern decoder recipe and sparked the open-source LLM explosion. |
| Mar 2023 | GPT-4 (OpenAI, leak) | ~1.8T total; ~280B active; reported coarse MoE top-2 of 16 | Major capability milestone; architecture figures are leak-reported (Jul 2023), not official. |
| Jul 2023 | LLaMA 2 (Meta) | 7B-70B; Dense + GQA | Added RLHF chat alignment; introduced Grouped-Query Attention for faster inference at larger scales. |
| Dec 2023 | [Mixtral 8x7B (Mistral AI, open)](https://arxiv.org/abs/2401.04088) | 46.7B total; 12.9B active; coarse MoE top-2 of 8 | First open-weight MoE with broad adoption; showed sparse routing could beat dense models at equal active-parameter cost. |
| May 2024 | [DeepSeek V2 (DeepSeek)](https://arxiv.org/abs/2405.04434) | 236B total; 21B active; fine-grained MoE + MLA | Introduced MLA (compressed KV cache) and DeepSeekMoE with fine-grained experts and shared expert isolation. |
| Jul 2024 | [LLaMA 3.1 (Meta)](https://ai.meta.com/research/publications/the-llama-3-herd-of-models/) | 405B; Dense + 128K context | First open model widely seen as competitive with top closed models at frontier scale. |
| Sep 2024 | OpenAI o1 (OpenAI) | 671B total; 37B active; MoE + FP8 training + MTP | Shifted emphasis from pre-training scaling to test-time compute scaling; used RL for extended internal reasoning before answers. |
| Dec 2024 | DeepSeek V3 (DeepSeek, open) | 671B total; 37B active; MoE + FP8 training + MTP | Auxiliary-loss-free load balancing, FP8 training, and multi-token prediction; redefined efficiency expectations. |
| Jan 2025 | DeepSeek R1 (DeepSeek, open) | 671B total; 37B active; V3 arch + pure RL post-training | First open model reported to match o1-level reasoning using pure RL, without SFT warm-up. |
| Jul 2025 | Kimi K2 (Moonshot AI, open) | 1T total; 32B active; MoE + MLA + MuonClip | 1T-scale open MoE from Moonshot AI; used MuonClip for training stability at trillion-parameter scale. |
| Apr 2026 | Kimi K2.6 (Moonshot AI, open) | 1T total; 32B active; MoE + MLA + MuonClip, 384 experts, 256K context | Expanded to 384 experts and 256K context; highlighted agent-swarm workflows with up to 300 parallel sub-agents across 4,000 coordinated steps. |

## Simplified Hybrid GPT/LLaMA Example (MicroGPT-style)

This is a compact educational architecture that is GPT-2-inspired but includes a few modern LLaMA-like choices.

Reference implementation: [microgpt.py](https://github.com/eniompw/microgpt/blob/main/microgpt.py)

### Key Architectural Ideas

- Decoder-only Transformer stack with causal self-attention.
- Token + position embeddings at input.
- Residual connections around attention and MLP sublayers.
- RMSNorm (LLaMA-style) instead of LayerNorm.
- Bias-free linear layers (LLaMA-style simplification).
- ReLU in the MLP for simplicity (instead of GPT-2's GeLU).