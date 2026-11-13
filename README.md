# Transformer from Scratch

A learning project: building the transformer architecture from the ground up in PyTorch, one piece at a time, to understand how it actually works.

## Why

The transformer isn't only for language models. The same core building blocks (attention, feed-forward layers, residual connections, normalization) power models across most of modern machine learning:

- **Language models**: GPT-style next-token prediction
- **Image classification**: Vision Transformers (ViT), which treat an image as a sequence of patches
- **Reinforcement learning**: e.g. Decision Transformers, which model trajectories as sequences
- **Diffusion**: e.g. Diffusion Transformers (DiT) as the denoising backbone

The goal is to understand the architecture well enough to apply it to any of these. The plan is to build each component by hand instead of reaching for `nn.Transformer` or `nn.MultiheadAttention`, and to check each one against PyTorch's reference implementations where possible.

## Progress

**Core architecture**

- [x] Single-head scaled dot-product attention (`src/attn/single_head.py`)
- [x] Multi-head attention (`src/attn/multi_head.py`)
- [ ] Causal masking
- [ ] Positional encodings
- [ ] Feed-forward block
- [ ] Layer norm and residual connections
- [ ] Full transformer block (encoder / decoder)

**Applications**

- [ ] Language model
- [ ] Image classification
- [ ] Reinforcement learning
- [ ] Diffusion

## Structure

```
src/
  attn/
    single_head.py   # one attention head: softmax(QKᵀ / √d) V
    multi_head.py    # several heads run in parallel, concatenated, then projected
```

## Setup

The only dependency so far is [PyTorch](https://pytorch.org/get-started/locally/).

```bash
pip install torch
```

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (2017)
