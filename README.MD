# Lightweight Transformer from Scratch (PyTorch)

A modular, zero-framework implementation of a Decoder-only Transformer Language Model built from scratch in PyTorch.

## Architecture Features
- **Multi-Head Self-Attention** with scaled dot-product matrix operations and causal masking.
- **Pre-Layer Normalization (Pre-LN)** design pattern for numerical stability.
- **Sinusoidal / Learned Position Embeddings** tied with token embeddings.
- **Weight Tying** between token embedding matrix and output projection head.

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt