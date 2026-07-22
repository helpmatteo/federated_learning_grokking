"""1-layer decoder-only transformer for modular arithmetic (Nanda et al. 2023).

The canonical mechanistic-interpretability grokking model: d_model=128, 4 heads
(d_head=32), MLP 512 with ReLU, learned positional embeddings, untied embed /
unembed, and — importantly — NO LayerNorm (Nanda's setup, so the weight-norm
and Fourier story is not confounded by normalisation).

Input convention: to keep one dataset representation across every architecture,
this consumes the same concatenated one-hot the GrokNet does — x is (batch, 2p),
the first p columns the one-hot of operand a, the next p of operand b. A one-hot
times the embedding matrix is exactly an embedding lookup, so no argmax or token
tensor is needed; the two operands become the length-2 sequence the attention
runs over. Trained with cross-entropy over the p output classes.
"""

import math

import torch
import torch.nn as nn


class GrokFormer(nn.Module):
    def __init__(self, p: int, d_model: int = 128, n_heads: int = 4,
                 d_mlp: int = 512):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model {d_model} not divisible by n_heads {n_heads}")
        self.p = p
        self.P = p                      # output classes (mirrors GrokNet.P)
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.seq_len = 2                # (a, b)

        # Token embedding (shared across the two positions) and unembedding.
        # Untied: separate matrices, per Nanda.
        self.W_E = nn.Parameter(torch.randn(p, d_model) / math.sqrt(d_model))
        self.W_pos = nn.Parameter(torch.randn(self.seq_len, d_model) / math.sqrt(d_model))
        self.W_U = nn.Parameter(torch.randn(d_model, p) / math.sqrt(d_model))

        # Single attention block (no biases, no LayerNorm).
        self.W_Q = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_K = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(n_heads, d_model, self.d_head) / math.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(n_heads, self.d_head, d_model) / math.sqrt(d_model))

        # MLP.
        self.W_in = nn.Parameter(torch.randn(d_model, d_mlp) / math.sqrt(d_model))
        self.W_out = nn.Parameter(torch.randn(d_mlp, d_model) / math.sqrt(d_mlp))

    def _embed(self, x):
        """(batch, 2p) one-hot -> (batch, 2, d_model) sequence embeddings."""
        p = self.p
        a_onehot = x[:, :p]
        b_onehot = x[:, p:2 * p]
        emb_a = a_onehot @ self.W_E          # (batch, d_model)
        emb_b = b_onehot @ self.W_E
        seq = torch.stack([emb_a, emb_b], dim=1)   # (batch, 2, d_model)
        return seq + self.W_pos.unsqueeze(0)       # add positional

    def forward(self, x):
        h = self._embed(x)                   # (batch, 2, d_model)

        # Multi-head self-attention. einsum over (batch b, seq s, head h, dim d).
        q = torch.einsum("bsd,hde->bhse", h, self.W_Q)   # (batch, head, seq, d_head)
        k = torch.einsum("bsd,hde->bhse", h, self.W_K)
        v = torch.einsum("bsd,hde->bhse", h, self.W_V)

        scores = torch.einsum("bhse,bhte->bhst", q, k) / math.sqrt(self.d_head)
        attn = scores.softmax(dim=-1)
        z = torch.einsum("bhst,bhte->bhse", attn, v)     # (batch, head, seq, d_head)
        attn_out = torch.einsum("bhse,hed->bsd", z, self.W_O)   # (batch, seq, d_model)

        h = h + attn_out                     # residual (no LayerNorm)

        mlp_hidden = torch.relu(h @ self.W_in)
        mlp_out = mlp_hidden @ self.W_out
        h = h + mlp_out                      # residual

        # Read out from the final position (the "b" slot), unembed to logits.
        logits = h[:, -1, :] @ self.W_U      # (batch, p)
        return logits
