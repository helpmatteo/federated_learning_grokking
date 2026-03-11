"""Two-layer MLP for grokking modular arithmetic (Gromov 2023, Eqs 1-2).

Architecture (mean-field parametrization, scaling absorbed into init):
    h_k^(1)(x) = sum_j W_kj^(1) * x_j       (W1 init ~ N(0, 1/D))
    z_k^(1)(x) = phi(h_k^(1)(x))
    h_q^(2)(x) = sum_k W_qk^(2) * z_k^(1)(x) (W2 init ~ N(0, 1/N^2))

This is equivalent to the paper's Eqs (1)-(2) where sqrt(1/D) and 1/N scaling
appear in the forward pass with W ~ N(0,1).  Absorbing the scaling into the
initialization produces identical outputs at init but allows standard learning
rates to work without requiring lr ~ O(N * sqrt(D)).

No biases.
"""

import math
import torch
import torch.nn as nn


class GrokNet(nn.Module):
    def __init__(self, input_dim: int, hidden_width: int, output_dim: int,
                 activation: str = "quadratic"):
        super().__init__()
        self.D = input_dim       # 2p
        self.N = hidden_width    # N
        self.P = output_dim      # p

        # Mean-field init: equivalent to W_paper ~ N(0,1) with sqrt(1/D), 1/N in forward
        self.W1 = nn.Parameter(torch.randn(self.N, self.D) / math.sqrt(self.D))
        self.W2 = nn.Parameter(torch.randn(self.P, self.N) / self.N)

        self.activation = activation

    def _activate(self, h):
        if self.activation == "quadratic":
            return h ** 2
        elif self.activation == "relu":
            return torch.relu(h)
        elif self.activation == "gelu":
            return torch.nn.functional.gelu(h)
        elif self.activation == "abs":
            return torch.abs(h)
        elif self.activation == "quartic":
            return h ** 4
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def forward(self, x):
        # x: (batch, D)
        h1 = x @ self.W1.t()        # (batch, N)  — preactivations layer 1
        z1 = self._activate(h1)      # (batch, N)  — activations layer 1
        h2 = z1 @ self.W2.t()        # (batch, P)  — output (logits)
        return h2
