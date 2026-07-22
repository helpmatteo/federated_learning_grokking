"""Experiment configuration for Gromov (2023) 'Grokking modular arithmetic'."""

from dataclasses import dataclass
from typing import Literal


@dataclass
class Config:
    # --- Modular arithmetic task ---
    p: int = 97                          # modulus (prime)
    task: str = "addition"               # one of: addition, subtraction, division,
                                         #   x2_plus_y2, x_plus_y_squared, multiplication,
                                         #   x2_y2_xy, x3_xy2_y

    # --- Data ---
    alpha: float = 0.5                   # fraction of p^2 dataset used for training
    seed: int = 42                       # random seed for train/test split

    # --- Architecture ---
    model: str = "groknet"               # architecture family (see fedgrok.core.registry)
    hidden_width: int = 100              # N: width of the single hidden layer
    activation: Literal[
        "quadratic", "relu", "gelu",
        "abs", "quartic"
    ] = "quadratic"

    # --- Training ---
    loss: Literal["mse", "ce"] = "mse"   # mse: one-hot targets (Gromov); ce: class indices
    optimizer: Literal["gd", "adamw"] = "gd"
    lr: float = 50.0                     # learning rate (50 for GD, 1e-4 for AdamW)
    weight_decay: float = 0.0            # explicit weight decay
    momentum: float = 0.0               # momentum for GD (0.0 = vanilla GD)
    epochs: int = 10_000                 # number of full-batch epochs
    log_every: int = 100                 # log metrics every N epochs

    # --- Output ---
    output_dir: str = "results/baselines/centralized"
    save_weights: bool = False           # save final model weights
    checkpoint_every: int = 0            # save checkpoints every N epochs (0 = disabled)

    # Set to True internally when user explicitly passes a value
    _lr_set: bool = False
    _wd_set: bool = False
    _epochs_set: bool = False

    def apply_adamw_defaults(self):
        """Apply sensible AdamW defaults if user didn't override them."""
        if self.optimizer == "adamw":
            if not self._lr_set:
                self.lr = 1e-4
            if not self._wd_set:
                self.weight_decay = 1.0
            if not self._epochs_set:
                self.epochs = 5_000
