"""Training loop: full-batch gradient descent or AdamW with MSE loss."""

import json
import os
import time
import torch
import torch.nn as nn
from core.config import Config
from core.dataset import make_dataset
from core.model import GrokNet
from core.metrics import weight_norms, gradient_norms, compute_ipr, compute_accuracy
from core.utils import get_device, make_optimizer, make_targets_onehot


def train(cfg: Config):
    """Run the full training loop. Returns the history dict."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data (build one-hots on CPU, then move everything to device)
    x_train, y_train, x_test, y_test = make_dataset(cfg)
    p = cfg.p
    y_train_oh = make_targets_onehot(y_train, p)
    y_test_oh = make_targets_onehot(y_test, p)

    # Move data to device
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    y_train_oh = y_train_oh.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    y_test_oh = y_test_oh.to(device)

    # Model
    model = GrokNet(
        input_dim=2 * p,
        hidden_width=cfg.hidden_width,
        output_dim=p,
        activation=cfg.activation,
    ).to(device)

    optimizer = make_optimizer(model, cfg)
    loss_fn = nn.MSELoss()

    # History
    history = {
        "epoch": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "grad_norm_layer1": [], "grad_norm_layer2": [],
        "ipr": [],
    }

    start = time.time()

    for epoch in range(cfg.epochs + 1):
        # ── Forward + backward ──────────────────────────────────────────
        model.train()
        out_train = model(x_train)
        loss = loss_fn(out_train, y_train_oh)

        optimizer.zero_grad()
        loss.backward()

        # Log before step (so gradient norms are available)
        if epoch % cfg.log_every == 0:
            with torch.no_grad():
                out_test = model(x_test)
                test_loss = loss_fn(out_test, y_test_oh).item()
                train_acc = compute_accuracy(out_train, y_train)
                test_acc = compute_accuracy(out_test, y_test)

            wn = weight_norms(model)
            gn = gradient_norms(model)
            ipr = compute_ipr(model)

            history["epoch"].append(epoch)
            history["train_loss"].append(loss.item())
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["weight_norm_layer1"].append(wn["weight_norm_layer1"])
            history["weight_norm_layer2"].append(wn["weight_norm_layer2"])
            history["grad_norm_layer1"].append(gn.get("grad_norm_layer1", 0.0))
            history["grad_norm_layer2"].append(gn.get("grad_norm_layer2", 0.0))
            history["ipr"].append(ipr["ipr"])

            if epoch % (cfg.log_every * 10) == 0:
                elapsed = time.time() - start
                print(
                    f"[{epoch:>6d}/{cfg.epochs}]  "
                    f"train_loss={loss.item():.6f}  test_loss={test_loss:.6f}  "
                    f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                    f"ipr={ipr['ipr']:.4f}  ({elapsed:.1f}s)"
                )

        optimizer.step()

    # Save results
    os.makedirs(cfg.output_dir, exist_ok=True)
    tag = f"{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}_a{cfg.alpha}_s{cfg.seed}"
    history_path = os.path.join(cfg.output_dir, f"history_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    if cfg.save_weights:
        weights_path = os.path.join(cfg.output_dir, f"weights_{tag}.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

    return history, model
