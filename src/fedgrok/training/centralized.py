"""Training loop: full-batch gradient descent or AdamW, MSE or CE loss."""

import json
import os
import time
import torch
from fedgrok.core.config import Config
from fedgrok.data.modular import make_dataset
from fedgrok.core.registry import build_model, build_loss
from fedgrok.metrics.fourier import (
    weight_norms, gradient_norms, compute_ipr, compute_accuracy,
    fourier_spectrum, fourier_applicable,
)
from fedgrok.core.utils import get_device, make_optimizer


def train(cfg: Config):
    """Run the full training loop. Returns the history dict."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Data
    x_train, y_train, x_test, y_test = make_dataset(cfg)
    p = cfg.p

    # Move data to device, then prepare loss targets there (one-hot for MSE,
    # class indices for CE — build_loss owns that choice).
    x_train = x_train.to(device)
    y_train = y_train.to(device)
    x_test = x_test.to(device)
    y_test = y_test.to(device)

    loss = build_loss(cfg)
    loss_fn = loss.loss_fn
    y_train_target = loss.prepare_target(y_train, p)
    y_test_target = loss.prepare_target(y_test, p)

    # Model (via the registry, so cfg.model selects the architecture)
    model = build_model(cfg).to(device)

    optimizer = make_optimizer(model, cfg)

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
        train_loss_t = loss_fn(out_train, y_train_target)

        optimizer.zero_grad()
        train_loss_t.backward()

        # Log before step (so gradient norms are available)
        if epoch % cfg.log_every == 0:
            with torch.no_grad():
                out_test = model(x_test)
                test_loss = loss_fn(out_test, y_test_target).item()
                train_acc = compute_accuracy(out_train, y_train)
                test_acc = compute_accuracy(out_test, y_test)

            # weight_norms / gradient_norms / IPR are GrokNet-specific (read W1/W2).
            # On other architectures they don't apply — log NaN rather than crash.
            nan = float("nan")
            if fourier_applicable(model):
                wn = weight_norms(model)
                gn = gradient_norms(model)
                ipr_val = compute_ipr(model)["ipr"]
                wn1, wn2 = wn["weight_norm_layer1"], wn["weight_norm_layer2"]
                gn1 = gn.get("grad_norm_layer1", 0.0)
                gn2 = gn.get("grad_norm_layer2", 0.0)
            else:
                wn1 = wn2 = gn1 = gn2 = ipr_val = nan

            history["epoch"].append(epoch)
            history["train_loss"].append(train_loss_t.item())
            history["test_loss"].append(test_loss)
            history["train_acc"].append(train_acc)
            history["test_acc"].append(test_acc)
            history["weight_norm_layer1"].append(wn1)
            history["weight_norm_layer2"].append(wn2)
            history["grad_norm_layer1"].append(gn1)
            history["grad_norm_layer2"].append(gn2)
            history["ipr"].append(ipr_val)

            if epoch % (cfg.log_every * 10) == 0:
                elapsed = time.time() - start
                print(
                    f"[{epoch:>6d}/{cfg.epochs}]  "
                    f"train_loss={train_loss_t.item():.6f}  test_loss={test_loss:.6f}  "
                    f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                    f"ipr={ipr_val:.4f}  ({elapsed:.1f}s)"
                )

        # Save checkpoint if requested. The Fourier spectrum is GrokNet-specific;
        # other models still checkpoint their weights, just without it.
        if cfg.checkpoint_every > 0 and epoch % cfg.checkpoint_every == 0 and epoch > 0:
            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_epoch{epoch}.pt")
            torch.save(model.state_dict(), ckpt_path)
            if fourier_applicable(model):
                spec = fourier_spectrum(model)
                spec_path = os.path.join(ckpt_dir, f"spectrum_epoch{epoch}.pt")
                torch.save(spec["spectrum"], spec_path)

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
