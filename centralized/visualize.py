"""Plotting functions reproducing Figures 1, 3, 4, 5 from Gromov (2023)."""

import json
import os
import matplotlib.pyplot as plt
import numpy as np


def load_history(path):
    with open(path) as f:
        return json.load(f)


def plot_training_dynamics(history, output_dir="results/centralized", tag=""):
    """Reproduce Figure 1: loss, weight norms, accuracy, gradient norms."""
    epochs = history["epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # (a) Train and test loss
    ax = axes[0, 0]
    ax.plot(epochs, history["train_loss"], label="Train loss")
    ax.plot(epochs, history["test_loss"], label="Test loss")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(a) Train and test loss")
    ax.legend()

    # (b) Weight norms
    ax = axes[0, 1]
    ax.plot(epochs, history["weight_norm_layer1"], label="Weight norm layer 1")
    ax.plot(epochs, history["weight_norm_layer2"], label="Weight norm layer 2")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("(b) Weight norms")
    ax.legend()

    # (c) Train and test accuracy
    ax = axes[1, 0]
    ax.plot(epochs, history["train_acc"], label="Train accuracy")
    ax.plot(epochs, history["test_acc"], label="Test accuracy")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("(c) Train and test accuracy")
    ax.legend()

    # (d) Gradient norms
    ax = axes[1, 1]
    ax.plot(epochs, history["grad_norm_layer1"], label="Grad norm layer 1")
    ax.plot(epochs, history["grad_norm_layer2"], label="Grad norm layer 2")
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("(d) Gradient norms")
    ax.legend()

    fig.suptitle(f"Training Dynamics (Fig. 1){f' — {tag}' if tag else ''}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, f"fig1_dynamics{'_' + tag if tag else ''}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_ipr(history, output_dir="results/centralized", tag=""):
    """Reproduce Figure 5: IPR overlaid with train/test accuracy."""
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(8, 5))

    color_ipr = "tab:purple"
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Inverse Participation Ratio", color=color_ipr)
    ax1.plot(epochs, history["ipr"], color=color_ipr, label="IPR")
    ax1.tick_params(axis="y", labelcolor=color_ipr)

    ax2 = ax1.twinx()
    color_acc = "tab:green"
    ax2.set_ylabel("Accuracy (%)", color=color_acc)
    ax2.plot(epochs, history["train_acc"], color="tab:blue", alpha=0.6, label="Train acc")
    ax2.plot(epochs, history["test_acc"], color="tab:orange", alpha=0.6, label="Test acc")
    ax2.tick_params(axis="y", labelcolor=color_acc)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    fig.suptitle(f"IPR vs Accuracy (Fig. 5){f' — {tag}' if tag else ''}")
    fig.tight_layout()
    path = os.path.join(output_dir, f"fig5_ipr{'_' + tag if tag else ''}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_compare_histories(history_paths, labels, output_dir="results/centralized", title=""):
    """Plot test accuracy curves from multiple runs (for comparing optimizers/widths)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for path, label in zip(history_paths, labels):
        h = load_history(path)
        ax.plot(h["epoch"], h["test_acc"], label=label)
    ax.set_xlabel("Epochs")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title(title or "Test Accuracy Comparison")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(output_dir, "comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
