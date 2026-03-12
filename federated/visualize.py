"""Federated learning visualization: comparison plots against centralized baseline."""

import json
import os
import matplotlib.pyplot as plt


def load_history(path):
    with open(path) as f:
        return json.load(f)


def plot_fed_vs_centralized(fed_history, central_history, output_dir="results/federated", tag=""):
    """Compare federated and centralized training dynamics.

    Uses 'total_steps' (rounds * local_epochs) for federated x-axis
    and 'epoch' for centralized, to give a fair compute comparison.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    fed_x = fed_history["total_steps"]
    cen_x = central_history["epoch"]

    # (a) Test accuracy
    ax = axes[0, 0]
    ax.plot(cen_x, central_history["test_acc"], label="Centralized", alpha=0.8)
    ax.plot(fed_x, fed_history["test_acc"], label="Federated", alpha=0.8)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("(a) Test accuracy")
    ax.legend()

    # (b) Train accuracy
    ax = axes[0, 1]
    ax.plot(cen_x, central_history["train_acc"], label="Centralized", alpha=0.8)
    ax.plot(fed_x, fed_history["train_acc"], label="Federated", alpha=0.8)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("Train Accuracy (%)")
    ax.set_title("(b) Train accuracy")
    ax.legend()

    # (c) Test loss
    ax = axes[1, 0]
    ax.plot(cen_x, central_history["test_loss"], label="Centralized", alpha=0.8)
    ax.plot(fed_x, fed_history["test_loss"], label="Federated", alpha=0.8)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("MSE Loss")
    ax.set_title("(c) Test loss")
    ax.legend()

    # (d) IPR
    ax = axes[1, 1]
    ax.plot(cen_x, central_history["ipr"], label="Centralized", alpha=0.8)
    ax.plot(fed_x, fed_history["ipr"], label="Federated", alpha=0.8)
    ax.set_xlabel("Gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("(d) Inverse Participation Ratio")
    ax.legend()

    fig.suptitle(f"Federated vs Centralized{f' — {tag}' if tag else ''}", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, f"fed_vs_central{'_' + tag if tag else ''}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_partition_comparison(histories, labels, output_dir="results/federated"):
    """Compare different partition strategies (IID, operand, target)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Test accuracy
    ax = axes[0]
    for h, label in zip(histories, labels):
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        ax.plot(x, h["test_acc"], label=label)
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy by Partition Strategy")
    ax.legend()

    # IPR
    ax = axes[1]
    for h, label in zip(histories, labels):
        x = h.get("total_steps", h.get("round", range(len(h["ipr"]))))
        ax.plot(x, h["ipr"], label=label)
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("IPR by Partition Strategy")
    ax.legend()

    fig.suptitle("Partition Strategy Comparison", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_partition_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_client_scaling(histories, client_counts, output_dir="results/federated"):
    """Compare grokking dynamics across different numbers of clients."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for h, k in zip(histories, client_counts):
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        ax.plot(x, h["test_acc"], label=f"K={k}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Num Clients")
    ax.legend()

    ax = axes[1]
    for h, k in zip(histories, client_counts):
        x = h.get("total_steps", h.get("round", range(len(h["ipr"]))))
        ax.plot(x, h["ipr"], label=f"K={k}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("IPR vs Num Clients")
    ax.legend()

    fig.suptitle("Client Scaling (IID)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_client_scaling.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_dirichlet_sweep(histories, dirichlet_alphas, output_dir="results/federated"):
    """Compare grokking dynamics across Dirichlet concentration values.

    Lower alpha = more heterogeneous; higher alpha = closer to IID.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for h, da in zip(histories, dirichlet_alphas):
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        ax.plot(x, h["test_acc"], label=f"\u03b1={da}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Dirichlet \u03b1")
    ax.legend()

    ax = axes[1]
    for h, da in zip(histories, dirichlet_alphas):
        x = h.get("total_steps", h.get("round", range(len(h["ipr"]))))
        ax.plot(x, h["ipr"], label=f"\u03b1={da}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("IPR vs Dirichlet \u03b1")
    ax.legend()

    fig.suptitle("Dirichlet Heterogeneity Sweep (lower \u03b1 = more non-IID)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_dirichlet_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_participation_sweep(histories, fractions, partition, output_dir="results/federated"):
    """Compare grokking dynamics across participation rates on a fixed non-IID partition.

    fraction_train=1.0 is the full-participation baseline where non-IID effects cancel.
    Decreasing fraction reveals the true effect of the data heterogeneity.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for h, ft in zip(histories, fractions):
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        ax.plot(x, h["test_acc"], label=f"fraction={ft}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Participation Rate")
    ax.legend()

    ax = axes[1]
    for h, ft in zip(histories, fractions):
        x = h.get("total_steps", h.get("round", range(len(h["ipr"]))))
        ax.plot(x, h["ipr"], label=f"fraction={ft}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("IPR vs Participation Rate")
    ax.legend()

    fig.suptitle(f"Participation Rate Sweep (partition={partition})", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, f"fed_participation_sweep_{partition}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_local_epochs(histories, local_epoch_counts, output_dir="results/federated"):
    """Compare grokking dynamics across different numbers of local epochs (IID)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for h, le in zip(histories, local_epoch_counts):
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        ax.plot(x, h["test_acc"], label=f"local_epochs={le}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("Test Accuracy (%)")
    ax.set_title("Test Accuracy vs Local Epochs")
    ax.legend()

    ax = axes[1]
    for h, le in zip(histories, local_epoch_counts):
        x = h.get("total_steps", h.get("round", range(len(h["ipr"]))))
        ax.plot(x, h["ipr"], label=f"local_epochs={le}")
    ax.set_xlabel("Total gradient steps")
    ax.set_ylabel("IPR")
    ax.set_title("IPR vs Local Epochs")
    ax.legend()

    fig.suptitle("Local Epochs Sweep (IID)", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_local_epochs_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def plot_breaking_point(results, output_dir="results/federated"):
    """Combined 3-row figure showing where federated grokking breaks.

    results: dict with keys 'clients', 'participation', 'local_epochs',
             each containing {'histories': [...], 'labels': [...]}
    """
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))

    # Row 0: Client scaling (IID, full participation)
    if "clients" in results:
        data = results["clients"]
        for h, label in zip(data["histories"], data["labels"]):
            x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
            axes[0, 0].plot(x, h["test_acc"], label=label, alpha=0.8)
            axes[0, 1].plot(x, h["ipr"], label=label, alpha=0.8)
            axes[0, 2].plot(x, h["train_acc"], label=label, alpha=0.8)
        axes[0, 0].set_title("Test Accuracy")
        axes[0, 1].set_title("IPR")
        axes[0, 2].set_title("Train Accuracy")
        for ax in axes[0]:
            ax.set_xlabel("Total gradient steps")
            ax.legend(fontsize=8)
        axes[0, 0].set_ylabel("Client Scaling\n(IID, ft=1.0)\n\nAccuracy (%)")
        axes[0, 1].set_ylabel("IPR")
        axes[0, 2].set_ylabel("Accuracy (%)")

    # Row 1: Partial participation (K=20, target partition)
    if "participation" in results:
        data = results["participation"]
        for h, label in zip(data["histories"], data["labels"]):
            x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
            axes[1, 0].plot(x, h["test_acc"], label=label, alpha=0.8)
            axes[1, 1].plot(x, h["ipr"], label=label, alpha=0.8)
            axes[1, 2].plot(x, h["train_acc"], label=label, alpha=0.8)
        for ax in axes[1]:
            ax.set_xlabel("Total gradient steps")
            ax.legend(fontsize=8)
        axes[1, 0].set_ylabel("Partial Participation\n(K=20, target)\n\nAccuracy (%)")
        axes[1, 1].set_ylabel("IPR")
        axes[1, 2].set_ylabel("Accuracy (%)")

    # Row 2: Extreme local epochs (K=5, IID)
    if "local_epochs" in results:
        data = results["local_epochs"]
        for h, label in zip(data["histories"], data["labels"]):
            x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
            axes[2, 0].plot(x, h["test_acc"], label=label, alpha=0.8)
            axes[2, 1].plot(x, h["ipr"], label=label, alpha=0.8)
            axes[2, 2].plot(x, h["train_acc"], label=label, alpha=0.8)
        for ax in axes[2]:
            ax.set_xlabel("Total gradient steps")
            ax.legend(fontsize=8)
        axes[2, 0].set_ylabel("Extreme Local Epochs\n(K=5, IID)\n\nAccuracy (%)")
        axes[2, 1].set_ylabel("IPR")
        axes[2, 2].set_ylabel("Accuracy (%)")

    fig.suptitle("Breaking Point: Where Does Federated Grokking Fail?", fontsize=16, y=1.01)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_breaking_point.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path}")


def plot_fedprox_sweep(histories, mu_values, output_dir="results/federated"):
    """Compare FedAvg (mu=0) vs FedProx across proximal strength values.

    Tests whether the proximal regularization that reduces client drift
    also suppresses the weight growth needed for grokking.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    for h, mu in zip(histories, mu_values):
        label = "FedAvg" if mu == 0 else f"\u03bc={mu}"
        x = h.get("total_steps", h.get("round", range(len(h["test_acc"]))))
        axes[0].plot(x, h["test_acc"], label=label, alpha=0.8)
        axes[1].plot(x, h["ipr"], label=label, alpha=0.8)
        axes[2].plot(x, h["weight_norm_layer1"], label=label, alpha=0.8)

    axes[0].set_ylabel("Test Accuracy (%)")
    axes[0].set_title("Test Accuracy")
    axes[1].set_ylabel("IPR")
    axes[1].set_title("Inverse Participation Ratio")
    axes[2].set_ylabel("||W1||_F")
    axes[2].set_title("Weight Norm (Layer 1)")

    for ax in axes:
        ax.set_xlabel("Total gradient steps")
        ax.legend(fontsize=8)

    fig.suptitle("FedProx: Does Proximal Regularization Kill Grokking?", fontsize=14)
    fig.tight_layout()
    path = os.path.join(output_dir, "fed_fedprox_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")
