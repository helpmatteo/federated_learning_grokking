"""Federated Averaging via Flower simulation for grokking experiments.

Uses Flower's run_simulation with:
  - Custom NumPyClient that does local training on its partition
  - FedAvg strategy with centralized evaluation (evaluate_fn)
  - on_fit_config_fn to pass hyperparameters to clients

Each client reconstructs its data partition from the config + partition-id
(dataset is tiny so this is fast). The server evaluates the global model
on the full test set after each round.
"""

import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg
from flwr.server.strategy import FedAdam
from flwr.simulation import run_simulation

from federated.config import FedConfig
from federated.dataset import make_federated_datasets
from core.model import GrokNet
from core.utils import make_targets_onehot, make_optimizer, get_device
from core.metrics import weight_norms, compute_ipr, compute_accuracy, fourier_spectrum


# ── Dataset cache ────────────────────────────────────────────────────────────
# Avoids rebuilding the dataset on every client fit() call. Each unique config
# produces one cached entry; clients and server share the same CPU tensors.

_dataset_cache = {}


def _get_cached_datasets(cfg):
    """Return federated datasets, caching by config to avoid redundant work."""
    cache_key = (cfg.p, cfg.task, cfg.alpha, cfg.seed, cfg.num_clients, cfg.partition,
                 getattr(cfg, 'dirichlet_alpha', None))
    if cache_key not in _dataset_cache:
        _dataset_cache[cache_key] = make_federated_datasets(cfg)
    return _dataset_cache[cache_key]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_to_ndarrays(model):
    """Extract model weights as list of numpy arrays (always on CPU)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _ndarrays_to_state_dict(ndarrays, model):
    """Convert numpy arrays back to a state dict matching model's keys."""
    keys = list(model.state_dict().keys())
    return OrderedDict({k: torch.from_numpy(v) for k, v in zip(keys, ndarrays)})


def _make_model(cfg):
    """Create a fresh GrokNet from config (on CPU)."""
    return GrokNet(
        input_dim=2 * cfg.p,
        hidden_width=cfg.hidden_width,
        output_dim=cfg.p,
        activation=cfg.activation,
    )


def _cfg_to_fit_config(cfg: FedConfig, server_round: int):
    """Serialize config into a dict that Flower can send to clients."""
    return {
        "server_round": server_round,
        "p": cfg.p,
        "task": cfg.task,
        "alpha": cfg.alpha,
        "seed": cfg.seed,
        "num_clients": cfg.num_clients,
        "partition": cfg.partition,
        "dirichlet_alpha": cfg.dirichlet_alpha,
        "local_epochs": cfg.local_epochs,
        "lr": cfg.lr,
        "optimizer": cfg.optimizer,
        "weight_decay": cfg.weight_decay,
        "momentum": cfg.momentum,
        "hidden_width": cfg.hidden_width,
        "activation": cfg.activation,
        "proximal_mu": cfg.proximal_mu,
        "strategy": cfg.strategy,
        "server_lr": cfg.server_lr,
        "tau": cfg.tau,
        "track_client_drift": cfg.track_client_drift,
        "checkpoint_client_weights": cfg.checkpoint_client_weights,
    }


def _fit_config_to_cfg(config: dict) -> FedConfig:
    """Reconstruct FedConfig from the dict sent by the server."""
    return FedConfig(
        p=int(config["p"]),
        task=config["task"],
        alpha=float(config["alpha"]),
        seed=int(config["seed"]),
        num_clients=int(config["num_clients"]),
        partition=config["partition"],
        dirichlet_alpha=float(config["dirichlet_alpha"]),
        local_epochs=int(config["local_epochs"]),
        lr=float(config["lr"]),
        optimizer=config["optimizer"],
        weight_decay=float(config["weight_decay"]),
        momentum=float(config["momentum"]),
        hidden_width=int(config["hidden_width"]),
        activation=config["activation"],
        proximal_mu=float(config.get("proximal_mu", 0.0)),
        strategy=config.get("strategy", "fedavg"),
        server_lr=float(config.get("server_lr", 1.0)),
        tau=float(config.get("tau", 1e-3)),
        track_client_drift=bool(config.get("track_client_drift", True)),
    )


def compute_drift(w_before: list, w_after: list) -> float:
    """Compute Frobenius norm of weight difference: ||w_after - w_before||_F."""
    total = 0.0
    for wb, wa in zip(w_before, w_after):
        diff = wa - wb
        total += float(np.sum(diff ** 2))
    return float(np.sqrt(total))


# ── Flower Client ────────────────────────────────────────────────────────────

class GrokClient(NumPyClient):
    """Flower client that trains GrokNet on a local data partition."""

    def __init__(self, partition_id: int):
        self.partition_id = partition_id

    def fit(self, parameters, config):
        cfg = _fit_config_to_cfg(config)
        device = get_device()
        p = cfg.p

        # Load this client's data partition (cached across rounds)
        client_data, _, _, _, _ = _get_cached_datasets(cfg)
        x_local, y_local = client_data[self.partition_id]
        y_local_oh = make_targets_onehot(y_local, p)

        # Move data to device
        x_local = x_local.to(device)
        y_local_oh = y_local_oh.to(device)
        y_local = y_local.to(device)

        # Build model on CPU, load global weights, then move to device
        model = _make_model(cfg)
        state_dict = _ndarrays_to_state_dict(parameters, model)
        model.load_state_dict(state_dict)
        model.to(device)

        # NOTE: Optimizer state (momentum, Adam moments) is not preserved across
        # rounds. For SGD with momentum=0 this is fine. For AdamW, adaptive
        # estimates restart each round — a known limitation of vanilla FedAvg.
        optimizer = make_optimizer(model, cfg)
        loss_fn = nn.MSELoss()

        # FedProx: save global weights for proximal term
        proximal_mu = cfg.proximal_mu
        if proximal_mu > 0:
            global_params = [p.detach().clone() for p in model.parameters()]

        model.train()
        for _ in range(cfg.local_epochs):
            out = model(x_local)
            loss = loss_fn(out, y_local_oh)
            # FedProx proximal term: (mu/2) * ||w - w_global||^2
            if proximal_mu > 0:
                prox = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Return updated weights (moved to CPU) and metrics
        model.eval()
        with torch.no_grad():
            out = model(x_local)
            local_loss = loss_fn(out, y_local_oh).item()
            local_acc = compute_accuracy(out, y_local)

        updated_weights = _model_to_ndarrays(model)
        drift = compute_drift(parameters, updated_weights)
        weight_norm = float(sum(np.sum(w**2) for w in updated_weights) ** 0.5)
        local_ipr = compute_ipr(model)["ipr"]

        # Extract W1 for per-client Fourier analysis (only first-operand columns)
        # Serialize as bytes since Flower metrics only support Scalar types
        if config.get("checkpoint_client_weights", False):
            client_w1_bytes = model.W1.data[:, :cfg.p].cpu().numpy().tobytes()
        else:
            client_w1_bytes = None

        metrics_dict = {"loss": local_loss, "accuracy": local_acc, "drift": drift,
             "weight_norm": weight_norm, "ipr": local_ipr}
        if client_w1_bytes is not None:
            metrics_dict["w1_first_p"] = client_w1_bytes
            metrics_dict["w1_shape_0"] = model.W1.data.shape[0]
            metrics_dict["w1_shape_1"] = cfg.p

        return (
            updated_weights,
            len(y_local),
            metrics_dict,
        )

    def evaluate(self, parameters, config):
        # Server-side evaluation handles global metrics; skip client eval
        return 0.0, 0, {}


# ── Strategy builder ─────────────────────────────────────────────────────────

def _build_strategy(cfg, init_params, evaluate_fn, fit_metrics_aggregation_fn=None):
    """Build Flower strategy based on config.strategy field."""
    common_kwargs = dict(
        fraction_fit=cfg.fraction_train,
        fraction_evaluate=0.0,
        min_fit_clients=max(1, int(cfg.num_clients * cfg.fraction_train)),
        min_available_clients=cfg.num_clients,
        initial_parameters=init_params,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=lambda rnd: _cfg_to_fit_config(cfg, rnd),
    )
    if fit_metrics_aggregation_fn is not None:
        common_kwargs["fit_metrics_aggregation_fn"] = fit_metrics_aggregation_fn

    if cfg.strategy == "fedadam":
        return FedAdam(
            **common_kwargs,
            eta=cfg.server_lr,
            tau=cfg.tau,
        )
    else:
        # Both "fedavg" and "fedprox" use FedAvg strategy;
        # FedProx proximal term is applied client-side in GrokClient.fit()
        return FedAvg(**common_kwargs)


# ── Flower Server + Evaluation ───────────────────────────────────────────────

def fed_train(cfg: FedConfig):
    """Run FedAvg via Flower simulation. Returns history dict and final model."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Precompute global data (single call, also populates cache for clients)
    client_data, x_train_full, y_train_full, x_test, y_test = _get_cached_datasets(cfg)
    y_test_oh = make_targets_onehot(y_test, cfg.p)
    y_train_full_oh = make_targets_onehot(y_train_full, cfg.p)

    # Move evaluation data to device
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    y_test_oh = y_test_oh.to(device)
    x_train_full = x_train_full.to(device)
    y_train_full = y_train_full.to(device)
    y_train_full_oh = y_train_full_oh.to(device)

    # Print partition sizes
    print(f"Clients: {cfg.num_clients}, partition: {cfg.partition}, "
          f"samples per client: {[len(y) for _, y in client_data]}")

    # History (captured by closure in evaluate_fn)
    history = {
        "round": [], "total_steps": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "ipr": [],
        "mean_client_drift": [],
        "client_weight_divergence": [],
    }

    # Mutable container to capture final model parameters from evaluate_fn
    _final_ndarrays = [None]

    loss_fn = nn.MSELoss()
    start_time = time.time()

    # Mutable container for inter-callback communication (closure-shared)
    _round_metrics = {"mean_drift": 0.0, "weight_divergence": 0.0}
    _client_w1_cache = {"data": None}

    def _aggregate_fit_metrics(metrics_list):
        """Aggregate per-client fit metrics from GrokClient.fit()."""
        drifts = [m.get("drift", 0.0) for _, m in metrics_list]
        w_norms = [m.get("weight_norm", 0.0) for _, m in metrics_list]

        _round_metrics["mean_drift"] = float(np.mean(drifts)) if drifts else 0.0
        _round_metrics["weight_divergence"] = float(np.std(w_norms)) if len(w_norms) > 1 else 0.0

        # Capture per-client W1 if available (deserialize from bytes)
        w1_list = []
        for _, m in metrics_list:
            w1_bytes = m.get("w1_first_p")
            if w1_bytes is not None:
                shape = (int(m["w1_shape_0"]), int(m["w1_shape_1"]))
                w1_list.append(np.frombuffer(w1_bytes, dtype=np.float32).reshape(shape))
        _client_w1_cache["data"] = w1_list if w1_list else None

        return {
            "mean_drift": _round_metrics["mean_drift"],
            "weight_divergence": _round_metrics["weight_divergence"],
        }

    def evaluate_fn(server_round, parameters, config):
        """Centralized evaluation after each aggregation round."""
        _final_ndarrays[0] = parameters  # capture for final model reconstruction

        model = _make_model(cfg)
        state_dict = _ndarrays_to_state_dict(parameters, model)
        model.load_state_dict(state_dict)
        model.to(device)

        model.eval()
        with torch.no_grad():
            out_test = model(x_test)
            test_loss = loss_fn(out_test, y_test_oh).item()
            test_acc = compute_accuracy(out_test, y_test)
            out_train = model(x_train_full)
            train_loss = loss_fn(out_train, y_train_full_oh).item()
            train_acc = compute_accuracy(out_train, y_train_full)

        wn = weight_norms(model)
        ipr = compute_ipr(model)

        history["round"].append(server_round)
        # total_steps = rounds * local_epochs; counts per-model gradient steps
        # (accurate when fraction_train=1.0; approximate otherwise)
        history["total_steps"].append(server_round * cfg.local_epochs)
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["weight_norm_layer1"].append(wn["weight_norm_layer1"])
        history["weight_norm_layer2"].append(wn["weight_norm_layer2"])
        history["ipr"].append(ipr["ipr"])
        history["mean_client_drift"].append(_round_metrics["mean_drift"])
        history["client_weight_divergence"].append(_round_metrics["weight_divergence"])

        # Save checkpoint if requested
        if cfg.checkpoint_every > 0 and server_round > 0 and server_round % cfg.checkpoint_every == 0:
            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Global model checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_round{server_round}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # Global Fourier spectrum
            spec = fourier_spectrum(model)
            spec_path = os.path.join(ckpt_dir, f"spectrum_round{server_round}.pt")
            torch.save(spec["spectrum"], spec_path)

            # Per-client W1 (if available)
            if _client_w1_cache["data"] is not None:
                client_path = os.path.join(ckpt_dir, f"client_w1_round{server_round}.pt")
                torch.save(_client_w1_cache["data"], client_path)

        if server_round % max(1, cfg.num_rounds // 10) == 0:
            elapsed = time.time() - start_time
            print(
                f"[Round {server_round:>4d}/{cfg.num_rounds}]  "
                f"train_loss={train_loss:.6f}  test_loss={test_loss:.6f}  "
                f"train_acc={train_acc:.1f}%  test_acc={test_acc:.1f}%  "
                f"ipr={ipr['ipr']:.4f}  ({elapsed:.1f}s)"
            )

        return test_loss, {"test_accuracy": test_acc, "train_accuracy": train_acc}

    # Initial model for FedAvg
    init_model = _make_model(cfg)
    init_params = ndarrays_to_parameters(_model_to_ndarrays(init_model))

    # ── Build Flower apps ────────────────────────────────────────────────

    def client_fn(context: Context):
        partition_id = context.node_config["partition-id"]
        return GrokClient(partition_id=partition_id).to_client()

    # Capture cfg in closure for on_fit_config_fn
    fed_cfg = cfg

    def server_fn(context: Context):
        strategy = _build_strategy(fed_cfg, init_params, evaluate_fn,
                                   fit_metrics_aggregation_fn=_aggregate_fit_metrics)
        return ServerAppComponents(
            strategy=strategy,
            config=ServerConfig(num_rounds=fed_cfg.num_rounds),
        )

    client_app = ClientApp(client_fn=client_fn)
    server_app = ServerApp(server_fn=server_fn)

    # ── Run simulation ───────────────────────────────────────────────────
    prox_info = f", proximal_mu={cfg.proximal_mu}" if cfg.proximal_mu > 0 else ""
    print(f"\nStarting Flower simulation: {cfg.num_rounds} rounds, "
          f"{cfg.num_clients} clients, {cfg.local_epochs} local epochs{prox_info}\n")

    # Allocate fractional CUDA GPUs across clients; MPS is used via PyTorch
    # directly (not managed by Ray), so num_gpus stays 0 for MPS.
    num_gpus = 0.0
    if torch.cuda.is_available():
        num_gpus = 1.0 / cfg.num_clients

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config={"client_resources": {"num_cpus": 1, "num_gpus": num_gpus}},
    )

    # ── Reconstruct final trained model ──────────────────────────────────
    final_model = _make_model(cfg)
    if _final_ndarrays[0] is not None:
        state_dict = _ndarrays_to_state_dict(_final_ndarrays[0], final_model)
        final_model.load_state_dict(state_dict)
    final_model.to(device)

    # ── Save results ─────────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    dirichlet_suffix = f"_dir{cfg.dirichlet_alpha}" if cfg.partition == "dirichlet" else ""
    prox_suffix = f"_mu{cfg.proximal_mu}" if cfg.proximal_mu > 0 else ""
    tag = (f"fed_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}"
           f"_a{cfg.alpha}_K{cfg.num_clients}_le{cfg.local_epochs}"
           f"_ft{cfg.fraction_train}_{cfg.partition}{dirichlet_suffix}"
           f"{prox_suffix}_s{cfg.seed}")
    history_path = os.path.join(cfg.output_dir, f"history_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    if cfg.save_weights:
        weights_path = os.path.join(cfg.output_dir, f"weights_{tag}.pt")
        torch.save(final_model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

    return history, final_model
