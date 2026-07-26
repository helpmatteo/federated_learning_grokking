"""Federated Averaging via Flower simulation for grokking experiments.

Uses Flower's run_simulation with:
  - Custom NumPyClient that does local training on its partition
  - FedAvg strategy with centralized evaluation (evaluate_fn)
  - on_fit_config_fn to pass hyperparameters to clients

Each client reconstructs its data partition from the config + partition-id
(dataset is tiny so this is fast). The server evaluates the global model
on the full test set after each round.
"""

import dataclasses
import json
import os
import time
from collections import OrderedDict

import numpy as np
import torch

import flwr as fl
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerConfig, ServerAppComponents
from flwr.server.strategy import FedAvg, FedAdam, FedAvgM, FedYogi
from flwr.simulation import run_simulation

from fedgrok.core.fed_config import FedConfig
from fedgrok.data.partition import make_federated_datasets
from fedgrok.core.registry import build_model, build_loss
from fedgrok.data.registry import dataset_dims
from fedgrok.core.utils import make_optimizer, get_device
from fedgrok.metrics.fourier import (
    weight_norms, compute_ipr, compute_accuracy, fourier_spectrum, fourier_applicable,
)


# ── Dataset cache ────────────────────────────────────────────────────────────
# Avoids rebuilding the dataset on every client fit() call. Each unique config
# produces one cached entry; clients and server share the same CPU tensors.

_dataset_cache = {}


def _dataset_cache_key(cfg):
    return (cfg.p, cfg.task, cfg.alpha, cfg.seed, cfg.num_clients, cfg.partition,
            getattr(cfg, 'dirichlet_alpha', None))


def _get_cached_datasets(cfg):
    """Return federated datasets, caching by config to avoid redundant work."""
    cache_key = _dataset_cache_key(cfg)
    if cache_key not in _dataset_cache:
        _dataset_cache[cache_key] = make_federated_datasets(cfg)
    return _dataset_cache[cache_key]


# ── Per-client warm state ────────────────────────────────────────────────────
# fit() runs once per client per round. Rebuilding the model, moving it to the
# device, and re-copying the client's data to the device every round is pure
# overhead — the data never changes and the model's shape never changes; only
# its parameter *values* change each round. In Flower's simulation each client
# is a Ray actor (a persistent process), so a module-level dict keyed by the
# client's identity survives across that client's rounds — the same mechanism
# _dataset_cache relies on. We cache the on-device model and data tensors and,
# each round, copy the incoming global parameters into the model in place.

_client_cache = {}


def _get_warm_client(cfg, partition_id, device):
    """Return a cached (model, x_local, y_local_oh, y_local) for this client.

    Built once per (dataset, client) and reused across rounds. The model's
    parameter values are overwritten with the round's global weights by the
    caller, so its initialisation is irrelevant after the first round.
    """
    key = (_dataset_cache_key(cfg), partition_id)
    entry = _client_cache.get(key)
    if entry is None:
        client_data, _, _, _, _ = _get_cached_datasets(cfg)
        x_local, y_local = client_data[partition_id]
        x_local = x_local.to(device)
        y_local = y_local.to(device)
        # Loss target: one-hot for MSE, class indices for CE (build_loss owns it).
        y_local_target = build_loss(cfg).prepare_target(y_local, dataset_dims(cfg)[1])

        model = _make_model(cfg).to(device)
        entry = (model, x_local, y_local_target, y_local)
        _client_cache[key] = entry
    return entry


def _load_ndarrays_into(model, ndarrays):
    """Copy a list of parameter ndarrays into `model` in place, on its device.

    Equivalent to load_state_dict(ndarrays) but without reallocating tensors or
    moving anything across devices; the order matches _model_to_ndarrays.
    """
    with torch.no_grad():
        for param, arr in zip(model.state_dict().values(), ndarrays):
            param.copy_(torch.from_numpy(arr).to(param.device))


# ── Helpers ──────────────────────────────────────────────────────────────────

def _model_to_ndarrays(model):
    """Extract model weights as list of numpy arrays (always on CPU)."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


def _ndarrays_to_state_dict(ndarrays, model):
    """Convert numpy arrays back to a state dict matching model's keys."""
    keys = list(model.state_dict().keys())
    return OrderedDict({k: torch.from_numpy(v) for k, v in zip(keys, ndarrays)})


def _make_model(cfg):
    """Create a fresh model from config (on CPU), via the model registry."""
    return build_model(cfg)


# Field names of FedConfig, computed once. Every field is a Flower Scalar
# (int/float/str/bool), so the whole config round-trips through the fit-config
# dict without hand-listing fields — which is what previously dropped loss,
# model and batch_size and silently mis-trained clients.
_FEDCONFIG_FIELDS = {f.name for f in dataclasses.fields(FedConfig)}


def _cfg_to_fit_config(cfg: FedConfig, server_round: int):
    """Serialize the whole config into a dict Flower can send to clients.

    Generated from the dataclass fields so no field can be silently dropped.
    """
    config = dataclasses.asdict(cfg)
    config["server_round"] = server_round
    return config


def _fit_config_to_cfg(config: dict) -> FedConfig:
    """Reconstruct FedConfig from the dict sent by the server."""
    kwargs = {k: v for k, v in config.items() if k in _FEDCONFIG_FIELDS}
    return FedConfig(**kwargs)


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

        # Warm model + on-device data (built once per client, reused each round).
        # y_local_target is one-hot (MSE) or class indices (CE) per the loss.
        model, x_local, y_local_target, y_local = _get_warm_client(cfg, self.partition_id, device)

        # Overwrite the model with this round's global weights, in place.
        _load_ndarrays_into(model, parameters)

        # NOTE: Optimizer state (momentum, Adam moments) is intentionally NOT
        # preserved across rounds — a fresh optimizer restarts it, which is the
        # standard FedAvg/FedProx semantics. For SGD momentum=0 it is a no-op;
        # for AdamW the adaptive estimates restart each round.
        optimizer = make_optimizer(model, cfg)
        loss_fn = build_loss(cfg).loss_fn

        # FedProx: snapshot the global weights for the proximal term.
        proximal_mu = cfg.proximal_mu
        if proximal_mu > 0:
            global_params = [w.detach().clone() for w in model.parameters()]

        def _step(xb, yb):
            """One gradient step on a batch, with the FedProx term if enabled."""
            loss = loss_fn(model(xb), yb)
            if proximal_mu > 0:
                prox = sum(
                    torch.sum((p - gp) ** 2)
                    for p, gp in zip(model.parameters(), global_params)
                )
                loss = loss + (proximal_mu / 2.0) * prox
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train()
        bs = cfg.batch_size
        n_local = x_local.shape[0]
        for _ in range(cfg.local_epochs):
            if bs and bs > 0:
                # A local epoch is a shuffled minibatch pass. randperm is only
                # reached here, so the full-batch path stays RNG-identical.
                perm = torch.randperm(n_local, device=device)
                for i in range(0, n_local, bs):
                    idx = perm[i:i + bs]
                    _step(x_local[idx], y_local_target[idx])
            else:
                # Full-batch local GD: one local epoch = one step.
                _step(x_local, y_local_target)

        # Return updated weights (moved to CPU) and metrics
        model.eval()
        with torch.no_grad():
            out = model(x_local)
            local_loss = loss_fn(out, y_local_target).item()
            local_acc = compute_accuracy(out, y_local)

        updated_weights = _model_to_ndarrays(model)
        drift = compute_drift(parameters, updated_weights)
        weight_norm = float(sum(np.sum(w**2) for w in updated_weights) ** 0.5)
        # IPR is GrokNet-specific; NaN on other architectures.
        local_ipr = compute_ipr(model)["ipr"] if fourier_applicable(model, cfg) else float("nan")

        # Extract W1 for per-client Fourier analysis (only at checkpoint rounds)
        # Serialize as bytes since Flower metrics only support Scalar types
        server_round = int(config.get("server_round", 0))
        ckpt_every = int(config.get("checkpoint_every", 0))
        is_checkpoint_round = (ckpt_every > 0 and server_round > 0
                               and server_round % ckpt_every == 0)
        if (config.get("checkpoint_client_weights", False) and is_checkpoint_round
                and fourier_applicable(model, cfg)):
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
        # Server-side adaptive optimiser (Adam) on the pseudo-gradient.
        return FedAdam(**common_kwargs, eta=cfg.server_lr, tau=cfg.tau)
    if cfg.strategy == "fedyogi":
        # Server-side Yogi — Adam's sign-based sibling; often stronger than Adam.
        return FedYogi(**common_kwargs, eta=cfg.server_lr, tau=cfg.tau)
    if cfg.strategy == "fedavgm":
        # Server-side heavy-ball momentum on the pseudo-gradient. This IS
        # DiLoCo's outer optimiser, so it bridges to the local-SGD-at-scale line.
        return FedAvgM(
            **common_kwargs,
            server_learning_rate=cfg.server_lr,
            server_momentum=cfg.server_momentum,
        )
    # "fedavg" and "fedprox" both use the plain FedAvg strategy; FedProx's
    # proximal term is applied client-side in GrokClient.fit(). SCAFFOLD and
    # FedDyn are custom (per-client state) and handled in their own commit.
    return FedAvg(**common_kwargs)


# ── Flower Server + Evaluation ───────────────────────────────────────────────

def fed_train(cfg: FedConfig):
    """Run FedAvg via Flower simulation. Returns history dict and final model."""
    torch.manual_seed(cfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    # Drop warm client state from any previous run in this process. The cache is
    # keyed so stale entries can't be mis-served, but clearing frees their GPU
    # memory when many runs share one process (e.g. the test suite).
    _client_cache.clear()

    # Precompute global data (single call, also populates cache for clients)
    client_data, x_train_full, y_train_full, x_test, y_test = _get_cached_datasets(cfg)

    # Move evaluation data to device, then prepare loss targets there.
    x_test = x_test.to(device)
    y_test = y_test.to(device)
    x_train_full = x_train_full.to(device)
    y_train_full = y_train_full.to(device)

    loss_spec = build_loss(cfg)
    n_classes = dataset_dims(cfg)[1]
    y_test_target = loss_spec.prepare_target(y_test, n_classes)
    y_train_full_target = loss_spec.prepare_target(y_train_full, n_classes)

    # Print partition sizes
    print(f"Clients: {cfg.num_clients}, partition: {cfg.partition}, "
          f"samples per client: {[len(y) for _, y in client_data]}")

    # History (captured by closure in evaluate_fn)
    history = {
        "round": [], "total_steps": [], "sequential_steps": [],
        "n_participating": [],
        "train_loss": [], "test_loss": [],
        "train_acc": [], "test_acc": [],
        "weight_norm_layer1": [], "weight_norm_layer2": [],
        "ipr": [],
        "mean_client_drift": [],
        "client_weight_divergence": [],
    }

    # Mutable container to capture final model parameters from evaluate_fn
    _final_ndarrays = [None]

    loss_fn = loss_spec.loss_fn
    start_time = time.time()

    # Mutable container for inter-callback communication (closure-shared).
    # Flower calls aggregate_fit (hence _aggregate_fit_metrics) before
    # evaluate_fn within a round, so evaluate_fn reads this round's values.
    _round_metrics = {"mean_drift": 0.0, "weight_divergence": 0.0,
                      "n_participating": 0, "samples_this_round": 0}
    _client_w1_cache = {"data": None}

    # Running count of centralized-equivalent gradient steps (see evaluate_fn).
    _step_accum = {"total": 0.0}
    n_train_total = len(y_train_full)

    def _aggregate_fit_metrics(metrics_list):
        """Aggregate per-client fit metrics from GrokClient.fit()."""
        drifts = [m.get("drift", 0.0) for _, m in metrics_list]
        w_norms = [m.get("weight_norm", 0.0) for _, m in metrics_list]

        _round_metrics["mean_drift"] = float(np.mean(drifts)) if drifts else 0.0
        _round_metrics["weight_divergence"] = float(np.std(w_norms)) if len(w_norms) > 1 else 0.0

        # Actual work done this round, taken from what clients reported rather
        # than from fraction_train. This is exact under partial participation
        # and under Dirichlet partitions, where shards have unequal sizes.
        _round_metrics["n_participating"] = len(metrics_list)
        _round_metrics["samples_this_round"] = sum(int(n) for n, _ in metrics_list)

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

    # One eval model, reused across rounds (was rebuilt + moved to device every
    # round). evaluate_fn runs in the server process, so this is a plain closure
    # variable, not a Ray-actor cache. It is assigned below, AFTER init_model, so
    # its construction does not consume the torch RNG draw that seeds the initial
    # global weights — otherwise the whole run starts from a different point.
    _eval_model_box = [None]

    def evaluate_fn(server_round, parameters, config):
        """Centralized evaluation after each aggregation round."""
        _final_ndarrays[0] = parameters  # capture for final model reconstruction

        # total_steps must accumulate on EVERY round, independent of eval_every,
        # or skipped rounds would silently drop their gradient work. This is the
        # compute-matched x-axis: each round the participating clients compute E
        # local steps over `samples_this_round` examples, i.e. an equivalent of
        # E * samples_this_round / n_train_total full-training-set steps. At full
        # participation this reduces to E per round (the old `round * E`); it
        # over-counted only under partial participation.
        _step_accum["total"] += (
            cfg.local_epochs * _round_metrics["samples_this_round"] / n_train_total
        )

        is_checkpoint_round = (
            cfg.checkpoint_every > 0 and server_round > 0
            and server_round % cfg.checkpoint_every == 0
        )
        # Round 0 (initial state), the final round, checkpoint rounds, and every
        # eval_every-th round are logged; the rest skip the forward passes.
        is_eval_round = (
            server_round % max(1, cfg.eval_every) == 0
            or server_round == cfg.num_rounds
            or is_checkpoint_round
        )
        if not is_eval_round:
            return 0.0, {}

        model = _eval_model_box[0]
        _load_ndarrays_into(model, parameters)

        model.eval()
        with torch.no_grad():
            out_test = model(x_test)
            test_loss = loss_fn(out_test, y_test_target).item()
            test_acc = compute_accuracy(out_test, y_test)
            out_train = model(x_train_full)
            train_loss = loss_fn(out_train, y_train_full_target).item()
            train_acc = compute_accuracy(out_train, y_train_full)

        # weight_norms / IPR are GrokNet-specific; NaN on other architectures.
        if fourier_applicable(model, cfg):
            wn = weight_norms(model)
            wn1, wn2 = wn["weight_norm_layer1"], wn["weight_norm_layer2"]
            ipr_val = compute_ipr(model)["ipr"]
        else:
            wn1 = wn2 = ipr_val = float("nan")

        history["round"].append(server_round)
        # sequential_steps = rounds * E is the depth of the update chain,
        # independent of participation; total_steps (accumulated above) is the
        # compute-matched axis. The two diverge once fraction_train < 1.
        history["total_steps"].append(_step_accum["total"])
        history["sequential_steps"].append(server_round * cfg.local_epochs)
        history["n_participating"].append(_round_metrics["n_participating"])
        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["weight_norm_layer1"].append(wn1)
        history["weight_norm_layer2"].append(wn2)
        history["ipr"].append(ipr_val)
        history["mean_client_drift"].append(_round_metrics["mean_drift"])
        history["client_weight_divergence"].append(_round_metrics["weight_divergence"])

        # Save checkpoint if requested
        if is_checkpoint_round:
            ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)

            # Global model checkpoint
            ckpt_path = os.path.join(ckpt_dir, f"ckpt_round{server_round}.pt")
            torch.save(model.state_dict(), ckpt_path)

            # Global Fourier spectrum (GrokNet-specific; skipped otherwise)
            if fourier_applicable(model, cfg):
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
                f"ipr={ipr_val:.4f}  ({elapsed:.1f}s)"
            )

        return test_loss, {"test_accuracy": test_acc, "train_accuracy": train_acc}

    # Initial model for FedAvg. Its RNG draw MUST come before the eval model's,
    # since these initial weights are the run's starting point.
    init_model = _make_model(cfg)
    init_params = ndarrays_to_parameters(_model_to_ndarrays(init_model))

    # Now safe to build the reusable eval model (params get overwritten each
    # round, so its own initial values are irrelevant — only the RNG order was).
    _eval_model_box[0] = _make_model(cfg).to(device)

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

    # Allocate fractional CUDA GPUs across clients so all K can be co-scheduled
    # on one device; MPS is used via PyTorch directly (not managed by Ray), so
    # num_gpus stays 0 for MPS. The launcher pins one device per run with
    # CUDA_VISIBLE_DEVICES, so "one GPU" here means that run's assigned device.
    num_gpus = 0.0
    if torch.cuda.is_available():
        num_gpus = 1.0 / cfg.num_clients

    # Ray init tuning: the dashboard and the metrics exporter agent are pure
    # overhead for a short-lived single-run simulation. The exporter in
    # particular floods the logs with "Failed to establish connection to the
    # metrics exporter agent ... repeated Nx across cluster" retries. Disabling
    # both removes that traffic. log_to_driver=False keeps client stdout out of
    # the server log.
    backend_config = {
        "client_resources": {"num_cpus": 1, "num_gpus": num_gpus},
        "init_args": {
            "include_dashboard": False,
            "log_to_driver": False,
            "_metrics_export_port": None,
        },
    }

    run_simulation(
        server_app=server_app,
        client_app=client_app,
        num_supernodes=cfg.num_clients,
        backend_config=backend_config,
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
    adam_suffix = f"_adam_tau{cfg.tau}" if cfg.strategy == "fedadam" else ""
    slr_suffix = f"_slr{cfg.server_lr}" if cfg.strategy == "fedadam" else ""
    wd_suffix = f"_wd{cfg.weight_decay}" if cfg.weight_decay > 0 else ""
    tag = (f"fed_{cfg.task}_{cfg.optimizer}_p{cfg.p}_N{cfg.hidden_width}"
           f"_a{cfg.alpha}_K{cfg.num_clients}_le{cfg.local_epochs}"
           f"_ft{cfg.fraction_train}_{cfg.partition}{dirichlet_suffix}"
           f"{prox_suffix}{adam_suffix}{slr_suffix}{wd_suffix}_s{cfg.seed}")
    history_path = os.path.join(cfg.output_dir, f"history_{tag}.json")
    with open(history_path, "w") as f:
        json.dump(history, f)
    print(f"\nHistory saved to {history_path}")

    if cfg.save_weights:
        weights_path = os.path.join(cfg.output_dir, f"weights_{tag}.pt")
        torch.save(final_model.state_dict(), weights_path)
        print(f"Weights saved to {weights_path}")

    return history, final_model
