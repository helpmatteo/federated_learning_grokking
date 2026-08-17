# Plan: replicate the campaign on the four new setups

## Context

Every federated result this project has is from **one setup** — Gromov's quadratic
"GrokNet" MLP on modular addition, MSE, full-batch GD. Four other setups were built
during the v2 rewrite and verified to grok *centrally*, but none has ever been run
federated:

| tag | setup | config |
|---|---|---|
| **T** | Nanda transformer + modular | p=113, CE + AdamW lr 1e-3 wd 1.0, `hidden_width=128` |
| **S5-G** | GrokNet + S₅ composition | CE + AdamW, `hidden_width=256`, 120 classes |
| **S5-T** | Transformer + S₅ | CE + AdamW, `hidden_width=128` |
| **M** | Omnigrok MNIST | 3-layer MLP w200, `init_scale=9`, MSE + AdamW, `batch_size=200`, `n_train=1000` |

So the project's headline claims — the ~16% delay law, "structure beats randomness",
the K=97 partial transition — rest on a single architecture and a single task. This
plan makes those claims testable across setups.

Two things make this more than "run the existing manifest":

1. **Four blocking bugs corrupt new-setup runs**, two of them silently. They are
   verified below, not suspected.
2. **`t1_replication` is not a replication.** 96 of its 150 runs are the *old* setup;
   only 54 touch new setups; zero touch MNIST; K is fixed at 10; α is fixed; FedAvg
   only; 3 seeds; no checkpoints. It is an existence check. It has never been
   executed, so it can be rewritten at zero compute cost.

**Out of scope by instruction:** re-running or extending anchor-setup campaigns. The
banked 158 runs are the comparison baseline, untouched.

---

## Hard constraint: do not orphan the banked runs

`manifest.py:61-76` hashes the spec dict **verbatim**. Adding any key changes the run
id even at its default value — verified: `+dataset:"modular"` moves
`fede_addition_c3a3912baf88` → `fede_addition_cd8dfe74e2c3`. So:

- Never add a field to an existing manifest block. New fields go in **new** blocks only.
- Result-row changes (`run.py`) and collector changes are **safe** — ids hash the spec,
  not the row.
- The modular/GrokNet numerical path must stay bit-identical. Every fix below is
  checked against that.

---

## Phase 0 — Bug fixes (no compute)

### 0.1 Dirichlet partition drops classes — CRITICAL, verified

`data/partition.py:70,162,174` passes `p=cfg.p` and loops `for c in range(p)`. S₅ has
120 classes; `cfg.p` defaults to 97. Measured on the real config:

```
s5, K=10, alpha=0.5, seed=42, dirichlet(0.5)
train total 7200 → assigned 5846, DROPPED 1354 (18.8%)
classes present in shards: 97 of 120
max achievable train acc ≈ 81.2%   ← S₅ grok bar is 85.0
```

Every S₅-dirichlet cell would censor for a partitioning bug and read as a federated
null. Fix: class count from `dataset_dims(cfg)[1]`. Drop the dead `p` arg from
`_partition_by_operand` / `_partition_by_target` so this cannot recur.
*Modular is unaffected* (`cfg.p == n_classes`).

### 0.2 Dataset cache key collides across setups — CRITICAL, verified

`training/federated.py:46-48` omits `dataset, group_n, coset_subgroup, n_train, n_test`
(data-determining) and `model, loss, hidden_width, activation, n_layers, init_scale`
(which `_client_cache` at `:79` piggybacks on — it caches a built model *and* a
loss-specific target tensor). Measured:

```
modular/groknet/mse  key = (97,'addition',0.5,42,5,'iid',0.5)
s5/transformer/ce    key = (97,'addition',0.5,42,5,'iid',0.5)   COLLIDE
mse vs ce                                                        COLLIDE
mnist n_train 1000 vs 4000                                       COLLIDE
CrossEntropyLoss(logits, one_hot_float) = 1.7699  ← no error raised
```

The mse→ce collision **does not raise** — CE accepts one-hot as soft labels — so it
silently mis-trains. Fix: split into `_data_key(cfg)` and `_client_key(cfg, pid)`
(the latter adding model/loss/hidden_width/n_layers/activation/init_scale/n_heads/
d_mlp); use `_client_key` for `_client_cache` **and** the SCAFFOLD `cv_key` at
`:196`, which piggybacks the same broken key. Clear `_dataset_cache` alongside
`_client_cache`; correct the false comment at `federated.py:350-351`.

**Severity correction:** this does *not* implicate any banked result.
`launch_sweep.py:76-94` spawns one subprocess per spec, so `cfg` is constant within
a run and the key is never exercised for collision in the sweep path. It is live only
where one process calls `fed_train` more than once —
`experiments/exp_mechanistic_checkpoints.py:22` and any multi-setup test or analysis
driver. Fix it as a footgun; nothing needs re-running.

### 0.3 Collector destroys setup identity

`scripts/collect_runs.py:19-27` drops 11 fields `run.py` writes: `dataset, model, loss,
batch_size, init_scale, n_layers, group_n, coset_subgroup, checkpoint_every,
server_momentum, grok_threshold`. Consequence for this campaign: S5-GrokNet and
S5-Transformer differ **only** in `model` (dropped) and `hidden_width` (present but
not a cell key) — they merge into one survival curve. And S₅ specs omit `task`/`p`, so
an S₅ row lands as `task=addition, p=97`, indistinguishable from a mod-97 run.

Fix: add all 11 to `COLUMNS`. Also add `epochs`, `n_train`, `n_test`, `activation` to
the `run.py` row — `n_train` is MNIST's data-fraction axis and `epochs` is the
centralized censoring time; neither is currently recoverable.

### 0.4 `summarize_runs` pools silently

`scripts/summarize_runs.py:52` filters out cell keys absent from the CSV with no
warning. That is what would let 0.3's damage pass unnoticed. Make an absent
named key **warn loudly**.

### 0.5 Split the capability gate

`metrics/fourier.py:17-33` conflates "model exposes W1/P" with "dataset is cyclic",
and `centralized.py:99` / `federated.py:478` gate weight norms behind it. So S5-GrokNet
logs `weight_norm_layer1/2 = NaN` although Frobenius norms are basis-free and valid,
and the MNIST MLP has no weight-norm logging at all — when total weight norm *is* the
Omnigrok order parameter.

Fix: `weight_norms_applicable(model)` (hasattr W1/W2) for norms, `fourier_applicable`
(unchanged) for DFT/IPR/spectrum, plus a generic `total_weight_norm(model)` over
`state_dict`. Modular history is unchanged (that gate was already True).

### 0.6 `cfg.p` in the per-client weight slice

`federated.py:253,262` (and the deserialization at `:418-421`) use `cfg.p` as the W1
slice width. Wrong for S5-GrokNet (should be `model.P` = 120). Currently unreachable
behind the gate 0.5 is about to loosen — fix it *before* 0.5 lands, or it goes live.

### 0.7 Figure path — build new, do not patch

The hardcoded 95.0 in `plot_exp2..6` / `plot_drift_vs_grokking` is **not worth
patching**. Every one of those globs `results/experiments/expN_*/history_*.json` by
filename (`plot_exp3.py:22,382`, `plot_exp6_mechanistic.py:34-38`,
`plot_drift_vs_grokking.py:91-127`); none reads `runs_v2.csv` or `results/runs/<id>/`.
They are v1-shaped and structurally cannot consume v2 output, so they will never see a
new-setup run.

Instead build `scripts/plotting/v2_figures.py` driven by `results/data/runs_v2.csv`,
taking the bar from each row's own `grok_threshold` column (already written by
`run.py:107`, surfaced by 0.3). Retire `experiments/exp6_mechanistic.py`.

### 0.8 Post-hoc analyzer rebuilds the wrong dataset

`analysis/mechanistic.py:96-101` hardcodes `GrokNet(...)`, `make_dataset`, and
`Config(p=p, alpha=0.5, seed=42)` with default `task="addition"`. It cannot load a
transformer/MLP checkpoint, and it is **already silently wrong** for banked modular
runs at α=0.3. Take the config from the run's result row.

### 0.9 SCAFFOLD's control variate is invalid under AdamW — NEW

`training/scaffold.py:94` uses `scale = 1/(lr · n_steps)`, i.e. Option II's
`c_i⁺ = c_i − c + (x − y_i)/(ηK)`, which is derived from `x − y_i = η·Σg`. Under AdamW
`x − y_i ≈ η·Σ(m̂/(√v̂+ε)) + decay` — a *per-coordinate* preconditioned sum, so `c_i`
is off by a coordinate-varying factor, not a scalar. **All four new setups use AdamW**,
and SCAFFOLD is the load-bearing "is drift the mechanism?" arm of the algorithms axis.

Minimum honest fix: **raise** in `_build_strategy` when
`strategy == "scaffold" and optimizer == "adamw"`, stating that the estimator assumes
SGD. Do not silently produce numbers. Exclude SCAFFOLD from the AdamW algorithm cells.

### 0.10 Optimizer-state reset is an AdamW-only confound — NEW

`federated.py:176-180` deliberately rebuilds the optimizer each round. For GD at
`momentum=0` that is a genuine no-op — which is why the anchor's E-axis is clean. For
AdamW it means every round is E *bias-corrected cold-start* Adam steps, a
qualitatively different local trajectory. **The anchor's E-result and any new-setup
E-result are therefore not measuring the same quantity**, which is the single biggest
threat to the E-axis replication.

Fix: add `persist_local_opt_state: bool = False` to `FedConfig` (default reproduces
today's behaviour exactly, so zero effect on any banked run); when True, cache the
optimizer alongside the warm model in `_client_cache`. Quantify with a 12-run A/B in
Phase 2. This is also what would make an E=1 FedAvg↔centralized identity test possible
for AdamW at all — `tests/test_fedavg_identity.py` currently only holds for GD.

### 0.11 Orphan guard for the banked runs

Two cheap protections, landed **before** any schema change:
- a test asserting every line of every `manifests/*.jsonl` satisfies
  `run_id(spec − id) == spec["id"]` (passes today — verified 230/230 across
  `t1_replication`, `t2_k_breakdown`, `t2_boundary`);
- `write_manifest` refuses to overwrite a file if any existing line's id would change,
  plus a `FROZEN` set in `build_manifests.main()` for completed manifests.

---

## Phase 1 — Capability build (no compute)

### 1.1 Federated MNIST — `data/partition.py`
Only `operand` and `coset` need the grid; `iid`/`target`/`dirichlet` need just
`y_train`. Replace the blanket `NotImplementedError` at `:39-43`:
- grid path unchanged (preserves the RNG-consumption contract at `:48-50`, so
  published modular runs stay byte-identical);
- non-grid path sources tensors from `build_dataset(cfg)` and rejects only
  operand/coset, with a clear message.

**Must not call `split_indices`/`cfg.alpha`** on that path — MNIST's split is
`n_train`/`n_test` under a `torch.Generator` (`data/mnist.py:39-51`); reusing α would
desynchronize federated MNIST from centralized MNIST. Partitioners need numpy;
`load_mnist_subset` returns tensors. Rewrite `tests/test_groups.py:137-142`, which
currently asserts the block exists.

### 1.1b `label_block` — a structured partition that works on every setup
The anchor's structured arm is `operand` (a coherent row-block of the grid). That
transfers verbatim to S₅ (`ia % K`) but **does not exist for MNIST**, which would lose
the structure axis — the project's most interesting finding — on that setup entirely.
Also note `target` breaks on MNIST for K > 10 (`y % K` leaves clients 10..K−1 empty and
the guard at `partition.py:80` raises), and `coset` cannot be swept in K at all (S₄
gives exactly 5 cosets, A₅ gives 2; `partition.py:140-144` enforces the match).

Add `label_block`: sort train indices by label, `np.array_split` into K contiguous
chunks. Coherent at every K, never empty, identical semantics on modular / S₅ / MNIST,
~6 lines. Adding it to the `partition` Literal changes no existing run id.

### 1.2 A per-round progress scalar for every setup
The modular story's spine is `ipr` in the history. Nothing analogous is logged on any
new setup. Add, behind capability guards, NaN elsewhere:
- **S₅** → `coset_attribution` (`metrics/nonabelian.py`) — already correct and
  model-agnostic, just never called outside tests;
- **transformer** → IPR of the DFT of `W_E` rows over the token dim, the direct
  analogue of `W1[:, :p]` (`transformer.py:37`);
- **all** → `total_weight_norm` from 0.5.

This changes the history dict → run the trajectory equivalence check on modular.

### 1.3 Generalize per-client weight capture
`w1_first_p` (`federated.py:245-263`) is the **only** client→server weight channel and
it is GrokNet-gated and `cfg.p`-shaped. Without it the mechanism arm has no data on any
new setup. Replace with a per-model signature matrix: GrokNet → `W1[:, :P]`,
transformer → `W_E`, MLP → `layers[0].weight`. Emit a warning when skipped — today it
is silent (`:511-520`).

### 1.4 Setup-aware post-hoc analyzer
Dispatch on `cfg.model`/`cfg.dataset`. Reuse `effective_rank` and `gini_coefficient`
(`metrics/fourier.py:115-141`) — both already take an arbitrary tensor.

### 1.5 `n_heads` / `d_mlp` as Config fields
`core/registry.py:80-84` reads them via `getattr` with defaults 4/512, but they are not
dataclass fields — so `manifest.build_config` **raises** if a manifest sets them.
Transformer capacity is neither sweepable nor recorded. Add them; reference them in
**new blocks only** (see the hard constraint). Note `d_model = hidden_width` must be
divisible by `n_heads`.

---

## Phase 2 — Per-setup calibration (cheap, centralized) → **GATE A**

The load-bearing stage. Every FL cell must sit just above its own setup's data cliff,
and each new setup's cliff is unknown. The project's hardest-won lesson
(`build_manifests.py:386-411`) is that a censored cell at the wrong budget is an
artifact, not a result.

There is also **no pipeline-produced centralized anchor** for T / S5-G / S5-T — the
published grok times (transformer 6,200; S₅ ~14,000) came from ad-hoc runs with no
result JSON. Verified: zero transformer or S₅ runs exist on disk. Without an anchor
there is no denominator for a delay ratio.

| manifest | cells | runs |
|---|---|---|
| `t5_anchor_transformer` | α ∈ {0.2,0.25,0.3,0.4,0.5} × 5 seeds, centralized, p=113 | 25 |
| `t5_anchor_s5` | α ladder × {groknet, transformer} × 5 seeds | 50 |
| `t5_anchor_mnist` | `n_train` ∈ {500,1000,2000,4000} × `batch_size` ∈ {50,200} × 5 seeds | 40 |
| `t5_cost_probe` | 200-round FL run per setup at K ∈ {10,50} | 8 |

**The MNIST sizing problem this resolves.** At the Omnigrok grok point
(`n_train=1000`) every shard is ≤200 samples for any K ≥ 5, so `batch_size=200`
collapses a local epoch to a single full-batch step:

```
n_train  K   per client   steps/epoch @bs200
   1000  10         100    1  ← degenerate
   4000  10         400    2
   4000   5         800    4
```

So MNIST FL needs its own (`n_train`, `batch_size`) working point, and that point must
still grok. This is a real experiment, not a formality.

**The cost probe.** The fitted model `(9.8 + 1.291·K + 0.418·E)·rounds/10⁴` is
GrokNet-only. Measured inputs for the new setups:

| setup | params | ms/step | weight payload / client / round |
|---|---|---|---|
| GrokNet + modular | 74,496 | 0.365 | 291 KB |
| Transformer | 225,792 | 2.236 | **882 KB** |
| GrokNet + S₅ | 92,160 | 0.342 | 360 KB |
| MLP + MNIST | 199,210 | 0.495 | 778 KB |

Compute is 6× for the transformer but wall-clock is ~99% orchestration, and
orchestration *is* weight-shipping — so the **3× payload** is the term that matters.
At K=50 the transformer moves 86 MB/round versus GrokNet's 28. Fit per setup rather
than extrapolating.

**Gate A:** each setup has a located cliff, a working point, a justified uniform
budget, and a measured cost model. Any setup that does not grok centrally is dropped
here — the `t0_poly_pilot` precedent, where `x²+xy+y²` was correctly excluded before
FL compute was spent.

---

## Phase 3 — K-sweep × partition (the headline) → **GATE B**

Per surviving setup, at its own working point, uniform budget, **5 seeds**:

- K ∈ {5, 10, 20, 50}
- partition ∈ {iid, dirichlet(**0.1**), structured}
  - structured = `operand` for T; `coset` for S5-G/S5-T (K=5 only — 5 cosets of S₄);
    MNIST has no algebraic structure, so iid + dirichlet only.
- `dirichlet_alpha=0.1`, matching the campaigns — `t1_replication` leaves it at the
  0.5 default, a weaker heterogeneity than anything the project has claimed results on.
- **`checkpoint_every` on.** It feeds the run-id hash, so enabling it later re-runs
  everything. Decide now, once.

≈ 200 runs. This is the direct test of the two headline claims — the delay law and
"structure beats randomness" — beyond the anchor.

**Gate B:** does the structure effect replicate? If it appears on S₅ (a non-abelian
group, no Fourier basis to share), the mechanism hypothesis in `RESULTS.md` §5.4 needs
rewriting — which is itself the most interesting outcome available here.

---

## Phase 4 — E-sweep

E ∈ {5, 10, 25, 50} × 3 partitions × K=10, **3 seeds** (this axis reads a trend, not a
fraction). ≈ 130 runs. Tests whether the delay-vs-E law is setup-independent.

## Phase 5 — Algorithms + mechanism (gated on B)

Server-LR calibration **per setup first** — the calibrated LRs in
`t3_algorithm_comparison:344-347` are setup-A-specific and there is no reason they
transfer to a CE/AdamW transformer. Running the comparison uncalibrated reproduces the
exp5 unfairness defect this project already diagnosed. Then 6 algorithms × the 2 setups
Phase 3 flags as most interesting. ≈ 150 runs, plus the mechanism analysis over the
Phase 3 checkpoints.

---

## Scale, honestly

≈ **600 runs**, dominated by Phases 3–5. On 12 slots with per-setup costs from Gate A,
expect **several days of wall-clock**, not hours. The full cross-product (4 setups × all
four axes × 5 seeds) is ~1,200 runs and I do not recommend it. Trims already applied:

- the 96 anchor-setup runs in `t1_replication` are **dropped** (out of scope);
- 3 seeds on the E-sweep, 5 on the headline;
- the fixed-per-client-K arm is dropped — it is geometrically impossible near a
  boundary (holding shard size needs α > 1 at K=50);
- Phase 5 restricted to 2 setups rather than 4.

Gates A and B exist so this can be stopped after ~90 runs with a real result.

---

## Verification

**Regression (must stay green):** 364 tests. Add — S₅ dirichlet assigns 100% of the
training set across all 120 classes; `_dataset_cache_key` separates the four setups
pairwise; the result row and CSV carry all 15 identity fields; `summarize_runs` warns
on an absent cell key; MNIST FL succeeds for iid/dirichlet and rejects operand/coset.

**Numerical (the hard constraint)** — cheapest first:

1. **Partition index equality** (seconds, no training). Worktree the pre-fix commit;
   over `task × p × α × seed × K × dirichlet_alpha`, assert the client index lists from
   `make_federated_datasets` are element-wise identical before and after 0.1 / 1.1 /
   1.1b. This is ~1000× cheaper than a trajectory diff and is where 0.1 actually lives.
2. **Trajectory diff** (`traj.py`, the harness that already caught two bugs): centralized
   modular p=97 α=0.5 GD 2000 epochs, and federated K=10 E=5 200 rounds × 3 partitions.
   Diff **over the intersection of history keys** — 0.5/1.2 add keys by design, so a
   whole-dict diff false-positives. Bar: worst |Δ| < 1e-5.
3. **Replay a banked run** — the real proof, and what I'd gate the Phase 0 commit on.
   162 history directories are on disk. Re-run three banked ids (one `t0_wd_grid`
   centralized; one `t2_k_breakdown` K=50 dirichlet, which exercises 0.1 on the anchor;
   one `t2_boundary` K=97 with `checkpoint_client_weights=True`, which exercises 1.3)
   and diff against each archived `history_*.json`. For the boundary run also `cmp` the
   regenerated `client_w1_round*.pt` bytes — for modular, `dataset_dims[0]//2 == p`
   exactly, so they must be byte-identical.
4. **Id stability:** recompute `run_id` for every spec in every manifest and assert all
   158 banked `results/data/runs/<id>.json` still exist.
5. Re-run `collect_runs.py`; confirm the §4/§5 tables in `RESULTS.md` regenerate.

**`scripts/validate_manifest.py` (new, highest value per line):** for every spec,
`build_config` and — for federated — construct the partition *without training*.
Catches unknown keys, empty shards, coset/K mismatches, `batch_size ≥ shard size`, and
`hidden_width % n_heads` across ~1000 runs in seconds. Run before every launch.

**Disk budget, per setup, before Phase 3:** the MNIST signature matrix is 200×784 =
627 KB/client → at K=50 × 20 checkpoints that is **627 MB/run**. Either subsample the
signature or lower `checkpoint_every` on MNIST mechanism cells.

**Smoke (before any serious compute):** one short run per setup (200 rounds, K=10) end
to end through `launch_sweep.py` — confirms transformer/MNIST/S₅ actually train under
`fed_train`, which **no test currently covers** (no test anywhere runs `fed_train` with
`dataset != "modular"` or `model != "groknet"`). Confirm checkpoints land and load back
through the 1.4 analyzer. Verify `--dry-run` shows no id collisions and that banked
runs are skipped.

**Ordering:** `launch_sweep.py` fills slots FIFO in manifest order with no
longest-first sort, so builders must emit long jobs first — worth ~18% wall-clock on
a mixed manifest.
