"""Check a manifest without training anything.

Every failure this catches is one that would otherwise surface hours into a
sweep, on a GPU, after the launcher had already committed slots to it:

  - a spec key that is neither a config field nor a tag (build_config raises);
  - a partition that leaves some client with no samples -- K too large for the
    data, or "target" with K > n_classes;
  - a partition a dataset cannot support (operand/coset need a grid; coset needs
    K to equal the coset count exactly);
  - batch_size >= the per-client shard, which silently degenerates a local epoch
    to a single full-batch step (this is not an error, so it is reported as a
    warning -- it is exactly the trap MNIST at n_train=1000 falls into);
  - hidden_width not divisible by n_heads on a transformer;
  - duplicate run ids within the manifest.

    python scripts/validate_manifest.py manifests/*.jsonl
    python scripts/validate_manifest.py manifests/s3_axes_B.jsonl --verbose
"""

import argparse
import collections
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from fedgrok.manifest import build_config, load_manifest, run_id
from fedgrok.core.registry import build_model
from fedgrok.data.partition import make_federated_datasets
from fedgrok.run import result_path


def check_spec(spec):
    """Return (errors, warnings) for one spec. No training, no GPU."""
    errors, warnings = [], []
    try:
        cfg = build_config(spec)
    except Exception as exc:
        return [f"build_config: {exc}"], []

    # Model construction is cheap on CPU and catches shape/divisibility problems.
    try:
        build_model(cfg)
    except Exception as exc:
        errors.append(f"build_model: {exc}")

    if spec.get("mode") != "federated":
        return errors, warnings

    try:
        client_data, _, _, _, _ = make_federated_datasets(cfg)
    except Exception as exc:
        errors.append(f"partition: {exc}")
        return errors, warnings

    sizes = [len(y) for _, y in client_data]
    bs = getattr(cfg, "batch_size", 0)
    if bs and bs >= min(sizes):
        warnings.append(
            f"batch_size={bs} >= smallest shard ({min(sizes)}): a local epoch "
            f"degenerates to one full-batch step for at least one client, so "
            f"local_epochs no longer means what it does elsewhere"
        )
    if min(sizes) < 10:
        warnings.append(f"smallest shard is {min(sizes)} samples (shards: {sizes})")
    return errors, warnings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifests", nargs="+")
    parser.add_argument("--verbose", action="store_true",
                        help="also report specs that are fine")
    parser.add_argument("--results-root", default="results/data/runs")
    args = parser.parse_args()

    total_bad = 0
    for path in args.manifests:
        specs = load_manifest(path)

        # An id appearing twice is normal and intended: a cell can belong to two
        # grouping blocks (an E-spine point that is also a K-sweep point), and the
        # content hash makes the launcher run it once. That is only a problem if
        # the two specs are actually DIFFERENT configs sharing an id, which would
        # mean a hash collision -- so distinguish the two cases rather than
        # flagging every repeat.
        TAGS = {"id", "tier", "group", "experiment", "setting", "algorithm",
                "label", "manifest", "setup"}
        by_id = collections.defaultdict(list)
        for spec in specs:
            core = tuple(sorted((k, v) for k, v in spec.items() if k not in TAGS))
            by_id[spec["id"]].append(core)
        repeated = {i: v for i, v in by_id.items() if len(v) > 1}
        collisions = [i for i, v in repeated.items() if len(set(v)) > 1]
        n_repeat_specs = sum(len(v) - 1 for v in repeated.values())
        for i in collisions:
            print(f"  ERROR  {i}: same id for {len(set(repeated[i]))} DIFFERENT "
                  f"configs -- run_id hash collision")

        n_err = n_warn = n_done = 0
        # Distinct configurations only: seeds share every structural property, so
        # validating one per cell is enough and keeps this fast on 400-run files.
        seen_shape = set()
        for spec in specs:
            if os.path.exists(result_path(spec, args.results_root)):
                n_done += 1
            shape = tuple(sorted((k, v) for k, v in spec.items()
                                 if k not in ("id", "seed", "output_dir")))
            if shape in seen_shape:
                continue
            seen_shape.add(shape)

            errors, warnings = check_spec(spec)
            for e in errors:
                print(f"  ERROR  {spec['id']}: {e}")
                n_err += 1
            for w in warnings:
                print(f"  WARN   {spec['id']}: {w}")
                n_warn += 1

        status = "OK" if not (n_err or collisions) else "FAILED"
        print(f"{os.path.basename(path):32s} {len(specs):4d} specs, "
              f"{len(seen_shape):3d} distinct cells, {n_done} already done "
              f"-> {status}"
              + (f"  [{n_err} errors]" if n_err else "")
              + (f"  [{n_warn} warnings]" if n_warn else "")
              + (f"  [{n_repeat_specs} shared with another block, run once]"
                 if n_repeat_specs else "")
              + (f"  [{len(collisions)} HASH COLLISIONS]" if collisions else ""))
        total_bad += n_err + len(collisions)

    if total_bad:
        print(f"\n{total_bad} blocking problem(s). Fix before launching.")
        sys.exit(1)
    print("\nAll manifests valid.")


if __name__ == "__main__":
    main()
