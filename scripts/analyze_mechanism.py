"""Regenerate RESULTS.md section 16 from banked checkpoints. No training.

    venv/bin/python scripts/analyze_mechanism.py confound   # 16.1
    venv/bin/python scripts/analyze_mechanism.py global     # 16.2
    venv/bin/python scripts/analyze_mechanism.py internals  # 16.3
    venv/bin/python scripts/analyze_mechanism.py all

WHY THE GLOBAL MODEL AND NOT THE CLIENTS. The frequency-consensus hypothesis
behind the partition result (RESULTS 5.4) looks like a per-client question, and
per-client weights were checkpointed specifically to answer it. They cannot, on
the arm that carries the claim: `client_signature` ships W1[:, :p] -- the
first-operand block -- and the `operand` partition shards by first operand, so
at K = 97 = p each client trains exactly ONE column of the matrix being read.
`confound` measures that. The global model is the same object in both arms, so
`global` reads it instead, and gets the pre-transition control for free.
"""
import argparse, csv, glob, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from fedgrok.metrics.fourier import spectral_ipr

CSV = "results/data/runs_v2.csv"
ROUNDS = tuple(range(1000, 20001, 1000))   # every checkpoint
TABLE  = (1000, 4000, 7000, 10000, 13000, 16000, 19000)  # the columns RESULTS 16.2 prints
RANGES = (2000, 4000, 6000, 8000, 10000, 20000)          # the pre-transition control


def boundary_k97():
    return [r for r in csv.DictReader(open(CSV))
            if r["group"] == "boundary" and r["num_clients"] == "97"]


def _ckpt(run_id, kind, rnd):
    p = f"results/runs/{run_id}/checkpoints/{kind}{rnd}.pt"
    return p if os.path.exists(p) else None


def confound():
    """16.1 -- is each client's deviation concentrated in one column?"""
    print("16.1  top-1 column share of per-client deviation energy from the "
          "across-client mean\n      (1.0 = the client moved a single column; "
          "the operand partition IS that column)\n")
    show = (1000, 5000, 9000, 13000, 17000, 20000)
    print(f"{'partition':<11}{'seed':<7}" + "".join(f"{r:>9}" for r in show))
    for part in ("iid", "operand"):
        for r in [x for x in boundary_k97() if x["partition"] == part][:2]:
            out = []
            for rnd in show:
                f = _ckpt(r["id"], "client_w1_round", rnd)
                if f is None:
                    out.append(float("nan")); continue
                W = np.stack(torch.load(f, map_location="cpu", weights_only=False))
                dev = W - W.mean(0, keepdims=True)
                e = (dev ** 2).sum(axis=1)
                out.append(float(np.median(e.max(axis=1) / e.sum(axis=1))))
            print(f"{part:<11}{r['seed']:<7}" + "".join(f"{v:>9.3f}" for v in out))


def _global_ipr(block):
    lo, hi = (0, 97) if block == "a" else (97, 194)
    rec = []
    for r in boundary_k97():
        for rnd in ROUNDS:
            f = _ckpt(r["id"], "ckpt_round", rnd)
            if f is None:
                continue
            W = torch.load(f, map_location="cpu", weights_only=True)["W1"]
            rec.append({"partition": r["partition"], "seed": int(r["seed"]),
                        "round": rnd, "grokked": r["grokked"].lower() == "true",
                        "ipr": spectral_ipr(W[:, lo:hi]),
                        "tfc": float(r["t_first_cross"])
                               if r["t_first_cross"] not in ("inf", "") else float("inf")})
    return rec


def global_(block="a"):
    """16.2 -- global-model spectral IPR, and the pre-transition control."""
    D = _global_ipr(block)
    sel = lambda **kw: [d for d in D if all(d[k] == v for k, v in kw.items())]
    name = "first-operand (sharded)" if block == "a" else "second-operand (NOT sharded)"
    print(f"\n16.2  per-neuron spectral IPR of the GLOBAL W1, {name} block")

    earliest = min(d["tfc"] for d in D if np.isfinite(d["tfc"]))
    print(f"      earliest first crossing anywhere: {earliest:.0f} steps "
          f"= round {earliest/5:.0f}; rounds below it are pre-transition for ALL runs\n")
    show = [r for r in TABLE if r in {d["round"] for d in D}]
    print(f"{'round':<12}" + "".join(f"{r:>9}" for r in show))
    for part in ("iid", "operand"):
        v = [np.median([d["ipr"] for d in sel(partition=part, round=r)]) for r in show]
        print(f"{part:<12}" + "".join(f"{x:>9.4f}" for x in v))

    print("\n      per-seed ranges (separated = no overlap between arms):")
    for r in [x for x in RANGES if x in {d["round"] for d in D}]:
        i = [d["ipr"] for d in sel(partition="iid", round=r)]
        o = [d["ipr"] for d in sel(partition="operand", round=r)]
        tag = "SEPARATED" if min(o) > max(i) else "overlap"
        pre = "pre-transition" if r < earliest / 5 else ""
        print(f"        round {r:<6} iid [{min(i):.4f}, {max(i):.4f}]  "
              f"operand [{min(o):.4f}, {max(o):.4f}]  "
              f"{np.median(o)/np.median(i):.2f}x  {tag:<10}{pre}")

    print("\n      within the IID arm -- does it predict WHICH seeds grok?")
    for r in [x for x in RANGES if x in {d["round"] for d in D}]:
        g = [d["ipr"] for d in sel(partition="iid", round=r, grokked=True)]
        c = [d["ipr"] for d in sel(partition="iid", round=r, grokked=False)]
        if not g or not c:
            continue
        print(f"        round {r:<6} grokked {sorted(round(x,4) for x in g)}  "
              f"censored {sorted(round(x,4) for x in c)}  "
              f"{'SEPARATED' if min(g) > max(c) else 'overlap'}")


IRR = ["irrep_u_5", "irrep_u_41", "irrep_u_32", "irrep_u_311",
       "irrep_u_221", "irrep_u_2111", "irrep_u_11111"]
DIMS = np.array([1, 4, 5, 6, 5, 4, 1])


def internals():
    """16.3 -- setup D's exact quadratic-circuit split across the alpha ladder."""
    rows = [r for r in csv.DictReader(open(CSV)) if r["group"] == "d_internals"]
    null = DIMS ** 2 / (DIMS ** 2).sum()
    acc, irr = {}, {}
    for r in rows:
        hs = glob.glob(f"results/runs/{r['id']}/history_*.json")
        if not hs:
            continue
        h = json.load(open(hs[0]))
        ep = np.array(h["epoch"], dtype=float)
        a = float(r["alpha"])
        for lab, st in (("plateau", 1800), ("dip", 3600), ("end", None)):
            i = len(ep) - 1 if st is None else int(np.argmin(np.abs(ep - st)))
            g = lambda k: (h[k][i] if h.get(k) and h[k][i] is not None else np.nan)
            acc.setdefault((a, lab), []).append(
                (g("test_acc"), g("circ_acc_marginal"), g("circ_acc_interaction"),
                 g("circ_share_interaction"), g("circ_units")))
            v = np.array([g(k) for k in IRR], dtype=float)
            s = np.nansum(v)
            if s > 0:
                irr.setdefault((a, lab), []).append(0.5 * np.abs(v / s - null).sum())

    alphas = sorted({k[0] for k in acc})
    for lab in ("plateau", "dip"):
        step = 1800 if lab == "plateau" else 3600
        print(f"\n16.3  setup D, exact circuit split at the {lab.upper()} (step {step})")
        print("      logit = A[c,a] + 2T[c,a,b] + B[c,b];  marginal = A+B, interaction = T")
        print(f"{'alpha':>7}{'test':>9}{'marginal':>10}{'interaction':>13}"
              f"{'T share':>10}{'T units':>9}{'T - test':>10}")
        for a in alphas:
            m = np.nanmedian(np.array(acc[(a, lab)]), axis=0)
            print(f"{a:>7}{m[0]:>9.1f}{m[1]:>10.1f}{m[2]:>13.1f}"
                  f"{m[3]:>10.3f}{m[4]:>9.1f}{m[2]-m[0]:>10.1f}")

    ups = sum(1 for k in acc if k[1] == "plateau")
    rise = 0; fall = 0
    for a in alphas:
        for p, d in zip(acc[(a, "plateau")], acc[(a, "dip")]):
            if np.isfinite(p[4]) and np.isfinite(d[4]):
                rise += d[4] > p[4]; fall += d[4] <= p[4]
    print(f"\n      circ_units 1800->3600: rises in {rise}, falls in {fall} "
          f"(the pilot's single-seed lead does not replicate)")

    print("\n      S_5 isotypic shares, total-variation distance from the "
          "dim^2-proportional null")
    print(f"{'alpha':>7}{'plateau':>10}{'dip':>9}{'end':>9}")
    for a in alphas:
        v = [np.median(irr.get((a, l), [np.nan])) for l in ("plateau", "dip", "end")]
        print(f"{a:>7}" + "".join(f"{x:>9.3f}" for x in v))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["confound", "global", "internals", "all"])
    a = ap.parse_args()
    if a.what in ("confound", "all"):
        confound()
    if a.what in ("global", "all"):
        global_("a")
        global_("b")
    if a.what in ("internals", "all"):
        internals()
