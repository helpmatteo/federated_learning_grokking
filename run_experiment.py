"""Unified CLI for running all experiments.

Usage:
    python run_experiment.py exp0
    python run_experiment.py exp1 --hidden_width 256
    python run_experiment.py exp2 --t_max 50000
    python run_experiment.py exp3a --alphas 0.1,0.15,0.2,0.3,0.5
    python run_experiment.py exp5 --hard_settings hard_settings.json
    python run_experiment.py plot --exp exp1 --results results/exp1_boundary/exp1_results.json
"""

import argparse
import json
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Run grokking experiments")
    subparsers = parser.add_subparsers(dest="experiment", help="Experiment to run")

    p0 = subparsers.add_parser("exp0", help="Width validation")
    p0.add_argument("--output_dir", default="results/exp0_width")

    p1 = subparsers.add_parser("exp1", help="Centralized phase boundary")
    p1.add_argument("--hidden_width", type=int, default=256)
    p1.add_argument("--output_dir", default="results/exp1_boundary")

    p2 = subparsers.add_parser("exp2", help="Aggregation effect")
    p2.add_argument("--alpha", type=float, default=None,
                    help="Single alpha value (for parallel single-cell runs)")
    p2.add_argument("--K", type=int, default=None,
                    help="Single K value (for parallel single-cell runs)")
    p2.add_argument("--alphas", type=str, default=None,
                    help="Comma-separated alphas (for multi-alpha runs)")
    p2.add_argument("--k_values", type=str, default=None,
                    help="Comma-separated K values (default: 2,5,10,20,50,97)")
    p2.add_argument("--hidden_width", type=int, default=256)
    p2.add_argument("--t_max", type=int, default=50000)
    p2.add_argument("--output_dir", default="results/exp2_aggregation")

    for sub in ["exp3a", "exp3a_kval", "exp3b"]:
        p = subparsers.add_parser(sub, help=f"Heterogeneity: {sub}")
        p.add_argument("--alpha", type=float, default=None,
                        help="Single alpha value (for parallel single-cell runs)")
        p.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alphas")
        p.add_argument("--dir_alpha", type=float, default=None,
                        help="Single Dirichlet alpha (for 3a single-cell runs)")
        p.add_argument("--partition", type=str, default=None,
                        help="Single partition type (for 3b single-cell runs)")
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--hidden_width", type=int, default=256)
        p.add_argument("--t_max", type=int, default=50000)
        p.add_argument("--output_dir", default="results/exp3_heterogeneity")

    for sub in ["exp4a", "exp4b", "exp4c"]:
        p = subparsers.add_parser(sub, help=f"Optimization: {sub}")
        p.add_argument("--alpha", type=float, default=None,
                        help="Single alpha value (for parallel single-cell runs)")
        p.add_argument("--alphas", type=str, default=None,
                        help="Comma-separated alphas")
        p.add_argument("--E", type=int, default=None,
                        help="Single E value (local epochs)")
        p.add_argument("--het", type=str, default=None, choices=["iid", "noniid"],
                        help="Heterogeneity setting (for 4a/4b single-cell)")
        p.add_argument("--frac", type=float, default=None,
                        help="Participation fraction (for 4b single-cell)")
        p.add_argument("--k", type=int, default=10)
        p.add_argument("--hidden_width", type=int, default=256)
        p.add_argument("--t_max", type=int, default=50000)
        p.add_argument("--output_dir", default="results/exp4_optimization")

    p5 = subparsers.add_parser("exp5", help="Algorithm rescue")
    p5.add_argument("--setting", type=str, default=None,
                    help="Single setting label (H1, H2, H3) for parallel runs")
    p5.add_argument("--algorithm", type=str, default=None,
                    help="Single algorithm label (e.g. FedProx-0.01) for parallel runs")
    p5.add_argument("--hidden_width", type=int, default=256)
    p5.add_argument("--t_max", type=int, default=50000)
    p5.add_argument("--output_dir", default="results/exp5_algorithms")

    p6 = subparsers.add_parser("exp6", help="Mechanistic analysis")
    p6.add_argument("--output_dir", default="results/exp6_mechanistic")

    pp = subparsers.add_parser("plot", help="Generate figures")
    pp.add_argument("--exp", required=True,
                    choices=["exp1", "exp2", "exp3a", "exp4a", "exp5", "exp6"])
    pp.add_argument("--results", required=True, help="Path to results JSON")
    pp.add_argument("--output_dir", default="results/figures")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.experiment == "exp0":
        from experiments.exp0_width import run_exp0
        run_exp0(output_dir=args.output_dir)

    elif args.experiment == "exp1":
        from experiments.exp1_boundary import run_exp1
        run_exp1(hidden_width=args.hidden_width, output_dir=args.output_dir)

    elif args.experiment == "exp2":
        from experiments.exp2_aggregation import run_exp2, run_exp2_cell
        k_values = [int(k) for k in args.k_values.split(",")] if args.k_values else None

        if args.alpha is not None and args.K is not None:
            # Single cell: python run_experiment.py exp2 --alpha 0.25 --K 10
            run_exp2_cell(alpha=args.alpha, K=args.K,
                          hidden_width=args.hidden_width,
                          t_max=args.t_max, output_dir=args.output_dir)
        elif args.alpha is not None:
            # One alpha, all K: python run_experiment.py exp2 --alpha 0.25
            run_exp2(alphas=[args.alpha], k_values=k_values,
                     hidden_width=args.hidden_width,
                     t_max=args.t_max, output_dir=args.output_dir)
        else:
            # Full sweep: python run_experiment.py exp2 --alphas 0.20,0.25,0.30,0.35,0.50
            alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
            run_exp2(alphas=alphas, k_values=k_values,
                     hidden_width=args.hidden_width,
                     t_max=args.t_max, output_dir=args.output_dir)

    elif args.experiment in ("exp3a", "exp3a_kval", "exp3b"):
        if args.experiment == "exp3a":
            if args.alpha is not None and args.dir_alpha is not None:
                from experiments.exp3_heterogeneity import run_exp3a_cell
                run_exp3a_cell(alpha=args.alpha, dir_alpha=args.dir_alpha,
                               k=args.k, hidden_width=args.hidden_width,
                               t_max=args.t_max, output_dir=args.output_dir)
            else:
                from experiments.exp3_heterogeneity import run_exp3a
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp3a(alphas=alphas or [0.20, 0.25, 0.30, 0.35, 0.50],
                          k_primary=args.k, hidden_width=args.hidden_width,
                          t_max=args.t_max, output_dir=args.output_dir)
        elif args.experiment == "exp3a_kval":
            if args.alpha is not None and args.dir_alpha is not None:
                from experiments.exp3_heterogeneity import run_exp3a_cell
                run_exp3a_cell(alpha=args.alpha, dir_alpha=args.dir_alpha,
                               k=args.k, hidden_width=args.hidden_width,
                               t_max=args.t_max, output_dir=args.output_dir)
            else:
                from experiments.exp3_heterogeneity import run_exp3a_k_validation
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp3a_k_validation(alphas=alphas, k_secondary=args.k,
                                       hidden_width=args.hidden_width, t_max=args.t_max,
                                       output_dir=args.output_dir)
        elif args.experiment == "exp3b":
            if args.alpha is not None and args.partition is not None:
                from experiments.exp3_heterogeneity import run_exp3b_cell
                run_exp3b_cell(alpha=args.alpha, partition=args.partition,
                               k=args.k, hidden_width=args.hidden_width,
                               t_max=args.t_max, output_dir=args.output_dir)
            else:
                from experiments.exp3_heterogeneity import run_exp3b
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp3b(alphas=alphas or [0.20, 0.25, 0.30, 0.35, 0.50],
                          k_primary=args.k, hidden_width=args.hidden_width,
                          t_max=args.t_max, output_dir=args.output_dir)

    elif args.experiment in ("exp4a", "exp4b", "exp4c"):
        if args.experiment == "exp4a":
            if args.alpha is not None and args.E is not None and args.het is not None:
                from experiments.exp4_optimization import run_exp4a_cell
                run_exp4a_cell(alpha=args.alpha, E=args.E, het=args.het,
                               k=args.k, hidden_width=args.hidden_width,
                               t_max=args.t_max, output_dir=args.output_dir)
            else:
                from experiments.exp4_optimization import run_exp4a
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp4a(alphas=alphas, k=args.k,
                          hidden_width=args.hidden_width, t_max=args.t_max,
                          output_dir=args.output_dir)
        elif args.experiment == "exp4b":
            if args.alpha is not None and args.frac is not None and args.het is not None:
                from experiments.exp4_optimization import run_exp4b_cell
                run_exp4b_cell(alpha=args.alpha, frac=args.frac, het=args.het,
                               k=args.k, hidden_width=args.hidden_width,
                               t_max=args.t_max, output_dir=args.output_dir)
            else:
                from experiments.exp4_optimization import run_exp4b
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp4b(alphas=alphas, k=args.k,
                          hidden_width=args.hidden_width, t_max=args.t_max,
                          output_dir=args.output_dir)
        elif args.experiment == "exp4c":
            if args.alpha is not None and args.E is not None:
                from experiments.exp4_optimization import run_exp4c_cell
                run_exp4c_cell(alpha=args.alpha, E=args.E,
                               k=args.k, hidden_width=args.hidden_width,
                               output_dir=args.output_dir)
            else:
                from experiments.exp4_optimization import run_exp4c
                alphas = [float(a) for a in args.alphas.split(",")] if args.alphas else None
                run_exp4c(alphas=alphas, k=args.k,
                          hidden_width=args.hidden_width,
                          output_dir=args.output_dir)

    elif args.experiment == "exp5":
        from experiments.exp5_algorithms import run_exp5, run_exp5_cell
        if args.setting is not None and args.algorithm is not None:
            run_exp5_cell(setting_label=args.setting, algo_label=args.algorithm,
                          hidden_width=args.hidden_width, t_max=args.t_max,
                          output_dir=args.output_dir)
        elif args.setting is not None:
            from experiments.exp5_algorithms import ALGORITHMS
            run_exp5(settings=[args.setting],
                     hidden_width=args.hidden_width, t_max=args.t_max,
                     output_dir=args.output_dir)
        else:
            run_exp5(hidden_width=args.hidden_width, t_max=args.t_max,
                     output_dir=args.output_dir)

    elif args.experiment == "exp6":
        from experiments.exp6_mechanistic import analyze_drift_vs_grokking
        analyze_drift_vs_grokking(
            results_dirs=["results/exp2_aggregation", "results/exp3_heterogeneity",
                          "results/exp4_optimization", "results/exp5_algorithms"],
            output_dir=args.output_dir,
        )

    elif args.experiment == "plot":
        from experiments import visualization as viz
        if args.exp == "exp1":
            viz.plot_exp1_phase_boundary(args.results, args.output_dir)
        elif args.exp == "exp2":
            viz.plot_exp2_aggregation(args.results, args.output_dir)
        elif args.exp == "exp3a":
            viz.plot_exp3a_phase_diagram(args.results, args.output_dir)
        elif args.exp == "exp4a":
            viz.plot_exp4a_drift(args.results, args.output_dir)
        elif args.exp == "exp5":
            viz.plot_exp5_algorithms(args.results, args.output_dir)
        elif args.exp == "exp6":
            viz.plot_exp6_drift_scatter(args.results, args.output_dir)

    else:
        print("No experiment specified. Use --help for options.")
        sys.exit(1)


if __name__ == "__main__":
    main()
