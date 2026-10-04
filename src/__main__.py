"""Reproduce the numerical experiments with ``python -m src``."""
from __future__ import annotations

import argparse
from pathlib import Path
import time


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="use smaller verification and search budgets, and a three-unit fleet")
    parser.add_argument("--output", type=Path, default=Path("artifacts"), help="output root (default: artifacts); contains results, tables and figures")
    parser.add_argument("--workers", type=positive_int, default=2, help="adversarial search processes (default: 2)")
    parser.add_argument("--figures-only", action="store_true", help="render figures from the supplied output root's existing results")
    args = parser.parse_args(argv)

    from . import experiments as E, plotting

    root = args.output.resolve()
    E.RES, E.TAB = str(root / "results"), str(root / "tables")
    plotting.RES, plotting.FIG = E.RES, str(root / "figures")
    if args.figures_only:
        plotting.make_all()
        print(f"Figures written to {plotting.FIG}")
        return 0

    units, cats, from_rts = E.get_units()
    if not from_rts:
        parser.error("the experiment suite requires data/gen.csv (RTS-GMLC)")
    E.CHECKS.clear()
    E.FINDINGS.clear()
    E.MACROS.clear()
    E.CHAINS.update(n=0, monotone=0)
    E.TIMINGS.clear()
    Path(E.RES).mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    quick = args.quick
    paths = 1000 if quick else 5000

    def run(label, function, *pos, **kw):
        print(f"{label}…", flush=True)
        before = time.perf_counter()
        result = function(*pos, **kw)
        E.TIMINGS[label] = time.perf_counter() - before
        print(f"  {E.TIMINGS[label]:.2f} s", flush=True)
        return result

    run("Exactness checks", E.exp_exactness, **(dict(n_bf=20, n_hist=6, n_mip=10) if quick else {}))
    run("Counter minimality", E.exp_minimality, **(dict(n_moore=50, n_value=10) if quick else {}))
    run("Base-case thresholds", E.exp_b1, **(dict(n=50, n_flat=20) if quick else {}))
    run("General-case sweeps", E.exp_general, **(dict(n_iid=50, n_gen=100, n_flat=20) if quick else {}))
    run("Persistence", E.exp_persistence, **(dict(n=20) if quick else {}))
    adv = run("Counterexamples and adversarial search", E.exp_counterexamples, workers=args.workers,
              **(dict(budget=100, seeds=(0,)) if quick else {}))
    single = run("Single-unit comparisons", E.exp_single_units, units, n_paths=paths)
    run("Base-case figure data", E.exp_base_figures, units)
    run("Sensitivity", E.exp_sensitivity, units)
    run("Fleet exactness", E.exp_fleet_checks, **(dict(n_bf=3, n_ed=30) if quick else {}))
    fleet = run("Fleet comparisons", E.exp_fleet, units, n_paths=paths, n_units=3 if quick else 5)
    runtime = run("Fleet runtime scaling", E.exp_runtime, units, max_N=3 if quick else 6)
    rows, by = run("State space", E.exp_statespace, units, cats)
    E.table_units(units, from_rts)
    E.table_statespace(by, rows)
    E.table_verification()
    E.table_findings(adv)
    E.macros_from_results(single, fleet, runtime, n_paths=paths)
    run("Figures", plotting.make_all)
    E.TIMINGS["Total pipeline"] = time.perf_counter() - started
    E.table_runtime(runtime, single, fleet, E.TIMINGS)
    E.write_macros()
    E.save_json("timings.json", E.TIMINGS)
    failures = sum(c["failures"] for c in E.CHECKS) + E.CHAINS["n"] - E.CHAINS["monotone"]
    print(f"{len(E.CHECKS)} verification groups; {failures} failures. Outputs: {root}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
