"""Aggregate PPO-vs-passive improvements over several independently trained seeds.

  python scripts/seed_summary.py --runs ppo_default ppo_seed1 ppo_seed2

Reads results/<run>/comparison.csv for each run (each produced by evaluate.py on that run's
best_model.zip, same scenarios and evaluation seeds) and writes
  results/seed_summary.csv        mean/std/min/max over training seeds per scenario and metric
  report/seed_table.tex           rows for the report's seed-robustness table
Positive improvement = better than passive (see training/metrics.py).
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import pandas as pd  # noqa: E402

from training.common import ROOT  # noqa: E402

METRICS = ["rms_acc", "peak_acc", "rms_defl", "rms_dtl", "return"]
LABELS = [("bump_10", "bump 10 m/s (T)"), ("rough_B_15", "ISO B 15 m/s (T)"), ("rough_C_15", "ISO C 15 m/s (T)"),
          ("smooth_A_15", "ISO A 15 m/s (U)"), ("flat_perturbed", "flat, +20 mm (U)"),
          ("heldout_bump_tall", "tall bump (H)"), ("heldout_rough_D_10", "ISO D 10 m/s (H)"),
          ("heldout_test_mixture", "test mixture (H)")]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", nargs="+", default=["ppo_default", "ppo_seed1", "ppo_seed2"])
    a = ap.parse_args()
    frames = []
    for run in a.runs:
        c = pd.read_csv(ROOT / "results" / run / "comparison.csv")
        c = c[(c.controller == "ppo") & c.metric.isin(METRICS)][["scenario", "metric", "improvement_pct_of_means"]]
        c["run"] = run
        frames.append(c)
    d = pd.concat(frames, ignore_index=True)
    g = d.groupby(["scenario", "metric"], sort=False).improvement_pct_of_means
    s = pd.DataFrame({"mean": g.mean(), "std": g.std(ddof=1), "min": g.min(), "max": g.max(), "n_training_seeds": g.size()}).reset_index()
    s.to_csv(ROOT / "results" / "seed_summary.csv", index=False)

    def cell(sc, m):
        r = s[(s.scenario == sc) & (s.metric == m)].iloc[0]
        return f"${r['mean']:+.1f}\\pm{r['std']:.1f}$"

    rows = [f"{lab} & {cell(sc, 'rms_acc')} & {cell(sc, 'rms_defl')} & {cell(sc, 'rms_dtl')}" for sc, lab in LABELS]
    (ROOT / "report" / "seed_table.tex").write_text(" \\\\\n".join(rows))
    print(s[s.metric.isin(["rms_acc", "rms_defl", "rms_dtl"])].round(2).to_string(index=False))


if __name__ == "__main__":
    main()
