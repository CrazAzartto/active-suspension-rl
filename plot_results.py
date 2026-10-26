"""Regenerate all figures from saved CSV/log data (nothing is typed in by hand).

  python plot_results.py --run ppo_default
  python plot_results.py --run ppo_default --scenarios rough_C_15 bump_10 --tmax 6

Reads  logs/<run>/{monitor/*.monitor.csv, validation_log.csv}   (training curves)
       results/<run>/{timeseries.csv, comparison.csv, summary.csv} (evaluation)
Writes figures/<run>/*.png (+ .pdf) and figures/<run>/captions.md
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
STYLE = {"passive": dict(color="#555555", lw=1.2, label="Passive (u = 0)"),
         "ppo": dict(color="#d62728", lw=1.2, label="PPO"),
         "skyhook_ref": dict(color="#1f77b4", lw=0.9, ls="--", label="Skyhook (hand-designed ref., not RL)")}
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 120,
                     "axes.spines.top": False, "axes.spines.right": False})
CAPTIONS = {}


def save(fig, out: Path, name: str, caption: str):
    fig.tight_layout()
    fig.savefig(out / f"{name}.png", dpi=300); fig.savefig(out / f"{name}.pdf")
    plt.close(fig)
    CAPTIONS[name] = caption
    print("wrote", out / f"{name}.png")


def training_curve(run: str, out: Path):
    mon = []
    for f in sorted(glob.glob(str(ROOT / "logs" / run / "monitor" / "*.monitor.csv"))):
        d = pd.read_csv(f, skiprows=1); mon.append(d)
    val = pd.read_csv(ROOT / "logs" / run / "validation_log.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.9))
    if mon:
        d = pd.concat(mon).sort_values("t"); d["steps"] = d["l"].cumsum()
        w = max(5, len(d) // 40)
        ax[0].plot(d["steps"], d["r"], color="#bbbbbb", lw=0.5, label="episode")
        ax[0].plot(d["steps"], d["r"].rolling(w, min_periods=1).mean(), color="#d62728", lw=1.4, label=f"rolling mean ({w})")
        ax[0].set_xlabel("environment steps"); ax[0].set_ylabel("training-episode return (raw reward)")
        ax[0].set_title("Training episodes ('train' road mixture)"); ax[0].legend(fontsize=7)
    ax[1].plot(val["timesteps"], val["val_return_mean"], "o-", color="#d62728", ms=3, label="PPO (deterministic)")
    ax[1].fill_between(val["timesteps"], val["val_return_mean"] - val["val_return_std"],
                       val["val_return_mean"] + val["val_return_std"], color="#d62728", alpha=0.15)
    ax[1].axhline(val["passive_return_mean"].iloc[0], color="#555555", ls="--", label="passive (same seeds)")
    ax[1].set_xlabel("environment steps"); ax[1].set_ylabel("validation return (mean ± std over seeds)")
    ax[1].set_title("Validation (fixed seeds < 100000)"); ax[1].legend(fontsize=7)
    save(fig, out, "fig1_training_curve",
         "Fig. 1: PPO learning curves. Left: raw return of training episodes (grey) with rolling mean. Right: periodic "
         "deterministic validation on fixed seeds from the training road mixture, against the passive suspension on the same "
         "seeds. Higher (less negative) return is better. Validation seeds are disjoint from the held-out evaluation seeds.")


def timeseries_fig(ts: pd.DataFrame, scenarios, signal, ylabel, name, title, caption, out, tmax, scale=1.0, controllers=None):
    controllers = controllers or [c for c in STYLE if c in ts.controller.unique()]
    fig, axes = plt.subplots(len(scenarios), 1, figsize=(7.2, 2.1 * len(scenarios)), sharex=False, squeeze=False)
    for ax, sc in zip(axes[:, 0], scenarios):
        d = ts[ts.scenario == sc]
        seed = d.seed.min()
        for c in controllers:
            x = d[(d.controller == c) & (d.seed == seed)]
            if tmax: x = x[x.time <= tmax]
            if len(x): ax.plot(x.time, scale * x[signal], **STYLE[c])
        ax.set_ylabel(ylabel); ax.set_title(f"{sc} (seed {seed})", fontsize=8, loc="left")
    axes[-1, 0].set_xlabel("time [s]"); axes[0, 0].legend(fontsize=7, ncol=3, loc="upper right")
    save(fig, out, name, caption)


def tyre_fig(ts, scenarios, out, tmax):
    ts = ts.copy(); ts["tyre_force_dynamic_kN"] = ts["tyre_force_dynamic"] / 1000.0
    timeseries_fig(ts, scenarios, "tyre_force_dynamic_kN", "dynamic tyre force Ft [kN]", "fig4_tyre_force_time", "",
                   "Fig. 4: Dynamic tyre force Ft = kt (zu - zr) over time (matched scenario and seed). Smaller excursions "
                   "indicate better road holding; the static load is 3.42 kN, so Ft <= -3.42 kN would be a contact-loss flag.", out, tmax)


def summary_fig(cmp_df: pd.DataFrame, summ: pd.DataFrame, out: Path):
    c = cmp_df[cmp_df.controller == "ppo"]
    metrics = [("rms_acc", "RMS acc."), ("peak_acc", "peak acc."), ("rms_defl", "RMS defl."),
               ("max_defl", "max defl."), ("rms_dtl", "RMS tyre load"), ("max_tyre_force", "max |Ft|")]
    scen = list(dict.fromkeys(c.scenario))
    fig, ax = plt.subplots(2, 1, figsize=(7.4, 6.0))
    w = 0.8 / len(scen)
    for i, sc in enumerate(scen):
        v = [c[(c.scenario == sc) & (c.metric == m)].improvement_pct_of_means.iloc[0] for m, _ in metrics]
        ax[0].bar(np.arange(len(metrics)) + i * w, v, w, label=sc)
    ax[0].axhline(0, color="k", lw=0.8)
    ax[0].set_xticks(np.arange(len(metrics)) + 0.4 - w / 2); ax[0].set_xticklabels([n for _, n in metrics])
    ax[0].set_ylabel("improvement vs passive [%]\n(positive = better)"); ax[0].legend(fontsize=6, ncol=2)
    ax[0].set_title("Change in physical metrics, PPO vs passive (means over seeds)", fontsize=9)
    s = summ[(summ.metric == "rms_acc") & summ.controller.isin(["passive", "ppo"])]
    x = np.arange(len(scen))
    for k, ctl in enumerate(["passive", "ppo"]):
        d = s[s.controller == ctl].set_index("scenario").loc[scen]
        ax[1].bar(x + (k - 0.5) * 0.38, d["mean"], 0.38, yerr=d["std"], capsize=2, color=STYLE[ctl]["color"], label=STYLE[ctl]["label"])
    ax[1].set_xticks(x); ax[1].set_xticklabels(scen, rotation=25, ha="right", fontsize=7)
    ax[1].set_ylabel("RMS sprung acceleration [m/s²]"); ax[1].legend(fontsize=7)
    ax[1].set_title("Absolute RMS acceleration (error bars: std over seeds)", fontsize=9)
    save(fig, out, "fig5_summary_metrics",
         "Fig. 5: Summary of physical metrics. Top: percentage improvement of PPO over the passive suspension "
         "(positive = PPO better; lower-is-better metrics are sign-flipped; undefined where passive ~ 0). Metrics trade off, so no single "
         "bar establishes superiority; negative bars are degradations. Bottom: absolute RMS acceleration with standard deviation across seeds.")


def effort_fig(ts, summ, scenarios, out, tmax):
    ts = ts[ts.controller.isin(["ppo", "skyhook_ref"])]
    timeseries_fig(ts, scenarios, "actuator_force", "actuator force u [N]", "fig6_actuator_force_time", "",
                   "Fig. 6: Actuator force over time on the same scenarios/seeds as Figs. 2-4. The actuator limit is ±1500 N "
                   "(the passive suspension applies 0 N and is therefore omitted).", out, tmax)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--scenarios", nargs="+", default=["rough_C_15", "bump_10", "heldout_test_mixture"])
    ap.add_argument("--tmax", type=float, default=None, help="optionally truncate time-series plots to [0, tmax] s")
    args = ap.parse_args()
    out = ROOT / "figures" / args.run; out.mkdir(parents=True, exist_ok=True)
    res = ROOT / "results" / args.run
    ts = pd.read_csv(res / "timeseries.csv"); cmp_df = pd.read_csv(res / "comparison.csv"); summ = pd.read_csv(res / "summary.csv")
    if (ROOT / "logs" / args.run / "validation_log.csv").exists():
        training_curve(args.run, out)
    sc = [s for s in args.scenarios if s in ts.scenario.unique()]
    timeseries_fig(ts, sc, "sprung_acceleration", "sprung acc. [m/s²]", "fig2_acceleration_time", "",
                   "Fig. 2: Sprung-mass vertical acceleration over time, passive vs PPO (and the hand-designed skyhook reference, "
                   "which is NOT an RL result), same road, speed, seed and initial state.", out, args.tmax)
    ts2 = ts.copy(); ts2["defl_mm"] = ts2.suspension_deflection * 1000
    timeseries_fig(ts2, sc, "defl_mm", "suspension deflection zs-zu [mm]", "fig3_deflection_time", "",
                   "Fig. 3: Suspension deflection zs - zu over time (matched conditions). Smaller excursions mean less rattle-space use.", out, args.tmax)
    tyre_fig(ts, sc, out, args.tmax)
    effort_fig(ts, summ, sc, out, args.tmax)
    summary_fig(cmp_df, summ, out)
    with open(out / "captions.md", "w") as f:
        for k, v in CAPTIONS.items(): f.write(f"**{k}** — {v}\n\n")


if __name__ == "__main__":
    main()
