"""Build every number, table and figure used by report/main.tex from the saved CSV/log files.

  python report/make_report_assets.py --run ppo_default
Writes report/{numbers.tex, results_table.tex, fig_curve.pdf, fig_time.pdf}.  Nothing is typed by hand.
"""
import argparse, json
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np, pandas as pd

R = Path(__file__).resolve().parent; ROOT = R.parent
ap = argparse.ArgumentParser(); ap.add_argument("--run", default="ppo_default"); ap.add_argument("--ts-scenario", default="rough_C_15")
ap.add_argument("--tmax", type=float, default=4.0); a = ap.parse_args()
res, logs = ROOT / "results" / a.run, ROOT / "logs" / a.run
cmp_ = pd.read_csv(res / "comparison.csv"); ep = pd.read_csv(res / "episodes.csv"); ts = pd.read_csv(res / "timeseries.csv")
val = pd.read_csv(logs / "validation_log.csv"); rt = pd.read_csv(res / "reward_terms.csv").set_index("controller")
tc = json.load(open(ROOT / "models" / a.run / "train_config.json")); st = json.load(open(ROOT / "models" / a.run / "training_status.json"))

def imp(sc, m, c="ppo"): return float(cmp_[(cmp_.scenario == sc) & (cmp_.controller == c) & (cmp_.metric == m)].improvement_pct_of_means.iloc[0])
def mean(sc, m, c): return float(ep[(ep.scenario == sc) & (ep.controller == c)][m].mean())
def n_ep(sc): return int((ep[(ep.scenario == sc) & (ep.controller == "ppo")]).shape[0])
def frac(sc): return float(cmp_[(cmp_.scenario == sc) & (cmp_.controller == "ppo") & (cmp_.metric == "rms_acc")].frac_seeds_better.iloc[0])

# ------------------------------------------------------------------ table
order = [("bump_10", "bump 10 m/s", "T"), ("rough_B_15", "ISO B 15 m/s", "T"), ("rough_C_15", "ISO C 15 m/s", "T"),
         ("smooth_A_15", "ISO A 15 m/s", "U"), ("flat_perturbed", "flat, +20 mm start", "U"),
         ("heldout_bump_tall", "tall bump 16 m/s", "H"), ("heldout_rough_D_10", "ISO D 10 m/s", "H"), ("heldout_test_mixture", "test mixture", "H")]
rows = []
for sc, lab, g in order:
    cl_p, cl_n = 100 * mean(sc, "contact_loss_frac", "passive"), 100 * mean(sc, "contact_loss_frac", "ppo")
    rows.append(f"{lab} ({g}) & {n_ep(sc)} & {mean(sc,'rms_acc','passive'):.3f} & {mean(sc,'rms_acc','ppo'):.3f} & "
                f"{imp(sc,'rms_acc'):+.1f} & {imp(sc,'peak_acc'):+.1f} & {imp(sc,'rms_defl'):+.1f} & "
                f"{imp(sc,'rms_dtl'):+.1f} & {imp(sc,'rms_acc','skyhook_ref'):+.1f} & "
                f"{mean(sc,'rms_force','ppo'):.0f} & {cl_p:.1f}/{cl_n:.1f}")
(R / "results_table.tex").write_text(" \\\\\n".join(rows))  # last row has no \\ ; main.tex adds it

# ------------------------------------------------------------------ numbers (macros)
best = val.loc[val.val_return_mean.idxmax()]
sc_tr, sc_ho = ["rough_B_15", "rough_C_15", "bump_10"], ["heldout_bump_tall", "heldout_rough_D_10", "heldout_test_mixture"]
rand = ["rough_B_15", "rough_C_15", "smooth_A_15", "heldout_rough_D_10", "heldout_test_mixture"]
M = {
 "TrainSteps": f"{int(st['timesteps']):,}".replace(",", "\\,"), "BestStep": f"{int(best.timesteps)//1000}", "WallTime": f"{st['wall_time_s']:.0f}",
 "ValBest": f"{best.val_return_mean:.0f}", "ValPassive": f"{best.passive_return_mean:.0f}",
 "AccHoLo": f"{min(imp(s,'rms_acc') for s in sc_ho):.0f}", "AccHoHi": f"{max(imp(s,'rms_acc') for s in sc_ho):.0f}",
 "AccRandLo": f"{min(imp(s,'rms_acc') for s in rand):.0f}", "AccRandHi": f"{max(imp(s,'rms_acc') for s in rand):.0f}",
 "DtlRandLo": f"{-max(imp(s,'rms_dtl') for s in rand):.0f}", "DtlRandHi": f"{-min(imp(s,'rms_dtl') for s in rand):.0f}",
 "DtlBumpOne": f"{imp('bump_10','rms_dtl'):+.1f}",
 "SatMax": f"{100*ep[ep.controller=='ppo'].sat_frac.max():.1f}", "PeakForceMax": f"{ep[ep.controller=='ppo'].peak_force.max():.0f}",
 "EarlyTerm": f"{int(ep.terminated_early.sum())}",
 "SkyAccC": f"{imp('rough_C_15','rms_acc','skyhook_ref'):.1f}", "SkyDtlC": f"{imp('rough_C_15','rms_dtl','skyhook_ref'):+.1f}",
 "TermAccP": f"{rt.loc['passive','acceleration']:.2f}", "TermAccN": f"{rt.loc['ppo','acceleration']:.2f}",
 "TermDefP": f"{rt.loc['passive','deflection']:.2f}", "TermDefN": f"{rt.loc['ppo','deflection']:.2f}",
 "TermTyreP": f"{rt.loc['passive','tyre_force']:.2f}", "TermTyreN": f"{rt.loc['ppo','tyre_force']:.2f}",
 "TermTotP": f"{rt.loc['passive','total']:.2f}", "TermTotN": f"{rt.loc['ppo','total']:.2f}",
 "ClDP": f"{100*mean('heldout_rough_D_10','contact_loss_frac','passive'):.1f}", "ClDN": f"{100*mean('heldout_rough_D_10','contact_loss_frac','ppo'):.1f}",
 "AllPaired": "all" if all(frac(s) == 1.0 for s in rand) else "not all",
 "NEpisodes": f"{len(ep[ep.controller=='ppo'])}",
}
# ------------------------------------------------------------------ multi-seed robustness (optional extra runs)
seed_runs = [r for r in ("ppo_seed1", "ppo_seed2") if (ROOT / "logs" / r / "validation_log.csv").exists()]
vals = {a.run: val, **{r: pd.read_csv(ROOT / "logs" / r / "validation_log.csv") for r in seed_runs}}
M["NSeeds"] = f"{len(vals)}"
M["ValBestAll"] = ", ".join(f"{v.val_return_mean.max():.0f}" for v in vals.values())
M["ValDipAll"] = f"{min(v.val_return_mean.min() for v in vals.values()):.0f}"
ss_path = ROOT / "results" / "seed_summary.csv"
if ss_path.exists():
    ss = pd.read_csv(ss_path)
    def ssv(m, col, scen): return ss[(ss.metric == m) & ss.scenario.isin(scen)][col]
    M["SeedAccRandLo"] = f"{ssv('rms_acc', 'min', rand).min():.0f}"; M["SeedAccRandHi"] = f"{ssv('rms_acc', 'max', rand).max():.0f}"
    M["SeedDtlRandLo"] = f"{-ssv('rms_dtl', 'max', rand).max():.0f}"; M["SeedDtlRandHi"] = f"{-ssv('rms_dtl', 'min', rand).min():.0f}"
(R / "numbers.tex").write_text("".join(f"\\newcommand{{\\N{k}}}{{{v}}}\n" for k, v in M.items()))

# ------------------------------------------------------------------ figures
plt.rcParams.update({"font.size": 6.5, "axes.grid": True, "grid.alpha": .3, "axes.spines.top": False, "axes.spines.right": False,
                     "font.family": "serif", "lines.linewidth": .9})
fig, ax = plt.subplots(figsize=(3.35, 1.45))
for (run, v), col in zip(vals.items(), ("#d62728", "#1f77b4", "#2ca02c")):
    sd = json.load(open(ROOT / "models" / run / "train_config.json"))["train_config"]["seed"]
    ax.plot(v.timesteps / 1e3, v.val_return_mean, "o-", ms=2, color=col, label=f"PPO seed {sd}")
ax.axhline(val.passive_return_mean.iloc[0], color="k", ls="--", label="passive")
lo = min(v.val_return_mean.min() for v in vals.values())
ax.set_xlabel("environment steps [$10^3$]"); ax.set_ylabel("validation return"); ax.set_ylim(lo - 300, 0)
ax.legend(fontsize=5.5, loc="lower right", ncol=2); fig.tight_layout(pad=.3); fig.savefig(R / "fig_curve.pdf"); plt.close(fig)

d = ts[ts.scenario == a.ts_scenario]; seed = d.seed.min()
panels = [("sprung_acceleration", "sprung acc. [m/s$^2$]", 1), ("suspension_deflection", "deflection $z_s-z_u$ [mm]", 1e3),
          ("tyre_force_dynamic", "dyn. tyre force $F_t$ [kN]", 1e-3), ("actuator_force", "actuator force $u$ [N]", 1)]
fig, axs = plt.subplots(1, 4, figsize=(7.2, 1.3))
for ax, (k, lab, sc) in zip(axs, panels):
    for c, col, ls in (("passive", "0.35", "-"), ("ppo", "#d62728", "-")):
        x = d[(d.controller == c) & (d.seed == seed) & (d.time <= a.tmax)]
        ax.plot(x.time, sc * x[k], color=col, ls=ls, label="passive" if c == "passive" else "PPO")
    ax.set_xlabel("time [s]"); ax.set_ylabel(lab)
axs[0].legend(fontsize=6, loc="lower left", ncol=2, framealpha=.8, borderpad=.2, handlelength=1.2); fig.tight_layout(pad=.3); fig.savefig(R / "fig_time.pdf"); plt.close(fig)
# summary bars: improvement vs passive (positive = better)
mets = [("rms_acc", "RMS acc"), ("peak_acc", "peak acc"), ("rms_defl", "RMS defl"), ("rms_dtl", "RMS tyre load")]
cols = ["#d62728", "#ff9896", "#1f77b4", "#2ca02c"]
fig, ax = plt.subplots(figsize=(3.35, 2.15)); h = 0.2
for j, ((m, lab), c) in enumerate(zip(mets, cols)):
    ax.barh(np.arange(len(order)) + (j - 1.5) * h, [imp(sc, m) for sc, _, _ in order], h, color=c, label=lab)
ax.set_yticks(np.arange(len(order))); ax.set_yticklabels([f"{l} ({g})" for _, l, g in order], fontsize=5.8); ax.invert_yaxis()
ax.axvline(0, color="k", lw=.7); ax.set_xlabel("PPO change vs passive [%] (positive = better)"); ax.legend(fontsize=5.8, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=4, frameon=False, columnspacing=1.0, handlelength=1.2)
fig.tight_layout(pad=.3); fig.savefig(R / "fig_bars.pdf"); plt.close(fig)
print((R / "numbers.tex").read_text()); print((R / "results_table.tex").read_text())
