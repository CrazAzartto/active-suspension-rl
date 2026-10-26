"""Evaluate a trained PPO policy against the passive suspension under IDENTICAL conditions.

  python evaluate.py --model models/ppo_default/best_model.zip --run ppo_default
  python evaluate.py --model models/smoke/final_model.zip --run smoke --n-seeds 2 --no-skyhook

For every scenario and seed, all controllers are rolled out with the same env config,
road, speed, reset seed, reset options and initial state (road height is asserted
bit-identical across controllers).  PPO acts deterministically.

Writes results/<run>/:
  episodes.csv    one row per (scenario, controller, seed): metrics, return, vehicle params, model id
  summary.csv     mean/std over seeds per (scenario, controller, metric)
  comparison.csv  PPO (and skyhook) vs passive per scenario and metric: improvement %, paired stats
  timeseries.csv  full time series for the first `--ts-seeds` seeds of each scenario
  eval_config.json  scenarios, seeds, env config, model id
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from environment import ActiveSuspensionEnv
from environment.evaluation import PassiveController, SkyhookController, rollout
from training.common import (ROOT, build_env_config, config_to_dict, load_json, save_json,
                             vehicle_columns)
from training.metrics import METRIC_INFO, compute_metrics, improvement_pct

TS_COLS = ["road_height", "sprung_acceleration", "suspension_deflection", "tyre_force_dynamic",
           "tyre_load_ratio", "actuator_force", "reward"]


def evaluate_scenarios(controllers: Dict[str, Any], env_cfg, scenarios: List[Dict[str, Any]],
                       seed_start: int, model_id: str, n_seeds_override: int | None = None,
                       ts_seeds: int = 1):
    """Return (episodes_df, timeseries_df). `controllers` maps name -> policy (first must be 'passive')."""
    ep_rows, ts_rows = [], []
    veh = vehicle_columns(env_cfg)
    u_max, f_static = env_cfg.actuator.u_max, env_cfg.vehicle.static_tyre_load
    for sc in scenarios:
        cfg = env_cfg.with_(speed=sc.get("speed", env_cfg.speed))
        env = ActiveSuspensionEnv(cfg, road=sc["road"])
        n = n_seeds_override if (n_seeds_override and not sc.get("deterministic")) else sc.get("n_seeds", 10)
        opts = copy.deepcopy(sc.get("options", {}))
        for i in range(n):
            seed = seed_start + i
            ref_road, ref_label = None, None
            for name, pol in controllers.items():
                traj = rollout(env, pol, seed=seed, options=opts)
                rs = env.road_sample
                if ref_road is None:
                    ref_road, ref_label = traj["road_height"].copy(), rs.label
                else:  # integrity check: matched conditions
                    assert np.array_equal(ref_road, traj["road_height"]) and rs.label == ref_label, \
                        f"road mismatch between controllers in {sc['name']} seed {seed}"
                m = compute_metrics(traj, f_static, u_max)
                ep_rows.append({"scenario": sc["name"], "group": sc.get("group", ""), "controller": name,
                                "seed": seed, "road_label": rs.label, "speed": rs.speed,
                                "duration_s": float(traj["time"][-1]), "n_steps": len(traj["time"]),
                                "initial_state_offset": str(opts.get("initial_state", "none")),
                                **veh, **m, "model_id": model_id if name == "ppo" else "n/a"})
                if i < ts_seeds:
                    df = pd.DataFrame({c: traj[c] for c in ["time"] + TS_COLS})
                    df.insert(0, "seed", seed); df.insert(0, "controller", name); df.insert(0, "scenario", sc["name"])
                    ts_rows.append(df)
    return pd.DataFrame(ep_rows), pd.concat(ts_rows, ignore_index=True)


def summarize(ep: pd.DataFrame) -> pd.DataFrame:
    metrics = list(METRIC_INFO)
    g = ep.groupby(["scenario", "controller"], sort=False)
    mean, std = g[metrics].mean(), g[metrics].std(ddof=0)
    n = g.size().rename("n_episodes")
    out = []
    for m in metrics:
        d = pd.DataFrame({"mean": mean[m], "std": std[m]}).join(n).reset_index()
        d.insert(2, "metric", m); out.append(d)
    s = pd.concat(out, ignore_index=True)
    s["desirable"] = s["metric"].map(lambda k: METRIC_INFO[k]["dir"])
    s["unit"] = s["metric"].map(lambda k: METRIC_INFO[k]["unit"])
    return s


def compare(ep: pd.DataFrame, baseline: str = "passive") -> pd.DataFrame:
    """Per scenario/metric: improvement of each non-baseline controller vs baseline (positive = better)."""
    rows = []
    for sc, d in ep.groupby("scenario", sort=False):
        base = d[d.controller == baseline].set_index("seed")
        for ctl in [c for c in d.controller.unique() if c != baseline]:
            oth = d[d.controller == ctl].set_index("seed").loc[base.index]
            for m, info in METRIC_INFO.items():
                pb, po = float(base[m].mean()), float(oth[m].mean())
                diff = oth[m].to_numpy() - base[m].to_numpy()
                n = len(diff)
                se = float(np.std(diff, ddof=1) / np.sqrt(n)) if n > 1 else float("nan")
                better = (diff < 0) if info["dir"] == "lower" else (diff > 0) if info["dir"] == "higher" else None
                rows.append({"scenario": sc, "controller": ctl, "metric": m, "desirable": info["dir"], "unit": info["unit"],
                             "passive_mean": pb, "controller_mean": po,
                             "improvement_pct_of_means": improvement_pct(m, pb, po),
                             "paired_mean_diff": float(diff.mean()),
                             "paired_diff_ci95_halfwidth": 1.96 * se if n > 1 else float("nan"),
                             "frac_seeds_better": float(np.mean(better)) if better is not None else float("nan"),
                             "n_pairs": n,
                             "note": "positive improvement_pct = better than passive; NaN = not meaningful (effort metric or passive ~ 0)"})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="path to a Stable-Baselines3 PPO .zip")
    ap.add_argument("--run", required=True, help="results/<run>/ output folder name")
    ap.add_argument("--scenarios", default="configs/eval_scenarios.json")
    ap.add_argument("--env-overrides", default=None, help="JSON file with env_overrides (default: Stage 1 defaults)")
    ap.add_argument("--n-seeds", type=int, default=None, help="override n_seeds of stochastic scenarios")
    ap.add_argument("--ts-seeds", type=int, default=1, help="how many seeds per scenario get saved time series")
    ap.add_argument("--no-skyhook", action="store_true", help="omit the hand-designed skyhook reference")
    args = ap.parse_args()

    from stable_baselines3 import PPO
    sc_cfg = load_json(args.scenarios)
    overrides = load_json(args.env_overrides) if args.env_overrides else None
    if overrides is None:  # use the overrides the model was trained with, so vehicle/reward match
        tc = Path(args.model).parent / "train_config.json"
        if tc.exists():
            overrides = load_json(tc)["train_config"].get("env_overrides", {})
            overrides = {k: v for k, v in overrides.items() if k != "timing"}  # evaluate on full 10 s episodes
    env_cfg = build_env_config(overrides)
    model = PPO.load(args.model, device="cpu")
    mp = Path(args.model).resolve()
    model_id = str(mp.relative_to(ROOT)) if mp.is_relative_to(ROOT) else str(mp)
    controllers: Dict[str, Any] = {"passive": PassiveController(), "ppo": model}
    if not args.no_skyhook:
        controllers["skyhook_ref"] = SkyhookController(env_cfg)

    ep, ts = evaluate_scenarios(controllers, env_cfg, sc_cfg["scenarios"], sc_cfg["seed_start"], model_id,
                                args.n_seeds, args.ts_seeds)
    out = ROOT / "results" / args.run
    out.mkdir(parents=True, exist_ok=True)
    ep.to_csv(out / "episodes.csv", index=False)
    summarize(ep).to_csv(out / "summary.csv", index=False)
    cmp_df = compare(ep)
    cmp_df.to_csv(out / "comparison.csv", index=False)
    ts.to_csv(out / "timeseries.csv", index=False)
    save_json({"model_id": model_id, "scenarios": sc_cfg, "env_config": config_to_dict(env_cfg),
               "controllers": list(controllers), "n_seeds_override": args.n_seeds}, out / "eval_config.json")

    show = ["rms_acc", "peak_acc", "rms_defl", "max_defl", "rms_dtl", "rms_force", "return"]
    print(f"\nResults in {out}\nMean over seeds (lower is better except return):")
    piv = ep.groupby(["scenario", "controller"], sort=False)[show].mean()
    print(piv.round(4).to_string())
    p = cmp_df[(cmp_df.controller == "ppo") & cmp_df.metric.isin(["rms_acc", "rms_defl", "rms_dtl", "return"])]
    print("\nPPO improvement vs passive [%] (positive = better):")
    print(p.pivot(index="scenario", columns="metric", values="improvement_pct_of_means").round(2).to_string())


if __name__ == "__main__":
    main()
