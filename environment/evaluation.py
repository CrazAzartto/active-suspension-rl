"""Controllers and physical evaluation metrics (independent of the reward).

Policies are anything with either
  * `predict(obs, deterministic=True) -> (action, state)`   (Stable-Baselines3 style), or
  * a plain callable `obs -> action`.

`rollout` / `evaluate` reset the environment with an explicit seed and options,
so different policies can be compared under IDENTICAL road, speed, initial state
and vehicle parameters (the road/speed/initial state are drawn only in reset()).
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np

from .config import EnvConfig
from .suspension_env import ActiveSuspensionEnv


class PassiveController:
    """Passive suspension: actuator force is always zero."""

    def predict(self, obs, deterministic: bool = True):
        return np.zeros(1, dtype=np.float32), None

    def __call__(self, obs):
        return np.zeros(1, dtype=np.float32)


class SkyhookController:
    """Optional reference active controller (NOT the RL agent; used only to show
    that the action channel can improve on passive and for sanity tests).

    Uses the absolute sprung-mass velocity reconstructed from the observation:
        zs_dot = (zs_dot - zu_dot) + zu_dot          (obs[2]*scale + obs[4]*scale)
    and applies u = -gain * zs_dot, saturated by the environment.
    """

    def __init__(self, config: EnvConfig, gain: float = 1500.0):
        self.s, self.u_max, self.gain = config.obs_scales, config.actuator.u_max, gain

    def __call__(self, obs):
        zs_dot = obs[2] * self.s.defl_rate + obs[4] * self.s.unsprung_vel
        return np.array([np.clip(-self.gain * zs_dot / self.u_max, -1.0, 1.0)], dtype=np.float32)

    def predict(self, obs, deterministic: bool = True):
        return self(obs), None


def _act(policy, obs):
    if hasattr(policy, "predict"):
        a, _ = policy.predict(obs, deterministic=True)
        return a
    return policy(obs)


def rollout(env: ActiveSuspensionEnv, policy, seed: int,
            options: Optional[Dict[str, Any]] = None) -> Dict[str, np.ndarray]:
    """Run one episode; return time series of physical quantities (and reward)."""
    obs, info = env.reset(seed=seed, options=options)
    keys = ["sprung_acceleration", "suspension_deflection", "suspension_velocity",
            "tyre_deflection", "tyre_force_dynamic", "tyre_load_ratio", "actuator_force",
            "actuator_power", "road_height"]
    rec: Dict[str, List[float]] = {k: [] for k in keys}
    rec["reward"], rec["contact_loss_in_interval"], rec["travel_limit_exceeded_in_interval"] = [], [], []
    rec["time"] = []
    done = False
    while not done:
        obs, r, term, trunc, info = env.step(_act(policy, obs))
        for k in keys:
            rec[k].append(info[k])
        rec["reward"].append(r)
        rec["contact_loss_in_interval"].append(info["contact_loss_in_interval"])
        rec["travel_limit_exceeded_in_interval"].append(info["travel_limit_exceeded_in_interval"])
        rec["time"].append(info["time"])
        done = term or trunc
    out = {k: np.asarray(v, dtype=np.float64) for k, v in rec.items()}
    out["terminated"] = np.asarray(term)
    out["termination_reason"] = np.asarray(str(info["termination_reason"]))
    return out


def episode_metrics(traj: Dict[str, np.ndarray], static_tyre_load: float) -> Dict[str, float]:
    """Physical evaluation metrics for one episode (all in SI units).

    rms_acc            RMS sprung acceleration [m/s^2] (unweighted: no ISO 2631 filter)
    peak_acc           max |sprung acceleration| [m/s^2]
    rms_defl, max_defl RMS / max |zs-zu| [m]
    rms_dtl            RMS dynamic tyre load / static load [-] (road-holding index)
    contact_loss_frac  fraction of control steps with the tyre-load ratio <= 0 in the step
    travel_viol_frac   fraction of control steps with |zs-zu| above SafetyParams.travel_limit
    rms_force, mean_power  actuator RMS force [N] and mean mechanical power u*(zs_dot-zu_dot) [W]
    return             sum of rewards (kept SEPARATE from the physical metrics)
    """
    a = traj["sprung_acceleration"]
    d = traj["suspension_deflection"]
    ft = traj["tyre_force_dynamic"]
    return {
        "rms_acc": float(np.sqrt(np.mean(a**2))),
        "peak_acc": float(np.max(np.abs(a))),
        "rms_defl": float(np.sqrt(np.mean(d**2))),
        "max_defl": float(np.max(np.abs(d))),
        "rms_dtl": float(np.sqrt(np.mean(ft**2)) / static_tyre_load),
        "contact_loss_frac": float(np.mean(traj["contact_loss_in_interval"])),
        "travel_viol_frac": float(np.mean(traj["travel_limit_exceeded_in_interval"])),
        "rms_force": float(np.sqrt(np.mean(traj["actuator_force"] ** 2))),
        "mean_power": float(np.mean(traj["actuator_power"])),
        "return": float(np.sum(traj["reward"])),
        "terminated_early": float(bool(traj["terminated"])),
    }


def evaluate(policy, config: Optional[EnvConfig] = None, road: Any = None,
             seeds: Iterable[int] = range(100000, 100010),
             options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate one policy over several seeds under a given road spec.

    Returns {"per_episode": [metrics...], "mean": {...}, "std": {...}}.
    Use the SAME config/road/seeds/options for passive and learned controllers.
    """
    env = ActiveSuspensionEnv(config=config, road=road)
    opts = dict(options or {})
    rows = []
    for s in seeds:
        traj = rollout(env, policy, int(s), opts)
        rows.append(episode_metrics(traj, env.config.vehicle.static_tyre_load))
    keys = rows[0].keys()
    mean = {k: float(np.mean([r[k] for r in rows])) for k in keys}
    std = {k: float(np.std([r[k] for r in rows])) for k in keys}
    return {"per_episode": rows, "mean": mean, "std": std}


def format_table(results: Dict[str, Dict[str, Any]]) -> str:
    """Pretty-print {controller_name: evaluate(...)} as a text table of means."""
    cols = ["rms_acc", "peak_acc", "rms_defl", "max_defl", "rms_dtl",
            "contact_loss_frac", "travel_viol_frac", "rms_force", "return"]
    head = f"{'controller':<14}" + "".join(f"{c:>18}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, res in results.items():
        lines.append(f"{name:<14}" + "".join(f"{res['mean'][c]:>18.4f}" for c in cols))
    return "\n".join(lines)
