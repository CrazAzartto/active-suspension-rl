"""Physical evaluation metrics computed from `environment.evaluation.rollout` trajectories.

Everything is derived from quantities exposed by Stage 1 (info dict); nothing is invented.
Tyre contact: Stage 1 flags contact loss when the tyre-load ratio 1 + Ft/((ms+mu)g) <= 0.
The linear tyre cannot physically lift off, so this is a *flag*, and it is reported
as such (min_tyre_load_ratio / contact_loss_frac), not as a guarantee.
"""
from __future__ import annotations

from typing import Dict

import numpy as np

# direction: "lower" / "higher" = desirable direction; "cost" = control effort (passive spends 0,
# so a percentage change versus passive is not meaningful).
METRIC_INFO: Dict[str, Dict[str, str]] = {
    "rms_acc":           {"dir": "lower",  "unit": "m/s^2", "desc": "RMS sprung-mass acceleration (comfort; unweighted, no ISO 2631 filter)"},
    "peak_acc":          {"dir": "lower",  "unit": "m/s^2", "desc": "peak |sprung-mass acceleration|"},
    "rms_defl":          {"dir": "lower",  "unit": "m",     "desc": "RMS suspension deflection |zs-zu| (rattle-space use)"},
    "max_defl":          {"dir": "lower",  "unit": "m",     "desc": "max |zs-zu|"},
    "rms_dtl":           {"dir": "lower",  "unit": "-",     "desc": "RMS dynamic tyre force / static load (road holding)"},
    "rms_tyre_force":    {"dir": "lower",  "unit": "N",     "desc": "RMS dynamic tyre force Ft"},
    "max_tyre_force":    {"dir": "lower",  "unit": "N",     "desc": "max |Ft|"},
    "max_tyre_defl":     {"dir": "lower",  "unit": "m",     "desc": "max |zu-zr|"},
    "min_tyre_load_ratio": {"dir": "higher", "unit": "-",   "desc": "min of 1+Ft/static load over the control steps (<=0 => contact-loss flag)"},
    "contact_loss_frac": {"dir": "lower",  "unit": "-",     "desc": "fraction of control steps with tyre-load ratio <= 0 in the step"},
    "travel_viol_frac":  {"dir": "lower",  "unit": "-",     "desc": "fraction of control steps with |zs-zu| > travel_limit (0.10 m)"},
    "rms_force":         {"dir": "cost",   "unit": "N",     "desc": "RMS actuator force (control effort)"},
    "peak_force":        {"dir": "cost",   "unit": "N",     "desc": "peak |actuator force|"},
    "sat_frac":          {"dir": "cost",   "unit": "-",     "desc": "fraction of control steps with the command saturated at +/-u_max"},
    "mean_power":        {"dir": "cost",   "unit": "W",     "desc": "mean mechanical actuator power u*(zs_dot-zu_dot) (negative = absorbing)"},
    "return":            {"dir": "higher", "unit": "-",     "desc": "cumulative episode reward (training objective, kept separate)"},
    "terminated_early":  {"dir": "lower",  "unit": "-",     "desc": "1 if a Stage 1 guard terminated the episode"},
}


def compute_metrics(traj: Dict[str, np.ndarray], static_tyre_load: float, u_max: float,
                    travel_limit_unused=None) -> Dict[str, float]:
    a = traj["sprung_acceleration"]
    d = traj["suspension_deflection"]
    ft = traj["tyre_force_dynamic"]
    u = traj["actuator_force"]
    sat = np.abs(u) >= u_max * (1.0 - 1e-9)
    return {
        "rms_acc": float(np.sqrt(np.mean(a ** 2))),
        "peak_acc": float(np.max(np.abs(a))),
        "rms_defl": float(np.sqrt(np.mean(d ** 2))),
        "max_defl": float(np.max(np.abs(d))),
        "rms_dtl": float(np.sqrt(np.mean(ft ** 2)) / static_tyre_load),
        "rms_tyre_force": float(np.sqrt(np.mean(ft ** 2))),
        "max_tyre_force": float(np.max(np.abs(ft))),
        "max_tyre_defl": float(np.max(np.abs(traj["tyre_deflection"]))),
        "min_tyre_load_ratio": float(np.min(traj["tyre_load_ratio"])),
        "contact_loss_frac": float(np.mean(traj["contact_loss_in_interval"])),
        "travel_viol_frac": float(np.mean(traj["travel_limit_exceeded_in_interval"])),
        "rms_force": float(np.sqrt(np.mean(u ** 2))),
        "peak_force": float(np.max(np.abs(u))),
        "sat_frac": float(np.mean(sat)),
        "mean_power": float(np.mean(traj["actuator_power"])),
        "return": float(np.sum(traj["reward"])),
        "terminated_early": float(bool(traj["terminated"])),
    }


def improvement_pct(metric: str, passive: float, other: float, eps: float = 1e-12) -> float:
    """Percentage IMPROVEMENT of `other` relative to `passive`; positive = better, negative = worse.

    lower-is-better : 100*(passive-other)/|passive|
    higher-is-better: 100*(other-passive)/|passive|
    Returns NaN when not meaningful: effort ("cost") metrics, or |passive| < eps.
    """
    direction = METRIC_INFO[metric]["dir"]
    if direction == "cost" or not np.isfinite(passive) or abs(passive) < eps:
        return float("nan")
    delta = (passive - other) if direction == "lower" else (other - passive)
    return float(100.0 * delta / abs(passive))
