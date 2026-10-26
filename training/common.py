"""Shared helpers: JSON configs, EnvConfig construction, vectorised-env factory.

Training-only settings (PPO hyper-parameters, seeds, logging) live in JSON files under
configs/ and are kept apart from the physical parameters of the environment, which are
only ever expressed as overrides of the Stage 1 dataclasses (`env_overrides`).
"""
from __future__ import annotations

import dataclasses
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from environment import (ActiveSuspensionEnv, ActuatorParams, EnvConfig, RewardParams,
                         SafetyParams, TimingParams, VehicleParams)

ROOT = Path(__file__).resolve().parent.parent

_GROUPS = {"vehicle": VehicleParams, "actuator": ActuatorParams, "timing": TimingParams,
           "reward": RewardParams, "safety": SafetyParams}


def load_json(path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj: Any, path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def build_env_config(overrides: Optional[Dict[str, Any]] = None) -> EnvConfig:
    """Stage 1 EnvConfig with optional overrides, e.g.
    {"vehicle": {"cs": 1500.0}, "reward": {"w_acc": 2.0}, "road": "train", "speed": 15.0}.
    No override => exactly EnvConfig() (the Stage 1 defaults)."""
    o = dict(overrides or {})
    kwargs: Dict[str, Any] = {}
    for name, cls in _GROUPS.items():
        if name in o:
            kwargs[name] = cls(**o.pop(name))
    for k in ("road", "speed", "initial_state_noise"):
        if k in o:
            kwargs[k] = o.pop(k)
    if o:
        raise ValueError(f"unknown env_overrides keys: {sorted(o)}")
    return EnvConfig(**kwargs)


def config_to_dict(cfg: EnvConfig) -> Dict[str, Any]:
    """Full, JSON-serialisable record of an EnvConfig (for saving next to a model/results)."""
    d = dataclasses.asdict(cfg)
    d["road"] = cfg.road if isinstance(cfg.road, (str, list, dict)) else str(cfg.road)
    return d


def vehicle_columns(cfg: EnvConfig) -> Dict[str, float]:
    return {f"veh_{k}": v for k, v in dataclasses.asdict(cfg.vehicle).items()}


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def make_vec_env(cfg: EnvConfig, n_envs: int, seed: int, monitor_dir: Optional[str] = None):
    """DummyVecEnv of `n_envs` ActiveSuspensionEnv (env i is first reset with seed+i; later
    resets continue each env's own RNG stream => reproducible).  Observations/actions are
    NOT wrapped or altered.  Monitor records the RAW (un-normalised) episode reward."""
    from stable_baselines3.common.env_util import make_vec_env as sb3_make_vec_env
    from stable_baselines3.common.vec_env import DummyVecEnv
    if monitor_dir:
        Path(monitor_dir).mkdir(parents=True, exist_ok=True)
    return sb3_make_vec_env(lambda: ActiveSuspensionEnv(cfg), n_envs=n_envs, seed=seed,
                            vec_env_cls=DummyVecEnv, monitor_dir=monitor_dir)
