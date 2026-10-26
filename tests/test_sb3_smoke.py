"""Stable-Baselines3 PPO smoke test (skipped automatically if SB3 is not installed).

Only verifies that the environment works unchanged with SB3 (check_env, a few PPO
updates, model.predict inside the evaluation helpers).  It does NOT test learning quality.
"""
import numpy as np
import pytest

sb3 = pytest.importorskip("stable_baselines3")

from stable_baselines3 import PPO  # noqa: E402
from stable_baselines3.common.env_checker import check_env  # noqa: E402
from stable_baselines3.common.env_util import make_vec_env  # noqa: E402

from environment import ActiveSuspensionEnv, EnvConfig, TimingParams  # noqa: E402
from environment.evaluation import evaluate  # noqa: E402


def test_sb3_check_env():
    check_env(ActiveSuspensionEnv(), warn=True)


def test_ppo_trains_and_is_evaluable():
    cfg = EnvConfig(timing=TimingParams(episode_duration=2.0))
    venv = make_vec_env(lambda: ActiveSuspensionEnv(cfg), n_envs=2, seed=0)
    model = PPO("MlpPolicy", venv, n_steps=128, batch_size=64, n_epochs=2, device="cpu", seed=0, verbose=0)
    model.learn(total_timesteps=512)
    res = evaluate(model, cfg, road="rough_C", seeds=range(100000, 100002))
    assert np.isfinite(res["mean"]["rms_acc"]) and np.isfinite(res["mean"]["return"])
