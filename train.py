"""Train PPO (Stable-Baselines3) on the Stage 1 ActiveSuspensionEnv.

  python train.py --config configs/ppo_default.json            # full run
  python train.py --config configs/ppo_smoke.json              # ~1 min integration smoke run
  python train.py --config configs/ppo_default.json --seed 1 --run-name ppo_seed1

Outputs
  models/<run>/final_model.zip, best_model.zip (best validation return), checkpoints/*.zip,
  vecnormalize.pkl (reward-normalisation statistics, only needed to CONTINUE training),
  train_config.json (everything needed to reproduce)
  logs/<run>/monitor/*.monitor.csv (raw training-episode rewards), progress.csv (SB3 logger),
  validation_log.csv (periodic validation vs passive on fixed seeds < 100000)
Observations and actions are passed to the agent unchanged.  Reward is normalised only
by VecNormalize(norm_obs=False, norm_reward=True) for optimisation stability; all logged
returns are raw environment rewards.
"""
from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

import numpy as np
import torch

from training.common import (ROOT, build_env_config, config_to_dict, git_commit, load_json,
                             make_vec_env, save_json)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/ppo_default.json")
    ap.add_argument("--seed", type=int, default=None, help="override config seed")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--total-timesteps", type=int, default=None)
    ap.add_argument("--n-envs", type=int, default=None)
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    import stable_baselines3 as sb3
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList, CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.utils import set_random_seed
    from stable_baselines3.common.vec_env import VecNormalize

    cfg_json = load_json(args.config)
    if args.seed is not None: cfg_json["seed"] = args.seed
    if args.run_name: cfg_json["run_name"] = args.run_name
    if args.total_timesteps: cfg_json["total_timesteps"] = args.total_timesteps
    if args.n_envs: cfg_json["n_envs"] = args.n_envs
    if args.device: cfg_json["ppo"]["device"] = args.device
    run, seed = cfg_json["run_name"], int(cfg_json["seed"])
    if seed >= 100000:
        raise ValueError("training seeds must stay below 100000 (>= 100000 is reserved for held-out evaluation)")
    hp, val = cfg_json["ppo"], cfg_json["validation"]

    env_cfg = build_env_config({**cfg_json.get("env_overrides", {}), "road": cfg_json.get("train_road", "train")})
    model_dir, log_dir = ROOT / "models" / run, ROOT / "logs" / run
    model_dir.mkdir(parents=True, exist_ok=True); log_dir.mkdir(parents=True, exist_ok=True)

    set_random_seed(seed)
    torch.set_num_threads(1)  # small MLPs: more threads only add overhead
    venv = make_vec_env(env_cfg, cfg_json["n_envs"], seed, monitor_dir=str(log_dir / "monitor"))
    if cfg_json.get("normalize_reward", True):
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=hp["gamma"])

    act = {"tanh": torch.nn.Tanh, "relu": torch.nn.ReLU}[hp["activation"]]
    model = PPO(
        "MlpPolicy", venv, learning_rate=hp["learning_rate"], n_steps=hp["n_steps"],
        batch_size=hp["batch_size"], n_epochs=hp["n_epochs"], gamma=hp["gamma"],
        gae_lambda=hp["gae_lambda"], clip_range=hp["clip_range"], ent_coef=hp["ent_coef"],
        vf_coef=hp["vf_coef"], max_grad_norm=hp["max_grad_norm"], seed=seed, device=hp["device"],
        policy_kwargs=dict(net_arch=dict(pi=hp["net_arch"], vf=hp["net_arch"]), activation_fn=act,
                           log_std_init=hp["log_std_init"]),
        verbose=0)
    model.set_logger(configure(str(log_dir), ["csv"]))

    save_json({
        "train_config": cfg_json, "env_config_resolved": config_to_dict(env_cfg),
        "versions": {"python": platform.python_version(), "torch": torch.__version__,
                     "sb3": sb3.__version__, "numpy": np.__version__, "git_commit": git_commit()},
        "command": f"python train.py --config {args.config} --seed {seed} --run-name {run}"
                   f" --total-timesteps {cfg_json['total_timesteps']} --n-envs {cfg_json['n_envs']}",
    }, model_dir / "train_config.json")

    from training.callbacks import ValidationCallback
    callbacks = CallbackList([
        CheckpointCallback(max(1, cfg_json["checkpoint_freq"] // cfg_json["n_envs"]),
                           save_path=str(model_dir / "checkpoints"), name_prefix="ppo"),
        ValidationCallback(env_cfg, val["seeds"], val["eval_freq"], log_dir / "validation_log.csv",
                           model_dir / "best_model", road=val.get("road", "train")),
    ])
    print(f"[train] run={run} seed={seed} steps={cfg_json['total_timesteps']} envs={cfg_json['n_envs']}", flush=True)
    t0 = time.time()
    model.learn(total_timesteps=cfg_json["total_timesteps"], callback=callbacks)
    model.save(str(model_dir / "final_model"))
    if isinstance(venv, VecNormalize):
        venv.save(str(model_dir / "vecnormalize.pkl"))
    save_json({"training_completed": True, "timesteps": int(model.num_timesteps),
               "wall_time_s": round(time.time() - t0, 1)}, model_dir / "training_status.json")
    print(f"[train] completed {model.num_timesteps} steps in {time.time() - t0:.0f}s -> {model_dir}", flush=True)


if __name__ == "__main__":
    main()
