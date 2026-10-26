"""Periodic validation callback.

Evaluates the CURRENT policy deterministically on fixed validation seeds drawn from the
training road mixture (seeds < 100000, disjoint from the held-out test seeds >= 100000),
logs physical metrics next to the passive baseline on the SAME seeds, and keeps the best
model (by mean episode return).  The held-out seeds/roads used in evaluate.py are never
used for model selection.
"""
from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Sequence

from stable_baselines3.common.callbacks import BaseCallback

from environment import EnvConfig
from environment.evaluation import PassiveController, evaluate


class ValidationCallback(BaseCallback):
    FIELDS = ["timesteps", "wall_time_s", "val_return_mean", "val_return_std", "val_rms_acc",
              "val_rms_defl", "val_rms_dtl", "val_rms_force", "val_terminated_early",
              "passive_return_mean", "passive_rms_acc", "passive_rms_defl", "passive_rms_dtl", "is_best"]

    def __init__(self, cfg: EnvConfig, seeds: Sequence[int], eval_freq: int, log_path: str,
                 best_model_path: str, road="train", verbose: int = 1):
        super().__init__(verbose)
        self.cfg, self.seeds, self.road = cfg, list(seeds), road
        self.eval_freq = max(1, int(eval_freq))
        self.log_path, self.best_model_path = Path(log_path), Path(best_model_path)
        self.best = -float("inf")
        self._t0 = time.time()
        self._passive = None
        self._last_eval_at = -1

    def _on_training_start(self) -> None:
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.log_path, "w", newline="") as f:
            csv.writer(f).writerow(self.FIELDS)
        self._passive = evaluate(PassiveController(), self.cfg, road=self.road, seeds=self.seeds)["mean"]
        self._run()  # untrained-policy reference point at step 0

    def _on_step(self) -> bool:
        if self.num_timesteps - max(self._last_eval_at, 0) >= self.eval_freq:
            self._run()
        return True

    def _on_training_end(self) -> None:
        if self._last_eval_at != self.num_timesteps:
            self._run()

    def _run(self) -> None:
        res = evaluate(self.model, self.cfg, road=self.road, seeds=self.seeds)
        m, s, p = res["mean"], res["std"], self._passive
        is_best = m["return"] > self.best
        if is_best:
            self.best = m["return"]
            self.model.save(str(self.best_model_path))
        self._last_eval_at = self.num_timesteps
        row = [self.num_timesteps, round(time.time() - self._t0, 1), m["return"], s["return"], m["rms_acc"],
               m["rms_defl"], m["rms_dtl"], m["rms_force"], m["terminated_early"],
               p["return"], p["rms_acc"], p["rms_defl"], p["rms_dtl"], int(is_best)]
        with open(self.log_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
        if self.verbose:
            print(f"[val] t={self.num_timesteps:>8d}  return={m['return']:9.1f} (passive {p['return']:9.1f})  "
                  f"rms_acc={m['rms_acc']:.3f} (passive {p['rms_acc']:.3f}){'  *best*' if is_best else ''}", flush=True)
