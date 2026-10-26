"""Mean per-step reward terms (acceleration, deflection, tyre, effort, rate) for passive vs PPO.

  python scripts/reward_term_breakdown.py --model models/ppo_default/best_model.zip --run ppo_default
Writes results/<run>/reward_terms.csv.  Same road/seeds/conditions for both controllers.
"""
import argparse, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import numpy as np, pandas as pd
from stable_baselines3 import PPO
from environment import ActiveSuspensionEnv, EnvConfig
from environment.evaluation import PassiveController

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True); ap.add_argument("--run", required=True)
ap.add_argument("--road", default="rough_C"); ap.add_argument("--speed", type=float, default=15.0)
ap.add_argument("--n-seeds", type=int, default=10)
a = ap.parse_args()
model = PPO.load(a.model, device="cpu")
env = ActiveSuspensionEnv(EnvConfig(speed=a.speed), road=a.road)
rows = []
for name, pol in (("passive", PassiveController()), ("ppo", model)):
    for s in range(100000, 100000 + a.n_seeds):
        obs, _ = env.reset(seed=s); done = False; acc = {}; n = 0
        while not done:
            act, _ = pol.predict(obs, deterministic=True)
            obs, r, te, tr, info = env.step(act); done = te or tr; n += 1
            for k, v in info["reward_terms"].items(): acc[k] = acc.get(k, 0.0) + v
        rows.append({"controller": name, "seed": s, **{k: v / n for k, v in acc.items()}})
df = pd.DataFrame(rows); df["total"] = df[["acceleration", "deflection", "tyre_force", "effort", "rate"]].sum(axis=1)
out = df.groupby("controller").mean(numeric_only=True).drop(columns="seed")
out.insert(0, "road", f"{a.road}@{a.speed}"); out.to_csv(os.path.join("results", a.run, "reward_terms.csv"))
print(out.round(4).to_string())
