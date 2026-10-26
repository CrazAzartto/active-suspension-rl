"""Compare controllers under IDENTICAL conditions (same config, road, speed, seeds).

Prints physical metrics (not just reward).  Passive = zero force.  Skyhook is only an
optional hand-designed reference showing that the action channel can help; the RL agent
of Stage 2 should be added to the `controllers` dict the same way
(any object with .predict(obs, deterministic=True), e.g. a Stable-Baselines3 model).

  python scripts/compare_controllers.py --road rough_C --speed 15
  python scripts/compare_controllers.py --road test            # held-out mixture
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from environment import EnvConfig  # noqa: E402
from environment.evaluation import PassiveController, SkyhookController, evaluate, format_table  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road", default="rough_C")
    ap.add_argument("--speed", type=float, default=15.0)
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=100000, help="held-out evaluation seeds start here")
    args = ap.parse_args()

    cfg = EnvConfig(speed=args.speed)
    seeds = range(args.seed_start, args.seed_start + args.n_seeds)
    controllers = {"passive": PassiveController(), "skyhook_ref": SkyhookController(cfg)}
    results = {n: evaluate(c, cfg, road=args.road, seeds=seeds) for n, c in controllers.items()}
    print(f"road={args.road} speed={args.speed} seeds={seeds.start}..{seeds.stop - 1}")
    print(format_table(results))


if __name__ == "__main__":
    main()
