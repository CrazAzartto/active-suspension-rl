"""Run the passive suspension (actuator force = 0) and print physical metrics.

Examples:
  python scripts/run_passive.py
  python scripts/run_passive.py --road rough_C --speed 15 --seed 3
  python scripts/run_passive.py --road bump --speed 10 --plot passive_bump.png
  python scripts/run_passive.py --road test --seed 100000      # held-out road mixture
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from environment import ActiveSuspensionEnv, EnvConfig  # noqa: E402
from environment.evaluation import PassiveController, episode_metrics, rollout  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--road", default="rough_C", help="flat|bump|rough|rough_A..rough_E|train|test")
    ap.add_argument("--speed", type=float, default=15.0, help="vehicle speed [m/s] (ignored by train/test presets)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot", default=None, help="optional PNG path (needs matplotlib)")
    args = ap.parse_args()

    env = ActiveSuspensionEnv(EnvConfig(), road=args.road, speed=args.speed)
    traj = rollout(env, PassiveController(), seed=args.seed)
    rs = env.road_sample
    print(f"road: {rs.label}   speed: {rs.speed:.2f} m/s   seed: {args.seed}")
    for k, v in episode_metrics(traj, env.config.vehicle.static_tyre_load).items():
        print(f"  {k:<20}{v:>12.5f}")
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        t = traj["time"]
        fig, ax = plt.subplots(4, 1, figsize=(8, 8), sharex=True)
        ax[0].plot(t, traj["road_height"]); ax[0].set_ylabel("road zr [m]")
        ax[1].plot(t, traj["sprung_acceleration"]); ax[1].set_ylabel("zs_ddot [m/s^2]")
        ax[2].plot(t, traj["suspension_deflection"]); ax[2].set_ylabel("zs-zu [m]")
        ax[3].plot(t, traj["tyre_load_ratio"]); ax[3].axhline(0, color="r", lw=0.8)
        ax[3].set_ylabel("tyre load ratio"); ax[3].set_xlabel("time [s]")
        fig.tight_layout(); fig.savefig(args.plot, dpi=120)
        print("saved", args.plot)


if __name__ == "__main__":
    main()
