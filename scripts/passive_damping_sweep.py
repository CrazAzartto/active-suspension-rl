"""Check that the default passive damper is not an artificially weak baseline.

Sweeps the passive damping coefficient cs (actuator force stays 0) on ISO class-B/C
roads and prints physical metrics and the reward return for each cs.  If a
different cs is clearly better for your purposes, use it for BOTH passive and
learned controllers (VehicleParams(cs=...)) so the comparison stays fair.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from environment import EnvConfig, VehicleParams  # noqa: E402
from environment.evaluation import PassiveController, evaluate  # noqa: E402

SEEDS = range(100000, 100010)
for road in ("rough_B", "rough_C"):
    print(f"\n=== {road}, speed 15 m/s, {len(SEEDS)} seeds ===")
    print(f"{'cs [N s/m]':>11}{'zeta':>7}{'rms_acc':>10}{'rms_defl':>10}{'rms_dtl':>9}{'return':>11}")
    for cs in (500, 750, 1000, 1500, 2000, 3000, 4000):
        cfg = EnvConfig(vehicle=VehicleParams(cs=float(cs)), speed=15.0)
        m = evaluate(PassiveController(), cfg, road=road, seeds=SEEDS)["mean"]
        print(f"{cs:>11}{cfg.vehicle.body_damping_ratio:>7.2f}{m['rms_acc']:>10.3f}"
              f"{m['rms_defl']:>10.4f}{m['rms_dtl']:>9.3f}{m['return']:>11.1f}")
