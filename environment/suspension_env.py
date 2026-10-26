"""Gymnasium environment: active quarter-car suspension.

See INTERFACE_CONTRACT.md for the full specification.  Summary:

* action      : Box(-1, 1, (1,), float32)  -> actuator force u = a * u_max [N]
* observation : Box(-inf, inf, (6,), float32), normalised, order:
      0  sprung-mass acceleration  zs_ddot / acc_scale
      1  suspension deflection     (zs - zu) / defl_scale
      2  suspension velocity       (zs_dot - zu_dot) / defl_rate_scale
      3  tyre deflection           (zu - zr) / tyre_defl_scale
      4  unsprung-mass velocity    zu_dot / unsprung_vel_scale
      5  previous action           a_{k-1}  (already in [-1, 1])
* reward      : see _reward(); negative weighted mean of squared normalised quantities
* termination : only by guards (non-finite state / hard limits); time limit => truncated

Physical state vs observation
-----------------------------
The physical state is x = [zs, zs_dot, zu, zu_dot] plus the road (zr, zr_dot) and
the road's future.  The agent sees none of zs, zu, zr, zr_dot individually and NO
future road.  What it sees is a sensor-style set of relative quantities.  The
road velocity is a hidden disturbance, so the problem is partially observed
(not Markov in the observation alone).  The previous action is included so that
the acceleration reading and the rate penalty are consistent; stacking recent
observations (e.g. VecFrameStack) may help a policy infer the road.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .config import EnvConfig
from .quarter_car import QuarterCar
from .road_profiles import RoadSample, sample_road

OBS_NAMES = (
    "sprung_acceleration",
    "suspension_deflection",
    "suspension_velocity",
    "tyre_deflection",
    "unsprung_velocity",
    "previous_action",
)


class ActiveSuspensionEnv(gym.Env):
    """Quarter-car active suspension with a normalised force action."""

    metadata = {"render_modes": []}

    def __init__(self, config: Optional[EnvConfig] = None, road: Any = None,
                 speed: Any = None, render_mode: Optional[str] = None):
        super().__init__()
        cfg = config if config is not None else EnvConfig()
        if road is not None:
            cfg = cfg.with_(road=road)
        if speed is not None:
            cfg = cfg.with_(speed=speed)
        if render_mode is not None:
            raise ValueError("ActiveSuspensionEnv does not support rendering")
        self.config = cfg
        self.car = QuarterCar(cfg.vehicle)
        self._h = cfg.timing.dt_sim
        self._n_sub = cfg.timing.substeps
        self._max_steps = cfg.timing.max_steps
        self._u_max = cfg.actuator.u_max
        self._f_static = cfg.vehicle.static_tyre_load
        self._f_ref = cfg.reward.f_ref_frac * self._f_static

        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(len(OBS_NAMES),), dtype=np.float32)

        self._road: Optional[RoadSample] = None
        self._zr: list = []
        self._zrd: list = []
        self._x: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
        self._k = 0
        self._a_prev = 0.0
        self._done = True

    # ------------------------------------------------------------------ helpers
    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def road_sample(self) -> Optional[RoadSample]:
        """Road, speed and seed used by the current episode (None before first reset)."""
        return self._road

    def _road_at(self, i: int) -> Tuple[float, float]:
        return self._zr[i], self._zrd[i]

    def _physical(self, x, zr: float, zrd: float, u: float) -> Dict[str, float]:
        acc = self.car.sprung_acceleration(x, zr, zrd, u)
        ft = self.car.tyre_force_dynamic(x, zr, zrd)
        d, dv = x[0] - x[2], x[1] - x[3]
        return {
            "sprung_acceleration": acc,
            "suspension_deflection": d,
            "suspension_velocity": dv,
            "tyre_deflection": x[2] - zr,
            "unsprung_velocity": x[3],
            "tyre_force_dynamic": ft,
            "tyre_load_ratio": 1.0 + ft / self._f_static,
        }

    def _obs(self, phys: Dict[str, float], a_prev: float) -> np.ndarray:
        s = self.config.obs_scales
        return np.array(
            [
                phys["sprung_acceleration"] / s.acc,
                phys["suspension_deflection"] / s.defl,
                phys["suspension_velocity"] / s.defl_rate,
                phys["tyre_deflection"] / s.tyre_defl,
                phys["unsprung_velocity"] / s.unsprung_vel,
                a_prev,
            ],
            dtype=np.float32,
        )

    # -------------------------------------------------------------------- reset
    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        """Start a new episode.

        options (all optional):
          "road"      : road specification (see road_profiles) overriding the config
          "speed"     : float or (lo, hi) overriding the config speed
          "road_seed" : int, seed of a stochastic road (for held-out evaluation)
          "initial_state": 4-sequence [zs, zs_dot, zu, zu_dot] ADDED to the
                           road-following initial state (e.g. a given perturbation)
        The road, speed and initial state depend only on (seed, options, config),
        never on the policy, so different controllers can be compared under
        identical conditions by resetting with the same seed/options.
        """
        super().reset(seed=seed)
        options = options or {}
        cfg = self.config
        road_spec = options.get("road", cfg.road)
        speed_spec = options.get("speed", cfg.speed)
        self._road = sample_road(road_spec, self.np_random, speed_spec, options.get("road_seed"))

        n_pts = 2 * self._max_steps * self._n_sub + 3  # half-step grid
        zr, zrd = self._road.profile.time_series(self._road.speed, 0.5 * self._h, n_pts)
        self._zr, self._zrd = zr.tolist(), zrd.tolist()

        # Road-following initial state: vehicle already in steady motion on the road
        # (zs = zu = zr(0), velocities = road velocity) => no start-up transient.
        x0 = np.array([zr[0], zrd[0], zr[0], zrd[0]], dtype=np.float64)
        if cfg.initial_state_noise > 0:
            x0 = x0 + self.np_random.normal(0.0, cfg.initial_state_noise, size=4)
        if "initial_state" in options:
            extra = np.asarray(options["initial_state"], dtype=np.float64)
            if extra.shape != (4,):
                raise ValueError("options['initial_state'] must have 4 entries")
            x0 = x0 + extra
        self._x = tuple(float(v) for v in x0)
        self._k = 0
        self._a_prev = 0.0
        self._done = False
        self._counters = {"contact_loss_steps": 0, "travel_violation_steps": 0}

        phys = self._physical(self._x, zr[0], zrd[0], 0.0)
        info = {
            **phys,
            "time": 0.0,
            "road": self._road.label,
            "road_seed": self._road.seed,
            "speed": self._road.speed,
            "actuator_force": 0.0,
            "road_height": zr[0],
        }
        return self._obs(phys, 0.0), info

    # --------------------------------------------------------------------- step
    def step(self, action):
        if self._done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        a_arr = np.asarray(action, dtype=np.float64).reshape(-1)
        if a_arr.size != 1:
            raise ValueError(f"action must have exactly 1 element, got shape {np.shape(action)}")
        a_raw = float(a_arr[0])
        if not math.isfinite(a_raw):
            raise ValueError("action must be finite")
        a = min(1.0, max(-1.0, a_raw))          # saturation of the command
        saturated = a != a_raw
        u = a * self._u_max                      # [N], held constant over the control step

        cfg = self.config
        rw = cfg.reward
        x = self._x
        n_sub, h = self._n_sub, self._h
        base = 2 * self._k * n_sub
        sum_acc2 = sum_d2 = sum_ft2 = 0.0
        max_abs_d = 0.0
        min_load_ratio = float("inf")
        for j in range(n_sub):
            i0 = base + 2 * j
            r0, rm, r1 = self._road_at(i0), self._road_at(i0 + 1), self._road_at(i0 + 2)
            # Quantities at the start of the sub-step (rectangle rule over the interval)
            acc = self.car.sprung_acceleration(x, r0[0], r0[1], u)
            ft = self.car.tyre_force_dynamic(x, r0[0], r0[1])
            d = x[0] - x[2]
            sum_acc2 += acc * acc
            sum_d2 += d * d
            sum_ft2 += ft * ft
            max_abs_d = max(max_abs_d, abs(d))
            min_load_ratio = min(min_load_ratio, 1.0 + ft / self._f_static)
            x = self.car.rk4_step(x, u, h, r0, rm, r1)
            if not all(math.isfinite(v) for v in x):
                break
        self._x = x
        self._k += 1

        # ---- end-of-step measurements
        i_end = 2 * self._k * n_sub
        zr_e, zrd_e = self._road_at(i_end)
        finite = all(math.isfinite(v) for v in x)
        phys = self._physical(x, zr_e, zrd_e, u) if finite else {
            k: float("nan") for k in ("sprung_acceleration", "suspension_deflection",
                                       "suspension_velocity", "tyre_deflection",
                                       "unsprung_velocity", "tyre_force_dynamic", "tyre_load_ratio")}

        # ---- termination / truncation
        sf = cfg.safety
        reason = None
        if not finite:
            reason = "non_finite_state"
        elif abs(phys["suspension_deflection"]) > sf.hard_suspension_limit:
            reason = "suspension_hard_limit"
        elif abs(phys["tyre_deflection"]) > sf.hard_tyre_limit:
            reason = "tyre_hard_limit"
        contact_loss_interval = min_load_ratio <= 0.0
        travel_violation_interval = max_abs_d > sf.travel_limit
        if reason is None and sf.contact_loss_terminates and contact_loss_interval:
            reason = "contact_loss"
        if reason is None and sf.travel_limit_terminates and travel_violation_interval:
            reason = "suspension_travel_limit"
        terminated = reason is not None
        truncated = (not terminated) and self._k >= self._max_steps

        # ---- reward (per-step, dimensionless, independent of dt_ctrl)
        terms = {
            "acceleration": rw.w_acc * (sum_acc2 / n_sub) / rw.a_ref**2,
            "deflection": rw.w_defl * (sum_d2 / n_sub) / rw.d_ref**2,
            "tyre_force": rw.w_tyre * (sum_ft2 / n_sub) / self._f_ref**2,
            "effort": rw.w_effort * a * a,
            "rate": rw.w_rate * ((a - self._a_prev) / 2.0) ** 2,
        }
        reward = -sum(terms.values()) if finite else 0.0
        if terminated:
            reward -= rw.termination_cost_per_step * (self._max_steps - self._k)
        if not math.isfinite(reward):
            reward = -rw.termination_cost_per_step * (self._max_steps - self._k)

        if contact_loss_interval:
            self._counters["contact_loss_steps"] += 1
        if travel_violation_interval:
            self._counters["travel_violation_steps"] += 1

        a_prev_for_obs = a
        self._a_prev = a
        self._done = terminated or truncated
        obs = self._obs(phys, a_prev_for_obs)
        if not np.all(np.isfinite(obs)):
            obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

        info = {
            **phys,
            "time": self._k * cfg.timing.dt_ctrl,
            "step": self._k,
            "road_height": zr_e,
            "road_velocity": zrd_e,
            "actuator_force": u,
            "action_applied": a,
            "action_saturated": saturated,
            "actuator_power": u * phys["suspension_velocity"] if finite else float("nan"),
            "contact_loss": bool(finite and phys["tyre_load_ratio"] <= 0.0),
            "contact_loss_in_interval": contact_loss_interval,
            "min_tyre_load_ratio_interval": min_load_ratio,
            "max_abs_deflection_interval": max_abs_d,
            "travel_limit_exceeded_in_interval": travel_violation_interval,
            "reward_terms": terms,
            "termination_reason": reason,
            "contact_loss_steps": self._counters["contact_loss_steps"],
            "travel_violation_steps": self._counters["travel_violation_steps"],
            "state": np.array(x, dtype=np.float64),
        }
        return obs, float(reward), terminated, truncated, info
