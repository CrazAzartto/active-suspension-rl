# INTERFACE CONTRACT — Stage 1 → Stage 2

Version 1.0. Do **not** rename the class or change the observation/action layout without
editing this file and bumping the version.

## 1. Import path and class

```python
from environment import ActiveSuspensionEnv, EnvConfig          # class + config
import gymnasium as gym; import environment                      # registers the id
env = gym.make("ActiveSuspension-v0")                            # same class, default config
```

Class: `environment.suspension_env.ActiveSuspensionEnv` (re-exported as `environment.ActiveSuspensionEnv`).
Run everything from the repository root (`active_suspension_rl/`).

## 2. Constructor

```python
ActiveSuspensionEnv(config: EnvConfig | None = None, road=None, speed=None, render_mode=None)
```

* `config=None` → `EnvConfig()` (all defaults below).
* `road` / `speed`, if given, override `config.road` / `config.speed`. (`render_mode` must be `None`; no rendering.)
* Every dataclass is frozen; use `cfg.with_(...)` or `dataclasses.replace` to derive variants, e.g.
  `EnvConfig(vehicle=VehicleParams(cs=1500.0), road="test")`.

### Default parameters (`environment/config.py`)

| Group | Parameter | Default | Unit |
|---|---|---|---|
| `VehicleParams` | `ms` sprung mass | 290 | kg |
| | `mu` unsprung mass | 59 | kg |
| | `ks` spring stiffness | 16 000 | N/m |
| | `cs` passive damping | 1 000 | N·s/m |
| | `kt` tyre stiffness | 190 000 | N/m |
| | `ct` tyre damping | 0 (off) | N·s/m |
| `ActuatorParams` | `u_max` | 1 500 | N |
| `TimingParams` | `dt_sim` (RK4 step) | 0.001 | s |
| | `dt_ctrl` (control step) | 0.01 (= 10 sim steps, exact) | s |
| | `episode_duration` | 10.0 → **1000 control steps** | s |
| `RewardParams` | `a_ref`, `d_ref`, `f_ref_frac` | 1.0 m/s², 0.015 m, 0.25 | |
| | `w_acc, w_defl, w_tyre, w_effort, w_rate` | 1.0, 0.3, 0.3, 0.05, 0.05 | – |
| | `termination_cost_per_step` | 10.0 | – |
| `SafetyParams` | `hard_suspension_limit` / `hard_tyre_limit` | 0.30 m / 0.15 m | |
| | `travel_limit` (counted, not terminal) | 0.10 | m |
| | `travel_limit_terminates`, `contact_loss_terminates` | False, False | |
| `ObservationScales` | `acc, defl, defl_rate, tyre_defl, unsprung_vel` | 2.0, 0.05, 0.5, 0.01, 0.5 | m/s², m, m/s, m, m/s |
| `EnvConfig` | `road`, `speed`, `initial_state_noise` | `"train"`, 15.0 m/s, 0.0 | |

Derived default properties: body bounce ≈ 1.18 Hz, wheel hop ≈ 9.40 Hz, body damping ratio ≈ 0.23,
static tyre load (ms+mu)·g = 3 424 N.

## 3. Action

`Box(low=-1, high=1, shape=(1,), dtype=float32)`; `a` is a normalised actuator-force command.

* Applied force **u = clip(a, −1, 1) · u_max** [N] (default ±1 500 N). Saturation is explicit and reported
  (`info["action_saturated"]`). Out-of-range finite values are clipped; NaN/inf or wrong size raise `ValueError`.
* **Sign:** u > 0 pushes the sprung mass **up** and the unsprung mass **down** (equal and opposite).
* The command is held constant (zero-order hold) for the whole control step (10 RK4 sub-steps).
* The actuator is an ideal force source (no lag/bandwidth/power limit). Scalars and lists are accepted by `step`.

## 4. Observation

`Box(-inf, inf, shape=(6,), dtype=float32)`. Values are `physical / scale` — **no clipping** is applied
(bounds are intentionally infinite; Gymnasium's checker warns about this, which is expected).

| idx | name (`OBS_NAMES`) | physical definition | unit | scale (default) |
|---|---|---|---|---|
| 0 | `sprung_acceleration` | z̈s at the current state with the force currently applied (previous action) | m/s² | 2.0 |
| 1 | `suspension_deflection` | zs − zu (>0 spring extended) | m | 0.05 |
| 2 | `suspension_velocity` | ż s − ż u | m/s | 0.5 |
| 3 | `tyre_deflection` | zu − zr (>0 tyre compressed beyond static) | m | 0.01 |
| 4 | `unsprung_velocity` | ż u (absolute vertical velocity) | m/s | 0.5 |
| 5 | `previous_action` | a(k−1) ∈ [−1, 1] (0 at reset) | – | 1 |

Notes / limitations (do not claim more than this):

* The physical state is [zs, żs, zu, żu] plus the road (zr, żr); the agent **does not** see zs, zu, zr, żr
  individually and **never** sees future road. Road velocity żr is a hidden disturbance, so the problem is
  **partially observed**, not Markov in the observation alone. Frame stacking (e.g. `VecFrameStack`) is
  allowed and may help; it is not required for the interface to work.
* Absolute sprung velocity is recoverable as obs[2]·0.5 + obs[4]·0.5 (used by the skyhook reference).
* Channel 0 is, in this ideal model, a function of channels 1, 2, 5 (z̈s = (−ks·d − cs·ḋ + u_prev)/ms); it is kept
  because a real accelerometer would provide it directly.
* At reset the car is road-following (zs = zu = zr(0), velocities = żr(0)) so obs[0,1,2,3,5] = 0 but obs[4] equals
  the road's initial vertical velocity (non-zero on rough roads).

## 5. Reset and step

```python
obs, info = env.reset(seed=None, options=None)
obs, reward, terminated, truncated, info = env.step(action)
```

**reset options** (all optional): `"road"` (road spec), `"speed"` (float or `(lo, hi)`), `"road_seed"` (int, stochastic
roads), `"initial_state"` (4-vector added to the road-following initial state, `[zs, żs, zu, żu]`).
`reset(seed=s)` seeds the env RNG; road choice, road parameters, speed, road seed and initial-state noise are
drawn **only** inside `reset`, never from policy actions → same `(seed, options, config)` ⇒ identical road/speed/initial
state for any controller. Same seed + same actions ⇒ bit-identical trajectory (tested).

**Termination semantics**
* `truncated=True` when `max_steps = 1000` is reached (time limit; value bootstrapping is appropriate).
* `terminated=True` only for numerical/physical-validity guards: `non_finite_state`, `suspension_hard_limit`
  (|zs−zu| > 0.30 m), `tyre_hard_limit` (|zu−zr| > 0.15 m); optionally `suspension_travel_limit` and `contact_loss`
  if enabled in `SafetyParams`. These are fail-safes: none was reached by the passive car on the shipped presets in the test run
  (one evaluation seed per preset), and an early termination adds `−termination_cost_per_step · remaining_steps` to the reward so ending
  an episode can never be profitable. `info["termination_reason"]` is `None` otherwise.
* `step` after the episode finished (or before `reset`) raises `RuntimeError`.

## 6. Reward

Per control step k, with action `a_k = clip(command)` and `a_{k−1}` the previous one, and overbar = mean over the
10 sub-step start instants of the control interval (so the scale does not depend on `dt_ctrl`):

```
r_k = − [ w_acc·  mean(z̈s²)/a_ref²
        + w_defl· mean((zs−zu)²)/d_ref²
        + w_tyre· mean(Ft²)/F_ref²
        + w_effort· a_k²
        + w_rate·  ((a_k − a_{k−1})/2)² ]

Ft = kt·(zu − zr) + ct·(żu − żr)   (dynamic tyre force, N),  F_ref = f_ref_frac·(ms+mu)·g = 856 N
```

Defaults: `a_ref=1 m/s²`, `d_ref=0.015 m`, `f_ref_frac=0.25`, `w = (1, 0.3, 0.3, 0.05, 0.05)`. Every term is
dimensionless. The reference scales are the order of magnitude of the passive car's RMS response on an ISO class-C road
at 15 m/s in this simulator (measured: 1.15 m/s², 0.0124 m, 0.256·static load) — a design choice, not a standard.
Reward ≤ 0 always; `info["reward_terms"]` holds each weighted term; `reward == −sum(terms)` (tested) unless the episode
terminated early (see above). Rewards are **not** normalised to [−1, 0]; typical per-step magnitude is O(1) on rough
roads and large on class D/E, so `VecNormalize(norm_reward=True)` is advisable.

**Tyre contact.** The tyre is a linear spring(-damper) that cannot separate from the road in the equations. Wheel-load
ratio `tyre_load_ratio = 1 + Ft/((ms+mu)g)`; **contact loss is flagged when this ratio ≤ 0** (`info["contact_loss"]`,
`info["contact_loss_in_interval"]`). Penalising Ft² only *discourages* load variation; it does not guarantee contact.
Once the ratio would be ≤ 0 the linear tyre model is no longer physical (it would pull the wheel to the road); events
are counted, not simulated.

Reward terms and evaluation metrics are separate: use `environment.evaluation` for physical metrics.

## 7. Road profiles and speed

Specification (`road=` in constructor/config or `options={"road": ...}`), resolved once per reset:

| spec | meaning |
|---|---|
| `"flat"` | z = 0 |
| `"bump"` | raised-cosine hump, h = 0.04 m, L = 2.5 m, starting at x = 10 m |
| `"rough"`, `"rough_A"` … `"rough_E"` | stochastic ISO-8608-style road (`"rough"` = class C); seed from env RNG or `options["road_seed"]` |
| `"train"` (default) | uniform mixture: flat, bump (h 0.02–0.04 m, L 1.5–3 m, 8–14 m/s), ISO B, ISO C (10–20 m/s) |
| `"test"` | held-out mixture: tall bump (h 0.05–0.06 m, L 3–4 m, 14–18 m/s), ISO A at 22–28 m/s, ISO D at 8–12 m/s |
| dict | `{"kind": "bump"/"random"/"flat", <params>, "speed": ..., "seed": ...}`; any 2-tuple value = uniform range |
| `RoadProfile` instance | e.g. `BumpRoad(height=0.06, length=2.0, start=8.0)`, `RandomRoad("D", seed=123)` |
| list/tuple of specs | one chosen uniformly per reset |

Speed: `config.speed` (default 15 m/s) or a spec's own `"speed"`; float or `(lo, hi)` range. Spatial road → time via
`zr(t) = z(v t)`, `żr(t) = v·z′(v t)`. All profiles are C¹ (no discontinuities).

**Held-out evaluation:** `"test"` shares no road class/bump size/speed range with `"train"`. For stochastic roads use
explicit evaluation seeds, e.g. `reset(seed=100000 + i)` (convention in `scripts/`, training seeds should stay below
100000). Fixed examples: `options={"road": "rough_D", "road_seed": 424242, "speed": 20.0}`.

## 8. Running a passive simulation

```bash
python scripts/run_passive.py --road rough_C --speed 15 --seed 3 [--plot out.png]
```
```python
from environment import ActiveSuspensionEnv
from environment.evaluation import PassiveController, rollout, episode_metrics
env = ActiveSuspensionEnv(road="bump", speed=10.0)
traj = rollout(env, PassiveController(), seed=0)               # dict of time series
print(episode_metrics(traj, env.config.vehicle.static_tyre_load))
```
Passive = `a = 0` at every step (actuator force exactly 0). `PassiveController().predict(obs)` returns zeros.

## 9. Physical measurements

Available in `info` after every `step` (and `reset`, except step-specific keys), SI units:
`sprung_acceleration`, `suspension_deflection`, `suspension_velocity`, `tyre_deflection`, `unsprung_velocity`,
`tyre_force_dynamic` [N], `tyre_load_ratio`, `actuator_force` [N], `action_applied`, `action_saturated`,
`actuator_power` [W] (= u·(żs−żu)), `contact_loss`, `contact_loss_in_interval`, `min_tyre_load_ratio_interval`,
`max_abs_deflection_interval`, `travel_limit_exceeded_in_interval`, `road_height`, `road_velocity`, `time`, `step`,
`speed`, `road`, `road_seed` (reset only), `reward_terms`, `termination_reason`, `contact_loss_steps`,
`travel_violation_steps` (running counts), `state` (np.array [zs, żs, zu, żu]).
End-of-step values describe the state at the end of the control step; `*_in_interval` flags look at all 10 sub-steps.

Evaluation helpers (`environment/evaluation.py`): `rollout`, `episode_metrics`, `evaluate`, `PassiveController`,
`SkyhookController` (reference only), `format_table`. Metrics: rms/peak sprung acceleration (unweighted, **no ISO 2631
frequency weighting**), rms/max suspension deflection, RMS dynamic tyre load / static load, contact-loss and
travel-violation fractions, actuator RMS force and mean power, and the reward return (kept separate).
A policy may be any object with `predict(obs, deterministic=True)` (e.g. an SB3 model) or a callable `obs -> action`.

## 10. Install and test commands

```bash
cd active_suspension_rl
python -m venv .venv && source .venv/bin/activate          # optional
pip install -r requirements.txt                             # numpy, gymnasium, pytest, matplotlib
python -m pytest -q                                         # 61 tests (59 + 2 SB3 smoke tests; the latter are skipped if SB3 is absent)
pip install -r requirements-stage2.txt                      # adds stable-baselines3 (+ PyTorch) for Stage 2
python -m pytest -q tests/test_sb3_smoke.py                 # SB3 check_env + short PPO run
```

Performance (this sandbox, CPU, one process): ≈ 8 000 env steps/s including resets of the rough-road synthesis;
`reset()` ≈ 75 ms for stochastic roads (it pre-computes the whole 10 s road), a 1000-step episode ≈ 0.12 s.
