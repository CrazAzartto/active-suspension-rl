# Stage 1 — Quarter-car active-suspension environment

Project: **RL-Based Active Suspension Control Using a Quarter-Car Simulation**.
This stage delivers the simulator, a Gymnasium environment, tests and the Stage 2 interface
(`INTERFACE_CONTRACT.md`). It does **not** contain training code or the report.
All code here was written from scratch for this assignment (no external RL repository was used or copied).
Note: the assignment requires a top-level `README.md` in the final repository; this file is the Stage 1 part of it.

## File tree

```
active_suspension_rl/
├── environment/
│   ├── __init__.py           exports + gym registration "ActiveSuspension-v0"
│   ├── config.py             ALL parameters (vehicle, actuator, timing, reward, safety, scales)
│   ├── quarter_car.py        2-DOF dynamics, RK4 step, linear matrices, energy
│   ├── road_profiles.py      flat / bump / ISO-8608-style random roads, train/test mixtures
│   ├── suspension_env.py     ActiveSuspensionEnv (Gymnasium)
│   └── evaluation.py         passive + skyhook(reference) controllers, rollout, physical metrics
├── scripts/
│   ├── run_passive.py            passive simulation (+ optional plot)
│   ├── compare_controllers.py    passive vs reference under identical conditions
│   └── passive_damping_sweep.py  checks the passive baseline is not artificially weak
├── tests/  test_dynamics.py  test_environment.py  test_sb3_smoke.py
├── INTERFACE_CONTRACT.md   README_stage1.md
├── requirements.txt  requirements-stage2.txt  pytest.ini
```

## Commands

```bash
pip install -r requirements.txt
python -m pytest -q
python scripts/run_passive.py --road rough_C --speed 15 --seed 3 --plot passive_rough.png
python scripts/compare_controllers.py --road rough_C --speed 15
python scripts/passive_damping_sweep.py
```

## Physical model and sign convention

Two-DOF quarter car, z axes **up**, all displacements relative to **static equilibrium**
(gravity is balanced by the static spring/tyre deflections, so `ms·g` and `mu·g` do not appear; the static tyre
load `(ms+mu)g` is used only to interpret the dynamic tyre force):

```
ms z̈s = −ks(zs−zu) − cs(żs−żu) + u
mu z̈u =  ks(zs−zu) + cs(żs−żu) − kt(zu−zr) − ct(żu−żr) − u
```

`u > 0` pushes the sprung mass up and the unsprung mass down (equal and opposite). Tyre damping `ct` is an option,
default **0** (the equations in the assignment). If `ct > 0` the same equation is used by the simulator, reward, metrics
and tests (the road velocity żr is then needed and is provided analytically). Nothing else is modelled:
no actuator dynamics, no bump stops, no nonlinear spring/damper, no tyre lift-off (see limitations).

**Parameters** (defaults, SI): ms 290 kg, mu 59 kg, ks 16 kN/m, cs 1 kN·s/m, kt 190 kN/m, ct 0.
They are typical round-number values of the size used for a mid-size car corner in the quarter-car literature —
**not** taken from a specific paper or datasheet, and not validated against a real vehicle. They give a body mode of
1.18 Hz, wheel-hop 9.40 Hz and body damping ratio ≈ 0.23 (printed by the `VehicleParams` properties).
Actuator limit 1 500 N (~0.5 g on the quarter body) is a design choice.

## Numerics

| item | value |
|---|---|
| integrator | fixed-step classical RK4, road sampled at t, t+h/2, t+h; actuator force held constant (ZOH) |
| integration step | 1 ms (stability limit of RK4 here ≫ 1 ms: fastest mode ≈ 59 rad/s) |
| control step | 10 ms (100 Hz) = exactly 10 sub-steps (enforced: non-integer ratios raise `ValueError`) |
| episode | 10 s = 1000 control steps; truncated at the end |
| initial condition | road-following equilibrium (zs = zu = zr(0), velocities = żr(0)) — no start-up transient; optional noise/offset via `initial_state_noise` / `options["initial_state"]` |
| saturation | command clipped to [−1, 1] → ±`u_max`, reported in `info` (it is a physical actuator limit) |
| safeguards | termination on non-finite state or on hard limits (0.30 m suspension, 0.15 m tyre) with a large per-remaining-step penalty; no clipping of states |

Tests show: RK4 matches the matrix-exponential solution to 1e-9, shows ≈4th-order convergence, conserves energy in the
undamped case, and stays bounded for 60 s on a class-E road with worst-case bang-bang force.

## Roads

Roads are analytic C¹ functions of distance, converted to time with a constant speed. `BumpRoad` is a raised-cosine hump;
`RandomRoad` is a seeded sum of 100 cosines (random phases and in-bin frequencies, 0.02–4 cycle/m) matching an ISO-8608-style PSD
`Gd(n) = Gd(n0)(n/n0)^−2` with the class centre values A…E (16·10⁻⁶ … 4096·10⁻⁶ m³ at n0 = 0.1 cycle/m).
The road height is shifted so z(0) = 0. See the contract for presets (`"train"` vs held-out `"test"`).
The ISO class values are the commonly used class-centre values as I recall them; I did not look them up in the standard, so
verify before citing them in the report. Roads are generated at `reset()` (~75 ms for stochastic roads).

## Reward design (short)

Negative weighted mean of squared **normalised** quantities: sprung acceleration (comfort), suspension deflection
(travel), dynamic tyre force (road holding), actuator effort and force-rate. Reference scales equal the passive
car's RMS response on a class-C road at 15 m/s (1.15 m/s², 0.012 m, 0.25·static load, rounded), so each term is O(1)
for a passive car on a rough road; weights express priority (comfort 1.0, deflection 0.3, tyre 0.3, effort/rate 0.05).
Exact formula, units and time-step treatment are in `INTERFACE_CONTRACT.md §6`. These weights are a first choice, **not**
tuned; Stage 2 may change them via `RewardParams` but must report that it did. Evaluation uses separate physical metrics.

## Passive baseline

`PassiveController` (force always 0, identical to the zero-action policy). `scripts/passive_damping_sweep.py` varies `cs` and
shows the default `cs = 1000` is close to the best passive setting for the reward on class B/C roads
(sweep over 500–4000 N·s/m, 10 seeds, 15 m/s: lowest RMS acceleration at cs = 750, highest reward return at cs = 1000 on both classes).
If a different `cs` is preferred, use it for passive **and** learned controllers.

## Sanity reference (not the RL agent)

A hand-written skyhook controller (`u = −1500·żs`, absolute sprung velocity reconstructed from the observation) is provided
only to show that the action channel can improve on passive and to give tests something non-trivial. On ISO class C,
15 m/s, seeds 100000–100009 (`scripts/compare_controllers.py`):

| controller | RMS acc [m/s²] | RMS defl [m] | RMS dyn. tyre load / static | return |
|---|---|---|---|---|
| passive | 1.153 | 0.0124 | 0.256 | −1863 |
| skyhook reference | 1.053 | 0.0092 | 0.256 | −1488 |

So there is measurable but modest headroom (≈9 % RMS acceleration with this simple controller); I have not computed an
optimal-control bound, so how much a trained PPO agent can gain is **unknown**. No RL results exist yet.

## Tests

`python -m pytest -q`: dynamics (force signs, equal/opposite actuator force, equilibrium, analytic steady states, energy,
matrix exponential, convergence, bump response, road smoothness/reproducibility, stability) and environment (spaces, shapes,
`gymnasium.utils.env_checker.check_env` on 5 road settings, termination/truncation, saturation, reproducibility, finite values,
reward terms, guards, passive behaviour, identical conditions across controllers, observation/info consistency).
`tests/test_sb3_smoke.py` runs SB3's `check_env` and a tiny PPO run if SB3 is installed (otherwise skipped).
Last run (Python 3.12.3, numpy 2.4.4, gymnasium 1.4.0, pytest 9.1.1, stable-baselines3 2.9.0, torch 2.14.1): **61 passed in ~12.6 s**. Earlier run without SB3 installed: 59 passed, SB3 file skipped.

## Assumptions and unresolved issues

1. Linear, bump-stop-free suspension: very large deflections (> 0.1 m) are possible and are only counted/penalised.
2. Linear tyre cannot lift off; contact loss is a flag (`tyre_load_ratio ≤ 0`), after which the model is outside its validity.
   Roads where this happens often (class D/E at speed, tall short bumps) should be treated as stress tests.
3. Ideal actuator: no lag, bandwidth, power or energy limit; the effort term is a simple quadratic penalty, not an energy model.
4. Partially observed: road velocity is hidden; no preview of the road is exposed (by design).
5. Reward weights/scales are untuned first choices; reward magnitude grows quickly on rough roads (consider reward normalisation).
6. RMS acceleration is unweighted (no ISO 2631 frequency weighting); do not call it an ISO comfort index in the report.
7. Observation space is unbounded (`±inf`) on purpose; use `VecNormalize` or keep the fixed scales.
8. Vehicle parameters are plausible, not identified from a real car; no experimental validation exists.
9. `reset()` for stochastic roads costs ~75 ms; if that dominates Stage 2 throughput, reduce `n_components` in `RandomRoad` or reuse roads.
10. Only a single vehicle speed per episode (constant speed).
