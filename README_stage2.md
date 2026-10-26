# Stage 2 — PPO training and evaluation

Adds a training/evaluation layer on top of the **unmodified** Stage 1 environment (`environment/`, see `INTERFACE_CONTRACT.md`).
No Stage 1 file was changed.

## Install and run (from the repository root)
```bash
pip install -r requirements-stage2.txt          # numpy, gymnasium, pytest, matplotlib, stable-baselines3 (+torch), pandas
python smoke_test.py                            # env/API smoke test
python -m pytest -q                             # 61 Stage 1 + 8 Stage 2 tests
python train.py --config configs/ppo_smoke.json # ~2 s integration run (4k steps)
python train.py --config configs/ppo_default.json   # full run: 300k steps, seed 0 (~2.5 min on 1 CPU core)
python evaluate.py --model models/ppo_default/best_model.zip --run ppo_default
python plot_results.py --run ppo_default --tmax 6
```
Reload and run inference: `PPO.load("models/ppo_default/best_model.zip", device="cpu").predict(obs, deterministic=True)`
(observations are passed raw; only the reward is normalised during training).

## Why PPO
Bounded continuous 1-D force command, dense reward, cheap simulator (~2000 PPO steps/s on one core), stable on-policy updates with clipped
objective; Gaussian policy actions are clipped to [-1,1] by the environment. (PPO is a standard choice here, not claimed to be optimal.)

## Settings (configs/ppo_default.json)
| item | value |
|---|---|
| total timesteps | 300 000 (4 envs x 1000-step rollouts, i.e. 75 updates) |
| policy / value net | MLP 64-64, tanh, initial log-std -0.5 |
| learning rate / batch / epochs | 3e-4 / 500 / 10 |
| gamma / GAE lambda / clip / entropy | 0.99 / 0.95 / 0.2 / 0.0 |
| seed | 0 (env i first reset with seed+i; training seeds must be < 100000) |
| validation | every 25k steps, 6 fixed seeds 90000-90005 on the `train` road mixture, vs passive on the same seeds; best model by validation return |
| checkpoints | every 50k steps (`models/<run>/checkpoints/`), plus `best_model.zip`, `final_model.zip` |
| reward handling | `VecNormalize(norm_obs=False, norm_reward=True)`; Monitor/validation/evaluation report RAW reward |
Environment parameters are the Stage 1 defaults (vehicle, reward weights, actuator limit); nothing was tuned or changed. Passive baseline uses the same vehicle.
Not used: frame stacking, road preview, any observation/action wrapper.

## Evaluation protocol (configs/eval_scenarios.json)
Seeds 100000+ (never used for training or model selection), 10 s episodes, speeds as listed, passive and PPO (deterministic) share environment,
road, speed, reset seed and initial state (road height is asserted bit-identical). `flat_perturbed` starts with the body +20 mm above equilibrium.
Deterministic scenarios (bump, flat_perturbed) use 1 seed; random-road scenarios use 20-30 seeds. A hand-designed skyhook controller is also
evaluated as a *reference only* (not RL, not tuned).

## Measured results (PPO, run `ppo_default`, best validation checkpoint = 275k steps)
Percent change of PPO vs passive, computed from means over seeds. **Positive = PPO better** (sign-flipped for lower-is-better metrics).
| scenario | n | RMS acc | peak acc | RMS defl | max defl | RMS tyre load | return |
|---|---|---|---|---|---|---|---|
| smooth_A_15 | 20 | +24.0 | +24.1 | -36.3 | -35.3 | -22.2 | +19.4 |
| flat_perturbed | 1 | +73.6 | +70.2 | -46.7 | -1.1 | +72.3 | +23.6 |
| bump_10 | 1 | +49.1 | +37.0 | +3.3 | -10.7 | +2.6 | +61.8 |
| rough_B_15 | 20 | +24.0 | +23.9 | -18.2 | -27.1 | -22.8 | +24.2 |
| rough_C_15 | 20 | +23.6 | +21.8 | -12.3 | -21.5 | -23.6 | +25.2 |
| heldout_bump_tall | 1 | +43.0 | +33.7 | +1.2 | -11.2 | -5.9 | +56.3 |
| heldout_rough_D_10 | 20 | +22.2 | +21.0 | -8.6 | -13.4 | -24.6 | +24.4 |
| heldout_test_mixture | 30 | +26.2 | +25.7 | -8.9 | -14.5 | -22.1 | +27.5 |

Reading: PPO gives a clear comfort gain (RMS/peak acceleration lower in every scenario, on every paired seed) but it **pays for it**:
RMS dynamic tyre load is 22-25 % *worse* on rough roads and suspension deflection is worse on most scenarios. Contact-loss flag rate
(tyre-load ratio <= 0) on class D at 10 m/s rises from 1.4 % to 4.3 % of steps and on the held-out mixture from 0.5 % to 1.4 %.
Actuator saturation was 0 % everywhere (mean peak force <= ~890 N of 1500 N). No episode terminated early.
The `return` improves everywhere because the reward weights comfort (w_acc = 1) above deflection/tyre (0.3 each); a different weighting would
trade differently. This is therefore a comfort-vs-road-holding trade-off, not a uniform improvement.
Raw numbers: `results/ppo_default/{episodes,summary,comparison}.csv`. Figures: `figures/ppo_default/` (captions in `captions.md`).

## Limitations
* One training seed only; no confidence over training randomness. Evaluation seeds give spread over roads only (paired CI columns in comparison.csv).
* A 1M-step run (`ppo_1M`, same seed/config) did not improve validation return: its best checkpoint is the same 275k-step model and later checkpoints drifted slightly worse (final val return -659.5 vs -612.2). Not evaluated further.
* Validation return fluctuates (e.g. -3666 at 50k steps) so checkpoint selection matters; the best-by-validation model was used.
* Unweighted RMS acceleration (no ISO 2631 filter). Contact loss is a flag in a linear tyre model, not simulated lift-off.
* Plausible but unvalidated vehicle parameters; ideal actuator; partially observed (road velocity hidden). Skyhook is untuned.
* Only one reward weighting was tried; weights were not tuned.
