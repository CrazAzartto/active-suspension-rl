# RL-Based Active Suspension Control Using a Quarter-Car Simulation

AI Course Mini Project (Assignment 2, IIT Jodhpur). A PPO agent learns to command an actuator force in a quarter-car
(2-DOF) active suspension on varying roads, and is compared with the passive suspension (zero actuator force).

* Team: **Darshan Prajapati** (B23CH1032, b23ch1032@iitj.ac.in), **Sunny Kumar** (B23ME1073, b23me1073@iitj.ac.in)
* GitHub: https://github.com/CrazAzartto/active-suspension-rl
* Report: `report/main.tex` -> `report/main.pdf` (2 pages, two columns)

## What is in the repository
```
environment/   quarter-car dynamics (RK4), road profiles, Gymnasium env "ActiveSuspension-v0", evaluation helpers
training/      shared helpers, physical metrics, validation callback
train.py  evaluate.py  plot_results.py  smoke_test.py        training / evaluation / figures / API smoke test
configs/       ppo_default.json (300k-step run), ppo_smoke.json (seconds), eval_scenarios.json (8 evaluation scenarios)
scripts/       passive run, controller comparison, passive-damping sweep, reward-term breakdown
tests/         physics + environment tests, training-pipeline tests, report-consistency tests (72 tests)
models/ppo_default/   trained PPO agent (best_model.zip = best validation checkpoint at 275k steps; final_model.zip; checkpoints/)
models/ppo_seed1/, models/ppo_seed2/   same config retrained with seeds 1 and 2 (robustness check, report Table II)
models/ppo_1M/        second run, 1M steps (no better validation return; kept for transparency)
logs/  results/ppo_default/  figures/ppo_default/  report/    training logs, CSV results, figures, LaTeX report
INTERFACE_CONTRACT.md   exact environment API (observation/action/reward definitions)
README_stage1.md  README_stage2.md  STAGE2_HANDOFF.md  docs/FINAL_AUDIT.md   development notes and audit
```
Interface (details in `INTERFACE_CONTRACT.md`): observation `Box(6,)` = [z̈s/2, (zs−zu)/0.05, (żs−żu)/0.5, (zu−zr)/0.01, żu/0.5,
previous action]; action `Box(1,)` in [−1,1] -> force ±1500 N; episode 1000 steps (10 s); reward = −weighted sum of normalised squared
acceleration, deflection, dynamic tyre force, effort and force-rate (formula in the contract §6 and the report).

## Install (run from the repository root)
Tested with Python 3.12 / 3.13, numpy 2.4–2.5, gymnasium 1.4, stable-baselines3 2.9, torch 2.14, pandas 3.x, matplotlib 3.10–3.11.

**Windows (PowerShell)**
```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
python -m pip install --upgrade pip
pip install -r requirements.txt
```
**Linux / macOS**
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```
On Linux, `torch` from PyPI is a large CUDA download; on a CPU-only machine install a CPU wheel of torch first (see pytorch.org).
LaTeX (pdflatex + latexmk, e.g. MiKTeX or TeX Live) is needed only to rebuild the report.

## Commands (same on every OS once the environment is active)
```bash
python -m pytest -q                                   # 72 tests, ~10-60 s
python smoke_test.py                                  # env/API smoke test (seconds)
# --- small smoke run of the whole pipeline (under a minute) ---
python train.py --config configs/ppo_smoke.json
python evaluate.py --model models/smoke/final_model.zip --run smoke --n-seeds 2 --no-skyhook
python plot_results.py --run smoke
# --- full experiment (training ~1.5-2.5 min on one CPU core; evaluation ~1-4 min) ---
python train.py --config configs/ppo_default.json                 # seed 0, 300k steps -> models/ppo_default, logs/ppo_default
python evaluate.py --model models/ppo_default/best_model.zip --run ppo_default
python scripts/reward_term_breakdown.py --model models/ppo_default/best_model.zip --run ppo_default
python plot_results.py --run ppo_default --tmax 6                 # figures/ppo_default/*.png|pdf + captions.md
# --- seed robustness (each ~1.5 min train + ~1 min eval) ---
python train.py --config configs/ppo_default.json --seed 1 --run-name ppo_seed1
python train.py --config configs/ppo_default.json --seed 2 --run-name ppo_seed2
python evaluate.py --model models/ppo_seed1/best_model.zip --run ppo_seed1 --no-skyhook
python evaluate.py --model models/ppo_seed2/best_model.zip --run ppo_seed2 --no-skyhook
python scripts/seed_summary.py --runs ppo_default ppo_seed1 ppo_seed2   # results/seed_summary.csv, report/seed_table.tex
# --- report ---
python report/make_report_assets.py --run ppo_default            # numbers.tex, results_table.tex, report figures from CSVs
cd report && latexmk -pdf main.tex                                # -> report/main.pdf
# optional: python train.py --config configs/ppo_default.json --run-name ppo_1M --total-timesteps 1000000
# optional: python scripts/passive_damping_sweep.py ; python scripts/compare_controllers.py --road rough_C
```
Note: re-running `train.py` with an existing run name overwrites that run's model and logs. Use `--run-name` for new experiments.

Reload and use the trained agent:
```python
from stable_baselines3 import PPO; from environment import ActiveSuspensionEnv
m = PPO.load("models/ppo_default/best_model.zip", device="cpu"); env = ActiveSuspensionEnv(road="rough_C")
obs, _ = env.reset(seed=100000); action, _ = m.predict(obs, deterministic=True)   # observations are passed raw
```

## Reproducibility
Training seeds must be < 100000; seeds >= 100000 are reserved for held-out evaluation.
Checked on 26 Oct 2026 in a fresh environment (Python 3.13.17, SB3 2.9.2, torch 2.14.1, CPU):
* `pytest -q`: 72 passed.
* Re-running `evaluate.py` on `models/ppo_default/best_model.zip` reproduced all four result CSVs bit-for-bit.
* Re-training with `configs/ppo_default.json` (seed 0) reproduced `logs/ppo_default/validation_log.csv` exactly.
* Two extra training seeds (`models/ppo_seed1`, `models/ppo_seed2`) reached best validation returns of −594 and −602
  (seed 0: −612; passive: −829) with the same comfort-vs-tyre-load trade-off (`results/seed_summary.csv`).

## Measured results (run `ppo_default`; details: `results/ppo_default/*.csv`, report Table I)
PPO lowers RMS/peak body acceleration in every evaluated scenario (22-26 % on random roads, 22-43 % on held-out roads,
49 % on the in-distribution bump) but **worsens RMS dynamic tyre load by 22-25 % on random roads** and suspension deflection in most
scenarios; the contact-loss flag rate rises on ISO class D (1.4 % -> 4.3 % of steps). Across three training seeds, random-road RMS
acceleration falls by 22-26 % and RMS tyre load rises by 20-29 % (report Table II).
This is a comfort/road-holding trade-off, not a uniform improvement.

## Limitations
Linear model without bump stops, tyre lift-off or actuator lag; vehicle parameters are plausible, not identified from a real car;
unweighted RMS acceleration (no ISO 2631 filter); partially observed (road velocity hidden, no preview); reward weights untuned;
validation return dips well below passive at 50-100k steps before recovering (seen in all three seeds tried).
See `docs/FINAL_AUDIT.md`.

## Originality and tooling
The simulator, environment, training and evaluation code were written for this project (no external RL repository forked or copied);
libraries used: NumPy, pandas, Matplotlib, Gymnasium, Stable-Baselines3, PyTorch. The code and report were produced with AI assistance
(Claude); the team should state this according to the course policy.
