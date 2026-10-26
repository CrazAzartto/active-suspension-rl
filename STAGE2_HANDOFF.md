# STAGE 2 HANDOFF

## 1. Files added (Stage 1 files unchanged)
`train.py`, `evaluate.py`, `plot_results.py`, `smoke_test.py`, `training/{__init__,common,metrics,callbacks}.py`,
`configs/{ppo_default,ppo_smoke,eval_scenarios}.json`, `tests/test_training_pipeline.py`, `README_stage2.md`, `STAGE2_HANDOFF.md`.
Modified: `requirements-stage2.txt` (added pandas). `evaluate.py` etc. sit at the repo root next to `environment/`.

## 2. Install
`pip install -r requirements-stage2.txt` (torch is a ~4 GB CUDA wheel from PyPI; needs ~7 GB free disk; install with `--no-cache-dir` if disk is tight).

## 3. Commands
See README_stage2.md ("Install and run").

## 4. Actual status
* Executed: `python smoke_test.py` (passed); `python -m pytest -q` -> **69 passed** (61 Stage 1 + 8 new); smoke training (4k steps) ran and
  its evaluate/plot path worked (smoke outputs deleted).
* Full training **completed**: `ppo_default`, seed 0, 300 000 steps, 139 s on 1 CPU core. Best validation checkpoint at 275k steps.
* Second run `ppo_1M` (1 000 000 steps, 489 s) completed; no better than `ppo_default` (same best checkpoint). Identical early validation
  values in both runs confirm same-seed reproducibility.
* Evaluation and plotting were run for `ppo_default` only (final test count was re-run after the last code edit; see section 7).

## 5. Where things are
Models: `models/ppo_default/{best_model,final_model}.zip`, `train_config.json`, `training_status.json`, checkpoints; `models/ppo_1M/`.
Logs: `logs/ppo_default/{validation_log.csv,progress.csv,monitor/}`, same for `ppo_1M`.
Results: `results/ppo_default/{episodes,summary,comparison,timeseries}.csv`, `eval_config.json`, `improvement_table.md`.
Figures: `figures/ppo_default/fig1..fig6 (.png/.pdf)`, `captions.md`.

## 6. Headline measured results
Table in README_stage2.md. Summary: PPO lowers RMS and peak sprung acceleration by ~22-26 % on random roads (all paired seeds), 43-49 % on bumps,
74 % on the flat-road free decay, including held-out roads; RMS tyre load worsens ~22-25 % on random roads and RMS deflection worsens on most scenarios;
contact-loss flag rate rises on class D. Do NOT describe this as uniform improvement. Skyhook is a hand-designed reference, not RL.

## 7. Bugs / limitations
* Fixed during development: `evaluate.py` relative-model-path bug (found by running it).
* Stage 1: no defects found. Not verified: ISO class roughness values (README_stage1 says recalled from memory) - check before citing.
* Single training seed; unweighted acceleration; contact loss is a flag only; reward weights untuned (see README_stage2 limitations).
* Figures were generated from CSV; only fig5 was visually inspected by me.

## 8. For Claude 3 (report, 2 pages, two-column, IIT Jodhpur assignment)
* Needs: GitHub link + team members at top; RL formulation (S: 6-D obs; A: 1-D continuous force in [-1,1]->+-1500 N; R: formula in INTERFACE_CONTRACT §6;
  environment/baseline: quarter-car RK4, passive baseline; algorithm: PPO) and graphs (use fig1, fig2, fig5; others if space permits).
* Quote only numbers from `results/ppo_default/*.csv`; state the comfort vs tyre-load/deflection trade-off and the single-seed limitation.
* Top-level `README.md` is still required by the assignment: combine README_stage1.md and README_stage2.md.
* Commit history must show development; repo is not yet a git repository.
