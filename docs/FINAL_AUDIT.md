# Project audit and submission checklist

Status labels: **VERIFIED** = executed or inspected; **FAILED**; **NOT VERIFIED**.
Last audit: 26 Oct 2026, fresh environment (Python 3.13.17, numpy 2.5.4, gymnasium 1.4.0, SB3 2.9.2, torch 2.14.1, CPU).

## 1. Executed checks
| Check | Result |
|---|---|
| `python -m pytest -q` | VERIFIED: 72 passed |
| `python smoke_test.py`; smoke train -> evaluate -> plot chain | VERIFIED: all ran |
| Actuator sign / equal-and-opposite force | VERIFIED: u = +100 N gives z̈s = +0.345 m/s², z̈u = −1.695 m/s²; ms·z̈s + mu·z̈u = 0 |
| RK4 accuracy | VERIFIED: env trajectories vs DOP853 (rtol 1e-11) on bump and ISO-C roads, with and without force: max relative error ≈ 1e-7 |
| Coupled modes (eigenvalues of A) | 1.14 Hz (ζ = 0.21) and 9.34 Hz (ζ = 0.15); now quoted in the report |
| Passive baseline not weak | VERIFIED: damping sweep 500–4000 N·s/m; cs = 1000 gives the best passive return on ISO B and C at 15 m/s |
| Passive force = 0 | VERIFIED: rms/peak force 0 in all 113 passive evaluation episodes |
| Evaluation reproducibility | VERIFIED: `evaluate.py` on `models/ppo_default/best_model.zip` reproduced episodes/summary/comparison/timeseries CSVs with max difference 0.0 |
| Training reproducibility | VERIFIED: retraining `configs/ppo_default.json` (seed 0) reproduced `logs/ppo_default/validation_log.csv` exactly |
| Seed robustness | VERIFIED: seeds 1 and 2 (`models/ppo_seed1`, `models/ppo_seed2`): best validation return −594 / −602 vs seed 0 −612 (passive −829); same trade-off; report Table II |
| Metric sampling | 100 Hz metrics vs 1 kHz reward integration: RMS acceleration 1.0557 vs 1.0556 m/s² (negligible) |
| Windows text encoding | VERIFIED: generated caption text is cp1252-encodable |

## 2. Known issues and status
| Issue | Status |
|---|---|
| Report/README previously listed the course staff instead of the team | FIXED: team is Darshan Prajapati (B23CH1032) and Sunny Kumar (B23ME1073) |
| GitHub URL placeholder | FIXED: https://github.com/CrazAzartto/active-suspension-rl |
| Git history | Repository created 26 Oct 2026; earlier development is not in git and no history was back-dated |
| Results use a single training seed | FIXED: three seeds, report Table II and Fig. 1 |
| Validation return drops to −3.7k…−5.9k at 50–100k steps before recovering (all seeds) | Reported in Fig. 1 and Sec. VI; optional future change: lower `log_std_init` |
| Vehicle parameters (290/59/16000/1000/190000) resemble a widely used quarter-car benchmark set | OPEN: cite the source after checking it |
| ISO 8608 class values and references [1]–[5] | NOT VERIFIED against the original sources |
| AI-assistance statement | OPEN: team decision per course policy |
| Linear tyre (contact loss only flagged), ideal actuator, no ISO 2631 weighting | Documented limitations |
