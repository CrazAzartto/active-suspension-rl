**fig1_training_curve** — Fig. 1: PPO learning curves. Left: raw return of training episodes (grey) with rolling mean. Right: periodic deterministic validation on fixed seeds from the training road mixture, against the passive suspension on the same seeds. Higher (less negative) return is better. Validation seeds are disjoint from the held-out evaluation seeds.

**fig2_acceleration_time** — Fig. 2: Sprung-mass vertical acceleration over time, passive vs PPO (and the hand-designed skyhook reference, which is NOT an RL result), same road, speed, seed and initial state.

**fig3_deflection_time** — Fig. 3: Suspension deflection zs - zu over time (matched conditions). Smaller excursions mean less rattle-space use.

**fig4_tyre_force_time** — Fig. 4: Dynamic tyre force Ft = kt (zu - zr) over time (matched scenario and seed). Smaller excursions indicate better road holding; the static load is 3.42 kN, so Ft <= -3.42 kN would be a contact-loss flag.

**fig6_actuator_force_time** — Fig. 6: Actuator force over time on the same scenarios/seeds as Figs. 2-4. The actuator limit is ±1500 N (the passive suspension applies 0 N and is therefore omitted).

**fig5_summary_metrics** — Fig. 5: Summary of physical metrics. Top: percentage improvement of PPO over the passive suspension (positive = PPO better; lower-is-better metrics are sign-flipped; undefined where passive ~ 0). Metrics trade off, so no single bar establishes superiority; negative bars are degradations. Bottom: absolute RMS acceleration with standard deviation across seeds.

