"""Consistency between saved results (CSV) and the numbers/tables used in report/. Skipped if results are absent."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results" / "ppo_default"
pytestmark = pytest.mark.skipif(not (RES / "episodes.csv").exists(), reason="results/ppo_default not present")


def test_passive_applies_zero_force_in_saved_results():
    ep = pd.read_csv(RES / "episodes.csv")
    p = ep[ep.controller == "passive"]
    assert (p.rms_force == 0).all() and (p.peak_force == 0).all() and (p.sat_frac == 0).all()


def test_matched_conditions_in_saved_results():
    ep = pd.read_csv(RES / "episodes.csv")
    a, b = ep[ep.controller == "passive"], ep[ep.controller == "ppo"]
    assert len(a) == len(b)
    for col in ("scenario", "seed", "road_label", "speed", "initial_state_offset"):
        assert (a[col].to_numpy() == b[col].to_numpy()).all()
    assert ep.seed.min() >= 100000  # held-out evaluation seeds only


def test_report_table_matches_csv():
    tab = (ROOT / "report" / "results_table.tex")
    if not tab.exists():
        pytest.skip("report assets not built")
    ep = pd.read_csv(RES / "episodes.csv")
    row = [r for r in tab.read_text().split("\\\\\n") if r.startswith("ISO C 15 m/s")][0]
    cells = [c.strip() for c in row.split("&")]
    d = ep[ep.scenario == "rough_C_15"]
    p, q = d[d.controller == "passive"].rms_acc.mean(), d[d.controller == "ppo"].rms_acc.mean()
    assert float(cells[2]) == pytest.approx(p, abs=5e-4) and float(cells[3]) == pytest.approx(q, abs=5e-4)
    assert float(cells[4]) == pytest.approx(100 * (p - q) / p, abs=0.06)
