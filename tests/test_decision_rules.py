"""The example decision rules and the frontier built on them (examples/decision_rules.py, decision_frontier.py)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))
import decision_frontier as df  # noqa: E402
import decision_rules as dr  # noqa: E402


@pytest.mark.parametrize("rule", list(dr.RULES))
def test_rules_are_linear_and_switch_at_their_threshold(rule):
    f, s = dr.RULES[rule], dr.SETTINGS
    for p in (0.1, 0.37, 0.8):
        assert f(p, s) == pytest.approx((1 - p) * f(0.0, s) + p * f(1.0, s), rel=1e-9, abs=1e-9)
    t = dr.effective_threshold(rule)
    assert 0 < t < 1 and f(t - 1e-6, s) < 0 < f(t + 1e-6, s)


def test_net_benefit_threshold_by_hand():
    # Value if covered 40,000 x 1.05 + 3,000 = 45,000; gain 0.95p; extra cost 9,450 + 9,000.
    assert dr.effective_threshold("Net benefit") == pytest.approx(18_450 / (0.95 * 45_000))


def test_threshold_edges_and_draws():
    assert dr.effective_threshold("Weighted sum", {**dr.SETTINGS, "omega": 0.3}) == 1.0
    assert dr.effective_threshold("Net benefit", {**dr.SETTINGS, "extra_drug_cost": -20_000.0}) == 0.0
    t = dr.threshold_draws("DOOR-MAT", 200, seed=3)
    assert ((t >= 0) & (t <= 1)).all() and t.std() > 0.02
    assert np.array_equal(t, dr.threshold_draws("DOOR-MAT", 200, seed=3))


def test_frontier_runs_and_b_days_fall_as_the_threshold_rises(monkeypatch):
    monkeypatch.setattr(df, "GRID", np.array([0.0, 0.2, 0.5, 1.0]))
    out = df.frontier(0.9, 3000, draws=10)
    c = out[out.curve == "predictor"].sort_values("t")
    assert np.all(np.diff(c.b_days.to_numpy()) <= 0) and c.b_days.iloc[-1] == 0
    assert out[out.curve == "never"].deaths_averted.iloc[0] == 0
    assert set(out.curve) >= set(dr.RULES) and out[out.curve.isin(dr.RULES)].t_low.notna().all()
