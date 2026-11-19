"""The day-by-day loop, the example decision models and the calibrated parameter file."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from ettb import env as E
from ettb.loop import run_loop
from ettb.params import default_params
from ettb.world import observe, sample_world

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import policies as pol  # noqa: E402
from full_loop import load_site  # noqa: E402

P = default_params()
N = 20_000


@pytest.fixture(scope="module")
def cohort():
    w = sample_world(5, N, P)
    return w, observe(5, w, P)


class _Fixed:
    def __init__(self, tau):
        self.tau = tau

    def reset(self, first_look):
        pass

    def act(self, day):
        return np.full(day["alive"].shape, day["t"] >= self.tau)


def _np(r):
    return {k: np.asarray(v) for k, v in r.items()}


@pytest.mark.parametrize("tau", [0, 2, 99])
def test_loop_reproduces_run_policy_exactly(cohort, tau):
    w, o = cohort
    a = run_loop(9, w, _Fixed(tau), obs=o)
    b = _np(E.run_policy(9, w, E.default_policy(E.FIXED, tau), obs=o))
    for k in b:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_threshold_policy_matches_the_built_in_prior_rule(cohort):
    w, o = cohort
    p = np.random.default_rng(0).uniform(size=N).astype(np.float32)
    a = run_loop(9, w, pol.Threshold(p, 0.7), obs=o)
    b = _np(E.run_policy(9, w, E.default_policy(E.PRIOR, c=0.7), prior=p, obs=o))
    for k in b:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


def test_numpy_belief_rule_agrees_with_the_built_in_one(cohort):
    # float64 numpy against float32 JAX: decisions may differ only for patients sitting on the threshold.
    w, o = cohort
    p = np.random.default_rng(1).uniform(0, 0.6, size=N).astype(np.float32)
    a = run_loop(9, w, pol.Belief(p, 0.5), obs=o)
    b = _np(E.run_policy(9, w, E.default_policy(E.BELIEF, c=0.5), prior=p, obs=o))
    assert b["stepped"].sum() > 100 and (b["step_day"] > 0).sum() > 50
    assert (a["step_day"] == b["step_day"]).mean() > 0.999


def test_model_cannot_step_after_the_result_or_twice(cohort):
    w, o = cohort
    r = run_loop(9, w, _Fixed(0), obs=o)
    late = run_loop(9, w, _Fixed(5), obs=o)           # day 5 is after every possible result day
    assert r["step_up"].sum(1).max() == 1
    assert not late["stepped"].any()


def test_model_sees_no_vitals_on_day_0_and_none_after_death(cohort):
    w, o = cohort
    r = run_loop(9, w, pol.Never(), obs=o)
    assert np.isnan(r["vitals"][:, 0]).all()
    dead = r["died"] & (r["death_day"] < 29)
    i = np.flatnonzero(dead)[0]
    assert np.isnan(r["vitals"][i, r["death_day"][i] + 1:]).all()


def test_mechanism_age_slope_moves_only_its_mechanism():
    w0 = _np(sample_world(13, 200_000, P))
    w1 = _np(sample_world(13, 200_000, dict(P, col_age_mech=np.array([0.0, 0.0, 1.0, 0.0]))))
    np.testing.assert_array_equal(w0["col"][:, [0, 1, 3]], w1["col"][:, [0, 1, 3]])
    old, young = w0["age"] >= 60, w0["age"] < 60
    r0 = w0["col"][old, 2].mean() / w0["col"][young, 2].mean()
    r1 = w1["col"][old, 2].mean() / w1["col"][young, 2].mean()
    assert r1 > 1.3 * r0


def test_calibrated_parameter_file_covers_every_parameter():
    rep = json.loads((ROOT / "params/ettb_v1_1d.json").read_text())
    assert set(rep["calibrated_params"]) == set(P)
    assert sorted(f["theta1"] for f in rep["outcome_family"]) == [0.6, 0.9, 1.2]
    wp, ep = load_site(ROOT / "params/ettb_v1_1d.json", 0.9)
    assert set(ep) <= set(E.default_env_params())
    assert np.asarray(wp["col_age_mech"])[2] > 0
