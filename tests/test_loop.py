"""The day-by-day loop, the example decision models and the calibrated parameter file."""
import sys
from pathlib import Path

import numpy as np
import pytest

from ettb import env as E
from ettb.loop import run_loop
from ettb.params import calibrated_site, default_params
from ettb.world import observe, sample_world

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))
import policies as pol  # noqa: E402

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


@pytest.mark.parametrize("theta1", [0.6, 0.9, 1.2])
def test_calibrated_site_sets_every_parameter(theta1):
    wp, ep = calibrated_site(theta1)
    assert set(wp) == set(P)
    assert set(ep) <= set(E.default_env_params()) and ep["theta1"] == theta1
    assert np.asarray(wp["col_age_mech"])[2] > 0


def test_calibrated_site_rejects_an_unknown_outcome_setting():
    with pytest.raises(ValueError):
        calibrated_site(0.75)


def test_expected_deaths_match_realised_deaths(cohort):
    w, o = cohort
    for model in (_Fixed(99), _Fixed(0)):
        r = run_loop(9, w, model, obs=o)
        se = np.sqrt(r["died"].mean() * (1 - r["died"].mean()) / N)
        assert abs(r["p_death"].mean() - r["died"].mean()) < 4 * se
        assert ((r["p_death"] >= 0) & (r["p_death"] <= 1)).all()


def test_expected_deaths_are_unchanged_where_treatment_does_not_matter(cohort):
    w, o = cohort
    never, always = run_loop(9, w, _Fixed(99), obs=o), run_loop(9, w, _Fixed(0), obs=o)
    x = {k: np.asarray(v) for k, v in E._arrays(w).items()}
    same = (x["a_A"] == x["a_B"]) | ~x["bact"]
    np.testing.assert_allclose(never["p_death"][same], always["p_death"][same], rtol=1e-6)


def test_long_turnaround_option(cohort):
    w, o = cohort
    ep = {"result_mean": 7.0}
    a = run_loop(9, w, _Fixed(5), env_params=ep, obs=o)
    b = _np(E.run_policy(9, w, E.default_policy(E.FIXED, 5), env_params=ep, obs=o))
    for k in b:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)
    rd = a["result_day"]
    assert 6.0 < rd.mean() < 8.0 and rd.min() >= 1 and rd.max() <= 14
    assert (a["step_day"] == 5).sum() > 0                  # stepping up on day 5 is now possible
    long0, short0 = run_loop(9, w, _Fixed(0), env_params=ep, obs=o), run_loop(9, w, _Fixed(0), obs=o)
    assert long0["b_days"].mean() > 2 * short0["b_days"].mean()           # longer on B before the result


def test_sa_anchor_site_loads_and_takes_a_turnaround():
    from ettb.params import sa_anchor_site
    w3, e3 = sa_anchor_site()
    w7, e7 = sa_anchor_site(turnaround=7.0)
    assert e3["result_mean"] == pytest.approx(3.1) and e7["result_mean"] == 7.0
    assert e3["theta1"] == pytest.approx(0.9)
    world = sample_world(1, 2000, w7)
    assert 0.4 < float(np.asarray(world["onset_hai"]).mean()) < 0.65     # Tygerberg: 57% hospital-acquired
