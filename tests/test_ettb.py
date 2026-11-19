"""ETTB v1 (JAX): replay, counterfactual invariances and the separation of truth from the record."""
import numpy as np
import pandas as pd
import pytest

from ettb import env as E
from ettb.params import default_params
from ettb.world import observe, sample_world, to_frame

P = default_params()


@pytest.fixture(scope="module")
def world():
    return sample_world(5, 120_000, P)


def _np(r):
    return {k: np.asarray(v) for k, v in r.items()}


def test_patient_identical_regardless_of_cohort():
    a = to_frame(sample_world(3, 300, P)).set_index("pid")
    b = to_frame(sample_world(3, 1000, P)).set_index("pid")
    pd.testing.assert_frame_equal(a.loc[[0, 150, 299]], b.loc[[0, 150, 299]])
    c = to_frame(sample_world(3, 100, P, start_id=200)).set_index("pid")
    pd.testing.assert_frame_equal(a.loc[[200, 299]], c.loc[[200, 299]])


def test_replay_is_exact(world):
    pol = E.default_policy(E.FIXED, 1)
    a, b = _np(E.run_policy(9, world, pol)), _np(E.run_policy(9, world, pol))
    for k in a:
        assert np.array_equal(a[k], b[k]), k


def test_counterfactual_invariances(world):
    never = _np(E.run_policy(9, world, E.default_policy(E.FIXED, 99)))
    always = _np(E.run_policy(9, world, E.default_policy(E.FIXED, 0)))
    x = {k: np.asarray(v) for k, v in E._arrays(world).items()}
    nonbact = ~x["bact"]
    same_activity = x["bact"] & (x["a_A"] == x["a_B"])
    for mask in (nonbact, same_activity):
        assert np.array_equal(never["died"][mask], always["died"][mask])
        assert np.array_equal(never["death_day"][mask], always["death_day"][mask])
    assert never["b_days"].sum() == 0 and not never["stepped"].any()
    surv = ~always["died"]
    assert np.all(always["b_days"][surv] == always["result_day"][surv])


def test_oracle_dominates_on_coverage_and_carbapenem_use(world):
    o = _np(E.run_policy(9, world, E.default_policy(E.ORACLE)))
    for tau in (0, 1, 99):
        f = _np(E.run_policy(9, world, E.default_policy(E.FIXED, tau)))
        assert o["inactive_days"].sum() <= f["inactive_days"].sum() + 1e-6
    assert not o["unnecessary_b"].any()


def test_inactive_therapy_harms_bacterial_patients(world):
    x = {k: np.asarray(v) for k, v in E._arrays(world).items()}
    never = _np(E.run_policy(9, world, E.default_policy(E.FIXED, 99)))
    oracle = _np(E.run_policy(9, world, E.default_policy(E.ORACLE)))
    helped = x["bact"] & ~x["a_A"] & x["a_B"]
    assert oracle["died"][helped].mean() < never["died"][helped].mean()


def test_observation_noise_only_touches_the_record(world):
    p = dict(P, record_completeness=0.5, ast_vme=0.05)
    o = {k: np.asarray(v) for k, v in observe(1, world, p).items()}
    truth = np.asarray(world["prior_r"])
    assert np.all(o["prior_r_obs"] <= truth)            # can only lose history, never invent it
    frac = o["prior_r_obs"][:, 0][truth[:, 0]].mean()
    assert 0.45 < frac < 0.55
    sir = np.asarray(world["sir"])
    flipped = (sir == 2) & (o["sir_obs"] == 0)
    assert 0.04 < flipped.sum() / (sir == 2).sum() < 0.06
