"""Bayesian WISCA: estimates match the simulator's truth, intervals behave, combinations and pooling work."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import wisca as W  # noqa: E402
from ettb.params import calibrated_site  # noqa: E402
from ettb.world import observe, sample_world  # noqa: E402


@pytest.fixture(scope="module")
def site():
    wp, _ = calibrated_site(0.9)
    w = sample_world(4, 60_000, wp)
    return w, observe(4, w, wp)


def test_large_sample_matches_truth(site):
    w, o = site
    est = W.Wisca().fit(w, o).table()
    truth = W.true_coverage(w)
    m = est.merge(truth, on=["syndrome", "regimen"])
    # Laboratory error and cultured-only isolates aside, a hospital-sized history recovers coverage closely.
    assert (m.coverage - m.true_coverage).abs().max() < 0.03
    assert ((m.true_coverage >= m.lo - 0.01) & (m.true_coverage <= m.hi + 0.01)).mean() > 0.9


def test_fewer_isolates_give_wider_intervals(site):
    w, o = site
    rng = np.random.default_rng(0)
    small = rng.random(len(np.asarray(w["pid"]))) < 0.03
    big = W.Wisca().fit(w, o).table()
    few = W.Wisca().fit(w, o, mask=small).table()
    assert ((few.hi - few.lo) > (big.hi - big.lo)).all()


def test_combination_covers_at_least_its_parts(site):
    w, o = site
    t = W.Wisca().fit(w, o).table().set_index(["syndrome", "regimen"]).coverage
    for s in W.STRATA:
        combo = t[(s, "piperacillin-tazobactam + amikacin")]
        assert combo >= t[(s, "piperacillin-tazobactam")] - 0.005
        assert combo >= t[(s, "amikacin")] - 0.005


def test_pooling_pulls_a_small_hospital_towards_the_reference(site):
    w, o = site
    rng = np.random.default_rng(1)
    small = rng.random(len(np.asarray(w["pid"]))) < 0.01
    ref = W.Wisca().fit(w, o)
    alone = W.Wisca().fit(w, o, mask=small).table().coverage.to_numpy()
    pooled = W.Wisca(prior=ref, prior_strength=200).fit(w, o, mask=small).table().coverage.to_numpy()
    target = ref.table().coverage.to_numpy()
    assert np.abs(pooled - target).mean() < np.abs(alone - target).mean()


def test_p_fail_is_one_minus_drug_a_coverage(site):
    w, o = site
    est = W.Wisca().fit(w, o)
    p = est.p_fail(w)
    st = W.strata(w)
    assert np.allclose(p[st == 0], 1 - est.coverage(0, "ceftriaxone").mean())
    assert np.allclose(p[st == 3], 1 - est.coverage(3, "piperacillin-tazobactam").mean())
