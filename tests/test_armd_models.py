"""The ARMD model loader: feature mapping, NumPy forwards, and (when the release is unpacked) the published models."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "examples"))

import armd_models as AM  # noqa: E402
from ettb.params import calibrated_site  # noqa: E402
from ettb.world import observe, sample_world  # noqa: E402

MANIFEST = json.loads((ROOT / "models" / "manifest.json").read_text())
RELEASED = (AM.MODELS / AM.RELEASE["name"] / "manifest.json").exists()


@pytest.fixture(scope="module")
def cohort():
    wp, _ = calibrated_site(0.9)
    w = sample_world(2, 3000, wp)
    return w, observe(2, w, wp)


def prep(columns, missing=(), log=()):
    d = len(columns)
    return dict(prep_columns=np.array(columns), prep_median=np.full(d, 2.0, np.float32),
                prep_missing=np.array(missing, np.int64), prep_log=np.array(log, np.int64),
                prep_mean=np.zeros(d, np.float32), prep_sd=np.full(d, 2.0, np.float32))


def test_shared_features_match_the_published_columns(cohort):
    w, o = cohort
    X = AM.shared_features(w, o, "ceftriaxone")
    assert list(X.columns) == MANIFEST["shared_columns"]
    assert X.notna().all().all()
    # One age band (or the 18-24 reference), and ward and ICU are exclusive and exhaustive.
    assert (X.filter(like="age_band_").sum(axis=1) <= 1).all()
    assert (X.care_icu + X.care_ward == 1).all()
    # Same-drug prior resistance is one of the prior resistant results.
    assert (X.prior_r_same <= X.prior_r_any).all() and (X.prior_r_count >= X.prior_r_any).all()


def test_same_drug_resistance_follows_the_drug(cohort):
    w, o = cohort
    pr = np.asarray(o["prior_r_obs"]).astype(bool)
    assert (AM.shared_features(w, o, "meropenem").prior_r_same == pr[:, 1]).all()
    assert (AM.shared_features(w, o, "piperacillin_tazobactam").prior_r_same == 0).all()


def test_preprocessing_imputes_logs_and_standardises():
    a = prep(["x", "y"], missing=[1], log=[0])
    X = pd.DataFrame({"x": [np.e - 1, 0.0], "y": [np.nan, 4.0]})
    A = AM._Prep(a)(X)
    np.testing.assert_allclose(A[:, 0], [0.5, 0.0], atol=1e-6)     # log1p, then / sd 2
    np.testing.assert_allclose(A[:, 1], [1.0, 2.0])                # NaN -> median 2, then / 2
    np.testing.assert_allclose(A[:, 2], [1.0, 0.0])                # missing indicator for y


def test_logistic_and_mlp_forwards():
    a = prep(["x"]) | dict(w=np.array([0.5, 2.0]))
    X = pd.DataFrame({"x": [0.0, 2.0]})
    np.testing.assert_allclose(AM.Logistic(a).predict(X), expit([0.5, 2.5]))
    # One-member MLP that passes x / 2 through ReLUs: output = sigmoid(relu(relu(z)) - 1).
    m = prep(["x"]) | {"member0_W0": np.eye(1), "member0_b0": np.zeros(1), "member0_W1": np.eye(1),
                       "member0_b1": np.zeros(1), "member0_W2": np.eye(1), "member0_b2": -np.ones(1)}
    X = pd.DataFrame({"x": [-4.0, 6.0]})
    np.testing.assert_allclose(AM.MLPEnsemble(m).predict(X), expit([-1.0, 2.0]))


def test_bayes_known_and_new_sites():
    k = 400
    rng = np.random.default_rng(0)
    a = prep(["x"]) | dict(beta=np.ones((k, 1)), mu_a=np.zeros(k), tau_a=np.full(k, 0.5),
                           intercept_site=np.column_stack([np.full(k, -1.0), rng.normal(size=k)]),
                           sites=np.array(["a", "b"]))
    m = AM.BayesLogistic(a)
    X = pd.DataFrame({"x": [0.0]})
    np.testing.assert_allclose(m.predict(X, site="a"), expit(-1.0))
    d = m.predict_draws(X)                     # unseen site: intercepts from the population of sites
    assert d.shape == (k, 1) and 0.4 < np.std(np.log(d / (1 - d))) < 0.6


@pytest.mark.skipif(not RELEASED, reason="release not downloaded (examples/armd_models.py: fetch())")
def test_released_models_load_and_respond_to_prior_resistance(cohort):
    import importlib.util
    has_xgb = importlib.util.find_spec("xgboost") is not None
    w, o = cohort
    checked = 0
    for e in AM.manifest()["models"]:
        if e["features"] != "shared" or e["setting"] != "inpatient":
            continue
        if e["family"] == "xgboost" and not has_xgb:
            continue
        checked += 1
        m = AM.load(e["family"], f"{e['setting']}_{e['specimen']}_{e['drug']}", "shared")
        X = AM.shared_features(w, o, e["drug"])
        p = m.predict(X)
        assert p.shape == (len(X),) and np.all((p >= 0) & (p <= 1)), e["file"]
        if e["drug"] == "ceftriaxone":         # a prior ceftriaxone-resistant isolate lowers P(susceptible)
            on, off = X.assign(prior_r_same=1.0, prior_r_any=1.0), X.assign(prior_r_same=0.0)
            assert m.predict(on).mean() < m.predict(off).mean(), e["file"]
    assert checked >= 36
