"""Predictors for the full-loop example: from the observed record to P(drug A fails).

A predictor here sees only the record, and is trained on what a hospital would have: culture results for the
patients with a bacterial infection, read through the laboratory's (imperfect) susceptibility testing. It is then
scored against the truth, which no hospital has.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import rankdata

from ettb import env as E
from ettb.params import ABX_CLASSES

# Variables a clinician or model can see before the culture result. Prior antibiotics, prior resistant isolates and
# coded comorbidities come from the observed record, so they carry its gaps.
RECORDED = ["age", "female", "immunosuppressed", "onset_hai", "hosp_12m_count", "hosp_90d", "care_home", "transfer_in",
            "urinary_catheter", "ppi", "has_prior_culture", "syndrome_bsi", "icu", "egfr_low"]


def drug_a(world):
    """Index into DRUGS of each patient's first drug: piperacillin-tazobactam if hospital onset, else ceftriaxone."""
    return np.where(np.asarray(world["onset_hai"]), 1, 0)


def record_table(world, obs) -> pd.DataFrame:
    w = {k: np.asarray(v) for k, v in world.items()}
    o = {k: np.asarray(v) for k, v in obs.items()}
    df = pd.DataFrame({k: w[k] for k in RECORDED}).astype(float)
    df["age"] = (df["age"] - 62) / 17
    for k in ("diabetes_obs", "ckd_obs", "hiv_obs"):
        df[k] = o[k].astype(float)
    for j, c in enumerate(ABX_CLASSES[:5]):
        df[f"prior_abx_{c}"] = o["prior_abx_obs"][:, j].astype(float)
    for j, d in enumerate(("ceftriaxone", "meropenem", "ciprofloxacin", "amikacin")):
        df[f"prior_r_{d}"] = o["prior_r_obs"][:, j].astype(float)
    return df


def culture_labels(world, obs):
    """(label, labelled): drug A tested resistant, known only for patients with a bacterial infection."""
    rows = np.arange(len(np.asarray(world["pid"])))
    sir_obs = np.asarray(obs["sir_obs"])[rows, drug_a(world)]
    return sir_obs == 2, np.asarray(world["bacterial"])


def truth(world, env_params) -> dict:
    """Hidden truth for scoring: bacterial infection, and whether drugs A and B are active. Activity follows the
    simulator's rules, so piperacillin-tazobactam can fail against an OXA-1 co-producer that tests susceptible."""
    e = {k: jnp.asarray(v, jnp.float32) for k, v in {**E.default_env_params(), **(env_params or {})}.items()}
    x = E._arrays(world, None, e)
    bact = np.asarray(x["bact"])
    a_fails = bact & (np.asarray(x["a_A"]) < 0.5)
    return {"bacterial": bact, "a_fails": a_fails, "b_helps": a_fails & (np.asarray(x["a_B"]) > 0.5)}


def auroc(score, label):
    label = np.asarray(label, bool)
    r = rankdata(score)
    n1, n0 = label.sum(), (~label).sum()
    return (r[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


class Logistic:
    """L2-penalised logistic regression fitted by Newton's method (numpy only)."""

    def __init__(self, l2=1e-3):
        self.l2 = l2

    def fit(self, df: pd.DataFrame, y):
        self.columns = list(df.columns)
        X = np.column_stack([np.ones(len(df)), df.to_numpy(float)])
        y = np.asarray(y, float)
        beta = np.zeros(X.shape[1])
        pen = self.l2 * len(y) * np.r_[0.0, np.ones(X.shape[1] - 1)]
        for _ in range(50):
            p = expit(X @ beta)
            grad = X.T @ (y - p) - pen * beta
            hess = (X * (p * (1 - p))[:, None]).T @ X + np.diag(pen)
            step = np.linalg.solve(hess, grad)
            beta += step
            if np.abs(step).max() < 1e-8:
                break
        self.coef = beta
        return self

    def predict(self, df: pd.DataFrame):
        X = np.column_stack([np.ones(len(df)), df[self.columns].to_numpy(float)])
        return expit(X @ self.coef)


class PriorIsolate:
    """One-variable rule: the resistance rate among labelled patients with and without a prior isolate on record
    resistant to ceftriaxone."""

    def fit(self, df, y):
        y = np.asarray(y, bool)
        flag = df["prior_r_ceftriaxone"].to_numpy() > 0.5
        self.rate = {True: y[flag].mean(), False: y[~flag].mean()}
        return self

    def predict(self, df):
        flag = df["prior_r_ceftriaxone"].to_numpy() > 0.5
        return np.where(flag, self.rate[True], self.rate[False])
