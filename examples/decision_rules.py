"""Published decision rules for stepping up to drug B, written as functions of p = P(drug A fails).

Each rule scores escalation; a positive score means step up. All of them are linear in p, because the chance that a
drug covers the infection is, so each one steps up exactly when p passes its effective threshold. The values below
are illustrative (money in rand); RANGES holds plausible ranges for seeing how far a threshold moves.
"""
from __future__ import annotations

import numpy as np

SETTINGS = {
    "value_per_life_year": 40_000.0,   # near a South African cost-effectiveness threshold
    "life_years_gained": 1.05,         # if the infection is covered (Leibovici et al. 1999)
    "bed_days_saved": 1.0,             # if the infection is covered (Leibovici et al. 1999)
    "qalys_gained": 0.84,
    "bed_cost": 3_000.0,
    "extra_drug_cost": 9_450.0,        # B minus A, treatment and side effects, one course
    "extra_community_cost": 9_000.0,   # B minus A, future resistance
    "b_covers_if_a_fails": 0.95,       # B covers whatever A covers, and most of what it misses
    "wtp_per_qaly": 40_000.0,
    "resistance_price": 150.0,         # price of one course's resistance pressure (Shrestha et al. 2018)
    "omega": 0.8,                      # weight on coverage (Boominathan et al. 2020)
    "overtreatment_score": 50.0,       # DOOR-MAT score for an unnecessarily broad drug
}
RANGES = {
    "value_per_life_year": (20_000, 80_000), "life_years_gained": (0.5, 2.0), "bed_days_saved": (0.0, 3.0),
    "qalys_gained": (0.4, 1.6), "bed_cost": (1_500, 6_000), "extra_drug_cost": (4_000, 20_000),
    "extra_community_cost": (0, 20_000), "b_covers_if_a_fails": (0.85, 0.99), "wtp_per_qaly": (20_000, 80_000),
    "resistance_price": (40, 260), "omega": (0.6, 0.95), "overtreatment_score": (20, 80),
}


def gain(p, s):
    """P(B covers the infection) - P(A covers it)."""
    return p * s["b_covers_if_a_fails"]


def value_covered(s):
    return s["value_per_life_year"] * s["life_years_gained"] + s["bed_cost"] * s["bed_days_saved"]


def net_benefit(p, s):
    """Monetised net benefit, as in TREAT (Andreassen et al. 1999). Weighting the community cost by lam / (1 - lam)
    gives the lam-weighted rule that trades the patient's benefit against the community's cost."""
    return gain(p, s) * value_covered(s) - s["extra_drug_cost"] - s["extra_community_cost"]


def willingness_to_pay(p, s):
    """QALYs gained at a willingness to pay, against cost net of bed-days saved and a price of resistance."""
    g = gain(p, s)
    return (s["wtp_per_qaly"] * s["qalys_gained"] * g - s["extra_drug_cost"]
            + g * s["bed_cost"] * s["bed_days_saved"] - s["resistance_price"])


def weighted_sum(p, s):
    """Coverage against avoiding the broader drug (Boominathan et al. 2020)."""
    return s["omega"] * gain(p, s) - (1 - s["omega"])


def door_mat(p, s):
    """Expected DOOR-MAT score: 100 for the narrowest active drug, 0 if inactive (Claeys et al. 2021)."""
    return (1 - p) * s["overtreatment_score"] + p * s["b_covers_if_a_fails"] * 100 - (1 - p) * 100


RULES = {"Net benefit": net_benefit, "Willingness to pay": willingness_to_pay, "Weighted sum": weighted_sum,
         "DOOR-MAT": door_mat}


def effective_threshold(rule, s=SETTINGS) -> float:
    """The p above which the rule steps up: 0 if it always does, 1 if it never does."""
    s0, s1 = RULES[rule](0.0, s), RULES[rule](1.0, s)
    if s0 > 0:
        return 0.0
    return 1.0 if s1 <= 0 else -s0 / (s1 - s0)


def threshold_draws(rule, n, seed=0) -> np.ndarray:
    """Effective thresholds for n settings drawn uniformly from RANGES."""
    rng = np.random.default_rng(seed)
    return np.array([effective_threshold(rule, {k: rng.uniform(*r) for k, r in RANGES.items()}) for _ in range(n)])
