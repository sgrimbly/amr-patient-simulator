"""Bayesian WISCA: weighted-incidence syndromic combination antibiogram.

An antibiogram is a hospital's table of how often each organism tested susceptible to each antibiotic, built from
its own culture results over a period (often a year). It is the usual evidence behind a local empiric guideline.

A WISCA answers the question a prescriber actually faces before the culture result: for a patient with this
syndrome, how likely is this regimen to cover whatever organism is causing it? It weights each organism's
susceptibility by how often that organism causes the syndrome, counts a combination as covering an isolate if any
of its drugs does, and (in the Bayesian version) gives an interval that reflects how few isolates a hospital has:

    coverage(regimen | syndrome) = sum over organisms o of  P(o | syndrome) * P(isolate of o susceptible to regimen)

with a Dirichlet posterior for the organism mix and a Beta posterior for each organism's susceptibility. Optional
pseudo-counts from a reference (pooled or national) antibiogram let a small hospital borrow strength.

Hebert et al. 2012 introduced the weighted-incidence approach; Bielicki et al. 2016 (J Antimicrob Chemother 71:794)
made it Bayesian and pooled local with multi-hospital data.

    import wisca
    w = wisca.Wisca().fit(world, obs)        # from the culture results in a simulated history
    w.table()                                # coverage, 95% interval and isolate count per syndrome and regimen
    p_fail = w.p_fail(world)                 # per patient: 1 - coverage of their drug A in their syndrome
"""
from __future__ import annotations

import numpy as np
import pandas as pd

DRUGS = ("ceftriaxone", "piperacillin_tazobactam", "ertapenem", "meropenem", "amikacin", "ciprofloxacin")
ORGANISMS = ("E. coli", "K. pneumoniae", "P. aeruginosa")
STRATA = ("community UTI", "hospital UTI", "community BSI", "hospital BSI")
REGIMENS = {
    "ceftriaxone": ("ceftriaxone",),
    "piperacillin-tazobactam": ("piperacillin_tazobactam",),
    "piperacillin-tazobactam + amikacin": ("piperacillin_tazobactam", "amikacin"),
    "ceftriaxone + amikacin": ("ceftriaxone", "amikacin"),
    "ertapenem": ("ertapenem",),
    "meropenem": ("meropenem",),
    "amikacin": ("amikacin",),
    "ciprofloxacin": ("ciprofloxacin",),
}


def strata(world) -> np.ndarray:
    """0 community UTI, 1 hospital UTI, 2 community BSI, 3 hospital BSI (the simulator's syndrome x onset)."""
    return 2 * np.asarray(world["syndrome_bsi"]).astype(int) + np.asarray(world["onset_hai"]).astype(int)


def covered(sir: np.ndarray, regimen) -> np.ndarray:
    """Isolate covered by the regimen: susceptible or intermediate (the simulator's activity rule) to any drug."""
    cols = [DRUGS.index(d) for d in REGIMENS[regimen]]
    return (sir[:, cols] <= 1).any(axis=1)


class Wisca:
    def __init__(self, draws: int = 4000, seed: int = 0, prior: "Wisca | None" = None, prior_strength: float = 0.0):
        """prior: a fitted reference WISCA (pooled or national); prior_strength: how many isolates it is worth per
        syndrome, spread over organisms by the reference's organism mix (0 = flat priors)."""
        self.draws, self.seed, self.prior, self.prior_strength = draws, seed, prior, prior_strength

    def fit(self, world, obs, mask=None):
        """Use the culture results available to the hospital: bacterial patients (the laboratory identifies the
        organism and reports susceptibility, with its own errors), optionally restricted by `mask`."""
        org = np.asarray(world["organism"])
        sir = np.asarray(obs["sir_obs"])
        use = np.asarray(world["bacterial"]).astype(bool) & (org <= 2)
        if mask is not None:
            use &= np.asarray(mask, bool)
        st = strata(world)
        self.n_org = np.zeros((4, 3))
        self.k = {r: np.zeros((4, 3)) for r in REGIMENS}
        for s in range(4):
            for o in range(3):
                m = use & (st == s) & (org == o)
                self.n_org[s, o] = m.sum()
                for r in REGIMENS:
                    self.k[r][s, o] = covered(sir[m], r).sum()
        self._draws = {}
        return self

    def _posterior(self):
        """Posterior draws: organism mix (strata x draws x organisms), susceptibility per regimen (same shape)."""
        if self._draws:
            return self._draws
        rng = np.random.default_rng(self.seed)
        a_org = 1.0 + self.n_org
        pk = {r: 0.0 for r in REGIMENS}
        pn = 0.0
        if self.prior is not None and self.prior_strength > 0:
            mix = self.prior.n_org / np.maximum(self.prior.n_org.sum(1, keepdims=True), 1)
            a_org = a_org + self.prior_strength * mix
            pn = self.prior_strength * mix                                  # pseudo-isolates per organism
            for r in REGIMENS:
                pk[r] = pn * (self.prior.k[r] + 1) / (self.prior.n_org + 2)  # reference susceptibility x weight
        mix = np.stack([rng.dirichlet(a_org[s], self.draws) for s in range(4)])
        sus = {}
        for r in REGIMENS:
            a = 1.0 + self.k[r] + pk[r]
            b = 1.0 + (self.n_org - self.k[r]) + (pn - pk[r])
            sus[r] = rng.beta(a[:, None, :], b[:, None, :], size=(4, self.draws, 3))
        self._draws = dict(mix=mix, sus=sus)
        return self._draws

    def coverage(self, stratum: int, regimen: str) -> np.ndarray:
        d = self._posterior()
        return (d["mix"][stratum] * d["sus"][regimen][stratum]).sum(-1)

    def table(self) -> pd.DataFrame:
        rows = []
        for s, name in enumerate(STRATA):
            for r in REGIMENS:
                c = self.coverage(s, r)
                rows.append(dict(syndrome=name, regimen=r, isolates=int(self.n_org[s].sum()), coverage=c.mean(),
                                 lo=np.quantile(c, 0.025), hi=np.quantile(c, 0.975)))
        return pd.DataFrame(rows)

    def p_fail(self, world) -> np.ndarray:
        """Per patient: 1 - posterior mean coverage of drug A (ceftriaxone if community onset, piperacillin-
        tazobactam if hospital onset) in the patient's syndrome. A predictor with no patient-level features."""
        st = strata(world)
        cov = np.array([self.coverage(s, "piperacillin-tazobactam" if s % 2 else "ceftriaxone").mean()
                        for s in range(4)])
        return 1 - cov[st]


def true_coverage(world) -> pd.DataFrame:
    """What the WISCA estimates, from the simulator's truth: share of all bacterial infections in each syndrome
    (cultured or not) whose isolate tests susceptible to the regimen, without laboratory error."""
    org = np.asarray(world["organism"])
    sir = np.asarray(world["sir"])
    bact = np.asarray(world["bacterial"]).astype(bool) & (org <= 2)
    st = strata(world)
    rows = []
    for s, name in enumerate(STRATA):
        m = bact & (st == s)
        for r in REGIMENS:
            rows.append(dict(syndrome=name, regimen=r, true_coverage=covered(sir[m], r).mean()))
    return pd.DataFrame(rows)
