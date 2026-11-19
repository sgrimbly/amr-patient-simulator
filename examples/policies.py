"""Decision models for the full-loop example. Each has reset(first_look) and act(day); see ettb/loop.py.

These are baselines for comparison, not recommendations. The thresholds used in full_loop.py are illustrative.
"""
from __future__ import annotations

import numpy as np
from scipy.special import expit, logit
from scipy.stats import norm


class Never:
    """Stay on drug A until the culture result."""

    def reset(self, first_look):
        pass

    def act(self, day):
        return np.zeros_like(day["alive"])


class AlwaysB:
    """Start everyone on drug B on day 0."""

    def reset(self, first_look):
        pass

    def act(self, day):
        return np.full_like(day["alive"], day["t"] == 0)


class Threshold:
    """Step up on day 0 when the predictor's P(A fails) exceeds c; never later."""

    def __init__(self, p_fail, c):
        self.p_fail, self.c = np.asarray(p_fail), c

    def reset(self, first_look):
        pass

    def act(self, day):
        return (day["t"] == 0) & (self.p_fail > self.c)


class Belief:
    """Combine the predictor with the daily vital signs, and step up once P(A fails) exceeds c.

    Two Kalman filters track severity, one assuming A is failing (severity drifts up by kappa a day) and one assuming
    it works (down by lam). Their likelihood ratio on the observed vital signs updates the predictor's prior odds.
    This is the numpy version of env.BELIEF; the filter's settings are the model's beliefs, not the simulator's truth.
    """

    def __init__(self, p_fail, c, phi=0.2, s0=0.0, kappa=0.6, lam=0.3, sigma=0.5, sigma_v=0.7, mu0=0.2, var0=1.3,
                 sev_obs_sd=0.5):
        self.prior = np.clip(np.asarray(p_fail, float), 1e-6, 1 - 1e-6)
        self.c = c
        self.phi, self.s0, self.kappa, self.lam = phi, s0, kappa, lam
        self.sigma, self.sigma_v, self.mu0, self.var0, self.sev_obs_sd = sigma, sigma_v, mu0, var0, sev_obs_sd

    def reset(self, first_look):
        k0 = self.var0 / (self.var0 + self.sev_obs_sd ** 2)
        m = self.mu0 + k0 * (first_look["presenting_severity"] - self.mu0)
        P = np.full_like(m, (1 - k0) * self.var0)
        self.m = {"fails": m.copy(), "works": m.copy()}
        self.P = {"fails": P.copy(), "works": P.copy()}
        self.loglik_ratio = np.zeros_like(m)

    def _filter(self, h, drift, v, seen):
        mp = (1 - self.phi) * self.m[h] + self.phi * self.s0 + drift
        Pp = (1 - self.phi) ** 2 * self.P[h] + self.sigma ** 2
        Sv = Pp + self.sigma_v ** 2
        K = Pp / Sv
        vv = np.where(seen, v, mp)
        self.m[h] = np.where(seen, mp + K * (vv - mp), mp)
        self.P[h] = np.where(seen, (1 - K) * Pp, Pp)
        return np.where(seen, norm.logpdf(vv, mp, np.sqrt(Sv)), 0.0)

    def p_fail_now(self):
        return expit(logit(self.prior) + self.loglik_ratio)

    def act(self, day):
        if day["t"] > 0:
            seen = ~np.isnan(day["vitals"])
            l_fails = self._filter("fails", self.kappa, day["vitals"], seen)
            l_works = self._filter("works", -self.lam, day["vitals"], seen)
            self.loglik_ratio += l_fails - l_works
        return self.p_fail_now() > self.c


class Oracle:
    """Step up on day 0 exactly when it helps: bacterial infection, A inactive and B active. It sees the truth, so
    it is a reference point, not a policy anyone could use."""

    def __init__(self, helps):
        self.helps = np.asarray(helps, bool)

    def reset(self, first_look):
        pass

    def act(self, day):
        return (day["t"] == 0) & self.helps


class MyPolicy:
    """Template. Replace act() with your decision model.

    reset() receives the presenting severity reading. act() is called every morning with day["t"], day["vitals"]
    (NaN when not measured), day["can_step"], day["result_back"] and day["alive"], and returns one bool per patient.
    Requests where can_step is False are ignored, and a step-up cannot be undone before the result.
    """

    def __init__(self, p_fail):
        self.p_fail = np.asarray(p_fail)

    def reset(self, first_look):
        pass

    def act(self, day):
        return np.zeros_like(day["alive"])
