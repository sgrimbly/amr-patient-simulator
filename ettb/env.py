"""One-shot step-up environment (ETTB v1-pre), in JAX.

Everyone starts drug A at t0 (ceftriaxone if community onset, piperacillin-tazobactam if
hospital onset). Before the culture result (day 2 or 3), a policy may switch ONCE,
irreversibly, to drug B (ertapenem; meropenem with Pseudomonas risk). From the result
day, directed therapy (active if any formulary drug is active) until day 30.

Dynamics (daily): S[t+1] = S[t] - phi*(S[t] - s0) + bact*(kappa*(1-a[t]) - lam*a[t])
                           - (1-bact)*lam_nb + sigma*eps[t]
Death: daily probability expit(theta0 + theta1*S[t]). Vital-sign composite each morning:
v[t] = S[t] + sigma_v*xi[t], measured with probability expit(m0 + m1*S[t]) (informative
missingness switch; m0 large = always measured).

All environment randomness is keyed by (patient id, day, stream), never by actions, so the
same patient can be replayed under any policy (exact counterfactuals).

Policy kinds (ints):
  0 FIXED      step up on day `tau` (99 = never)
  1 PRIOR      step up at t0 iff prior probability > c
  2 BELIEF     two-hypothesis Kalman filter (A inactive vs A active) from prior + vitals;
               step up at first review with posterior > c
  3 ORACLE     step up at t0 iff bacterial, A inactive and B active
  4 CLINICIAN  'historical' rule: step up at t0 if a prior resistant isolate is on record,
               otherwise when the observed vital composite exceeds c (with noise)
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import expit, logit, ndtri
from jax.scipy.stats import norm

H = 30
FIXED, PRIOR, BELIEF, ORACLE, CLINICIAN = 0, 1, 2, 3, 4


def default_env_params() -> dict:
    return {"phi": 0.2, "s0": 0.0, "kappa": 0.6, "lam": 0.3, "lam_nb": 0.3, "sigma": 0.5,
            "theta0": -5.5, "theta1": 0.9, "s_det": 2.5, "sigma_v": 0.7,
            "m0": 20.0, "m1": 0.0,                 # m0 large: vitals always measured
            "p_result_day3": 0.5,
            # v1.1c: probabilities of the result on day 1 and day 4 (0 reproduces the day-2/day-3 rule).
            "p_result_day1": 0.0, "p_result_day4": 0.0,
            # Structural alternative for drug activity (link 14). 0 = binary (S or I is fully active, v1).
            # 1 = graded: activity = expit(pta_slope * (S breakpoint + pta_mid - log2 MIC)), a probability of
            # PK/PD target attainment that falls as the MIC approaches and passes the breakpoint.
            "graded_activity": 0.0, "pta_mid": 1.0, "pta_slope": 2.0,
            # Clinical efficacy of piperacillin-tazobactam against an ESBL producer it tests active against, as a
            # fraction of full activity (reduced form of the OXA-1 and inoculum effects implicated by Henderson et
            # al. 2021). 1 reproduces earlier versions; fitted to MERINO in v1.1b.
            "tzp_esbl_eff": 1.0,
            # v1.1c: efficacy of TZP against an OXA-1 co-producing ESBL isolate it tests active against (explicit
            # replacement for tzp_esbl_eff; 1 reproduces earlier versions).
            "tzp_oxa1_eff": 1.0,
            # v1.1c: amikacin counts as active only for urinary infection or urinary-source bloodstream infection
            # (EUCAST systemic amikacin breakpoints are bracketed; CLSI P. aeruginosa amikacin is urinary-only).
            "amikacin_urinary_only": 0.0}


def default_policy(kind: int = FIXED, tau: int = 99, c: float = 0.5, belief: dict | None = None) -> dict:
    b = belief or {"phi": 0.2, "s0": 0.0, "kappa": 0.6, "lam": 0.3, "sigma": 0.5, "sigma_v": 0.7,
                   "mu0": 0.2, "var0": 1.3, "sev_obs_sd": 0.5}
    return {"kind": kind, "tau": tau, "c": c, **{f"b_{k}": v for k, v in b.items()}}


def _arrays(world, obs=None, e=None, directed=None):
    """Per-patient inputs the environment needs (truth for dynamics; observed for policies).

    Activity is binary (S or I counts as active, high dose) unless e["graded_activity"] is 1. `directed` fixes
    the drug given from the result day (an index into DRUGS); by default the patient gets an active drug.
    """
    sir = world["sir"]
    active = sir <= 1                                  # binary (v1): boolean arrays, as callers expect
    if e is not None and float(e["graded_activity"]) > 0.5:
        active = expit(e["pta_slope"] * (world["bp_s"] + e["pta_mid"] - world["log2mic"]))
        active = jnp.where(sir == 3, 0.0, active)      # fractional activity in [0, 1]
    if e is not None and float(e["amikacin_urinary_only"]) > 0.5:
        urinary = ~jnp.asarray(world["syndrome_bsi"]) | jnp.asarray(world["bsi_urinary"])
        active = jnp.asarray(active)
        active = active.at[:, 4].set(jnp.where(urinary, active[:, 4], jnp.zeros_like(active[:, 4])))
    if e is not None and float(e["tzp_oxa1_eff"]) < 1.0 and "oxa1" in world:
        active = jnp.asarray(active, jnp.float32)
        active = active.at[:, 1].multiply(jnp.where(jnp.asarray(world["oxa1"]), e["tzp_oxa1_eff"], 1.0))
    if e is not None and float(e["tzp_esbl_eff"]) < 1.0:
        tzp = 1                                        # DRUGS index of piperacillin-tazobactam
        active = jnp.asarray(active, jnp.float32)
        active = active.at[:, tzp].multiply(jnp.where(jnp.asarray(world["mech"])[:, 0], e["tzp_esbl_eff"], 1.0))
    hai = world["onset_hai"]
    pseudo = world["icu"] | world["immunosuppressed"]
    drug_a = jnp.where(hai, 1, 0)
    drug_b = jnp.where(pseudo, 3, 2)
    rows = jnp.arange(sir.shape[0])
    a_A = active[rows, drug_a]
    a_B = active[rows, drug_b]
    a_dir = active.max(axis=1) if directed is None else active[:, directed]
    prior_r = (obs["prior_r_obs"] if obs is not None else world["prior_r"])[:, 0]
    return dict(pid=world["pid"], bact=world["bacterial"], a_A=a_A, a_B=a_B, a_dir=a_dir,
                S0=world["severity"], sev_obs=world["severity_obs"], prior_r=prior_r)


def _episode(u, x, prior, e, pol):
    """Simulate one patient for H days. u: (H, 6) uniforms; columns: 0 severity noise,
    1 vitals noise, 2 clinician noise, 3 measurement, 4 result day (row 0 only), 5 death."""
    eps = ndtri(u[:, 0])
    xi = ndtri(u[:, 1])
    ur = u[0, 4]
    p1, p3, p4 = e["p_result_day1"], e["p_result_day3"], e["p_result_day4"]
    result_day = jnp.where(ur < p1, 1, jnp.where(ur < p1 + p3, 3, jnp.where(ur < p1 + p3 + p4, 4, 2)))
    bact = x["bact"]
    kind = pol["kind"]
    # Belief filter initial state from the observed presenting severity.
    v0, sdo = pol["b_var0"], pol["b_sev_obs_sd"]
    k0 = v0 / (v0 + sdo ** 2)
    m_init = pol["b_mu0"] + k0 * (x["sev_obs"] - pol["b_mu0"])
    P_init = (1 - k0) * v0
    oracle_step = bact & (x["a_A"] < 0.5) & (x["a_B"] > 0.5)

    def decide(t, stepped, logodds, v_meas, measured, clin_noise):
        open_ = (~stepped) & (t < result_day)
        fixed = t >= pol["tau"]
        prior_rule = (t == 0) & (prior > pol["c"])
        belief = expit(logodds) > pol["c"]
        oracle = (t == 0) & oracle_step
        clin = ((t == 0) & x["prior_r"]) | (measured & (v_meas + clin_noise > pol["c"]))
        go = jnp.select([kind == FIXED, kind == PRIOR, kind == BELIEF, kind == ORACLE, kind == CLINICIAN],
                        [fixed, prior_rule, belief, oracle, clin], False)
        return open_ & go

    def step(carry, t):
        S, alive, stepped, step_day, det, m0_, P0_, m1_, P1_, ll, stats = carry
        # Morning observation (vitals) and belief update (from day 1; day 0 uses presenting severity).
        v = S + e["sigma_v"] * xi[t]
        measured = (t > 0) & (u[t, 3] < expit(e["m0"] + e["m1"] * S))
        b = pol
        def kf(m, P, d):
            mp = (1 - b["b_phi"]) * m + b["b_phi"] * b["b_s0"] + d
            Pp = (1 - b["b_phi"]) ** 2 * P + b["b_sigma"] ** 2
            Sv = Pp + b["b_sigma_v"] ** 2
            K = Pp / Sv
            return (jnp.where(measured, mp + K * (v - mp), mp), jnp.where(measured, (1 - K) * Pp, Pp),
                    jnp.where(measured, norm.logpdf(v, mp, jnp.sqrt(Sv)), 0.0))
        m1n, P1n, l1 = kf(m1_, P1_, b["b_kappa"])      # H1: A inactive -> severity drifts up
        m0n, P0n, l0 = kf(m0_, P0_, -b["b_lam"])       # H0: A active (or not needed)
        upd = t > 0
        m1_, P1_, m0_, P0_ = (jnp.where(upd, m1n, m1_), jnp.where(upd, P1n, P1_),
                              jnp.where(upd, m0n, m0_), jnp.where(upd, P0n, P0_))
        ll = ll + jnp.where(upd, l1 - l0, 0.0)
        logodds = logit(jnp.clip(prior, 1e-6, 1 - 1e-6)) + ll
        clin_noise = 0.5 * ndtri(u[t, 2])
        go = alive & decide(t, stepped, logodds, v, measured | (t == 0), clin_noise)
        step_day = jnp.where(go, t, step_day)
        stepped = stepped | go
        # Which drug, and is it active?
        pre = t < result_day
        a = jnp.where(pre, jnp.where(stepped, x["a_B"], x["a_A"]), x["a_dir"])
        a = a.astype(jnp.float32)
        on_b_pre = pre & stepped
        # Death today (from today's severity), then severity moves.
        p_die = expit(e["theta0"] + e["theta1"] * S)
        dies = alive & (u[t, 5] < p_die)
        newS = S - e["phi"] * (S - e["s0"]) + jnp.where(bact, e["kappa"] * (1 - a) - e["lam"] * a, -e["lam_nb"]) \
            + e["sigma"] * eps[t]
        det = det | (alive & (S > e["s_det"]))
        inactive_pre = alive & pre & bact & (a < 0.5)
        stats = stats + jnp.stack([(alive & on_b_pre).astype(jnp.float32), inactive_pre.astype(jnp.float32)])
        alive = alive & ~dies
        return (newS, alive, stepped, step_day, det, m0_, P0_, m1_, P1_, ll, stats), (dies, a, S)

    init = (x["S0"], jnp.array(True), jnp.array(False), jnp.array(-1), jnp.array(False),
            m_init, P_init, m_init, P_init, jnp.array(0.0), jnp.zeros(2))
    (S, alive, stepped, step_day, det, *_r, stats), (dies, acts, s_path) = jax.lax.scan(step, init, jnp.arange(H))
    died = ~alive
    death_day = jnp.where(died, jnp.argmax(dies), -1)
    t_active = jnp.where(bact, jnp.argmax(acts > 0.5), 0)
    unnecessary = stepped & ((~bact) | (x["a_A"] > 0.5))
    return dict(died=died, death_day=death_day, deteriorated=det, step_day=step_day, stepped=stepped,
                b_days=stats[0], inactive_days=stats[1], unnecessary_b=unnecessary, time_to_active=t_active,
                result_day=result_day, severity_path=s_path)


@jax.jit
def _run(env_key, x, prior, e, pol):
    def one(pid, xi_, pr):
        u = jax.random.uniform(jax.random.fold_in(env_key, pid), (H, 6), minval=1e-7, maxval=1 - 1e-7)
        return _episode(u, xi_, pr, e, pol)
    return jax.vmap(one)(x["pid"], {k: v for k, v in x.items()}, prior)


def run_policy(seed: int, world: dict, policy: dict, env_params: dict | None = None, prior=None, obs=None,
               directed=None) -> dict:
    """Simulate every patient under `policy`. Same seed + world => same patients, same noise.
    `directed` fixes the drug from the result day (index into DRUGS; default: an active drug)."""
    e = {k: jnp.asarray(v, jnp.float32) for k, v in {**default_env_params(), **(env_params or {})}.items()}
    pol = {k: jnp.asarray(v, jnp.float32 if k not in ("kind", "tau") else jnp.int32) for k, v in policy.items()}
    x = _arrays(world, obs, e, directed)
    pr = jnp.zeros(x["pid"].shape[0]) if prior is None else jnp.asarray(prior, jnp.float32)
    return _run(jax.random.fold_in(jax.random.key(seed), 424242), x, pr, e, pol)
