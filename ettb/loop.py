"""Day-by-day interface to the step-up process, for decision models written outside JAX.

`env.run_policy` runs one of the built-in rules inside a single compiled loop. `run_loop` runs the same dynamics one
day at a time and asks a decision model, which can be any Python object, whether to step up each morning. The noise
is drawn exactly as in `run_policy` (keyed by seed, patient and day, never by the action), so the same seed and world
give the same patients under any decision model, and a rule run through either function gives the same outcomes.

A decision model needs two methods:

    reset(first_look)  called once before day 0. first_look["presenting_severity"] is the noisy severity reading
                       at presentation, one value per patient.
    act(day) -> bool   called each morning. Returns, per patient, whether to step up to drug B today.

`day` holds what a clinician could see that morning, as numpy arrays over patients:

    t              the day (0 to 29)
    vitals         the vital-sign composite v_t = S_t + noise (NaN on day 0 and when not measured)
    can_step       still alive, still on drug A and the culture result not yet back
    result_back    the culture result has arrived, so therapy is directed from today
    alive          alive this morning

A request to step up is ignored where can_step is False. The model never sees severity, the organism or whether
either drug works.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import expit, ndtri

from . import env as E

H = E.H


def _noise(seed: int, pid):
    """The (patients, H, 6) uniforms of env._run: same key, same patient, same day."""
    key = jax.random.fold_in(jax.random.key(seed), 424242)
    return jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(key, i), (H, 6), minval=1e-7,
                                                 maxval=1 - 1e-7))(pid)


@jax.jit
def _morning(S, t, u_t, e):
    """Vital signs this morning and whether they were measured (as in env._episode)."""
    v = S + e["sigma_v"] * ndtri(u_t[:, 1])
    measured = (t > 0) & (u_t[:, 3] < expit(e["m0"] + e["m1"] * S))
    return v, measured


@jax.jit
def _day(state, t, go, u_t, x, e, result_day):
    """One day after the decision: drug activity, death, then the severity transition (as in env._episode)."""
    S, alive, stepped, step_day, det, b_days, inactive_days = state
    go = alive & go & (~stepped) & (t < result_day)
    step_day = jnp.where(go, t, step_day)
    stepped = stepped | go
    pre = t < result_day
    a = jnp.where(pre, jnp.where(stepped, x["a_B"], x["a_A"]), x["a_dir"]).astype(jnp.float32)
    on_b_pre = pre & stepped
    p_die = expit(e["theta0"] + e["theta1"] * S)
    dies = alive & (u_t[:, 5] < p_die)
    newS = S - e["phi"] * (S - e["s0"]) + jnp.where(x["bact"], e["kappa"] * (1 - a) - e["lam"] * a, -e["lam_nb"]) \
        + e["sigma"] * ndtri(u_t[:, 0])
    det = det | (alive & (S > e["s_det"]))
    b_days = b_days + (alive & on_b_pre).astype(jnp.float32)
    inactive_days = inactive_days + (alive & pre & x["bact"] & (a < 0.5)).astype(jnp.float32)
    alive = alive & ~dies
    return (newS, alive, stepped, step_day, det, b_days, inactive_days), dies, a


def run_loop(seed: int, world: dict, model, env_params: dict | None = None, obs=None, directed=None) -> dict:
    """Simulate every patient with `model` deciding each morning. Returns the arrays `run_policy` returns, as numpy,
    plus the daily `vitals` (NaN when not seen), `step_up` requests and drug activity `alpha`, each (patients, H)."""
    e = {k: jnp.asarray(v, jnp.float32) for k, v in {**E.default_env_params(), **(env_params or {})}.items()}
    x = E._arrays(world, obs, e, directed)
    n = x["pid"].shape[0]
    u = _noise(seed, x["pid"])
    ur = u[:, 0, 4]
    p1, p3, p4 = e["p_result_day1"], e["p_result_day3"], e["p_result_day4"]
    result_day = jnp.where(ur < p1, 1, jnp.where(ur < p1 + p3, 3, jnp.where(ur < p1 + p3 + p4, 4, 2)))
    state = (jnp.asarray(x["S0"], jnp.float32), jnp.ones(n, bool), jnp.zeros(n, bool), jnp.full(n, -1),
             jnp.zeros(n, bool), jnp.zeros(n, jnp.float32), jnp.zeros(n, jnp.float32))
    model.reset({"presenting_severity": np.asarray(x["sev_obs"])})
    vitals, requests, alpha, dies_all, s_path = (np.full((n, H), np.nan, np.float32), np.zeros((n, H), bool),
                                                 np.zeros((n, H), np.float32), np.zeros((n, H), bool),
                                                 np.zeros((n, H), np.float32))
    rd = np.asarray(result_day)
    for t in range(H):
        u_t = u[:, t]
        v, measured = _morning(state[0], t, u_t, e)
        alive, stepped = np.asarray(state[1]), np.asarray(state[2])
        seen = np.asarray(measured) & alive
        vitals[:, t] = np.where(seen, np.asarray(v), np.nan)
        day = {"t": t, "vitals": vitals[:, t].copy(), "can_step": alive & ~stepped & (t < rd),
               "result_back": t >= rd, "alive": alive}
        go = np.asarray(model.act(day), bool) & day["can_step"]
        requests[:, t] = go
        s_path[:, t] = np.asarray(state[0])
        state, dies, a = _day(state, t, jnp.asarray(go), u_t, x, e, result_day)
        dies_all[:, t], alpha[:, t] = np.asarray(dies), np.asarray(a)
    S, alive, stepped, step_day, det, b_days, inactive_days = (np.asarray(s) for s in state)
    bact, a_A = np.asarray(x["bact"]), np.asarray(x["a_A"])
    died = ~alive
    return dict(died=died, death_day=np.where(died, dies_all.argmax(1), -1), deteriorated=det, step_day=step_day,
                stepped=stepped, b_days=b_days, inactive_days=inactive_days,
                unnecessary_b=stepped & (~bact | (a_A > 0.5)),
                time_to_active=np.where(bact, (alpha > 0.5).argmax(1), 0), result_day=rd, severity_path=s_path,
                vitals=vitals, step_up=requests, alpha=alpha)
