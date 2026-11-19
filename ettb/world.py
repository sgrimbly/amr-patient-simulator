"""JAX world generator (ETTB v1): one pure function per patient, vmapped and jit-compiled.

Each patient's randomness is jax.random.fold_in(base_key, patient_id), then a fixed
stream id per variable, so a patient's draws never depend on cohort size, other
patients or the policy (common random numbers / exact replay).

Truth (`sample_world`) and the observed record (`observe`) are separate functions:
observation noise switches (record completeness, comorbidity coding, AST errors)
act only on what a model or policy can see.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.special import expit, logit, ndtri

from .params import (BETA_LACTAM, BL_FQ, BP, COMPAT, DELTA, INTRINSIC, INTRINSIC_R, MAX_ADMISSIONS,
                     MAX_EPISODES, SD_V11C, SELECTING_CLASS, WT, WT_V11C, BP_V11C, DELTA_V11C)

# Fixed slot layout: each patient draws ONE block of uniforms; every variable owns fixed slots.
# This keeps common random numbers (a variable's draw never depends on other variables or the
# policy) while needing a single counter-based PRNG call per patient.
_LAYOUT = [("age", 1), ("sex", 1), ("dm", 1), ("ckd", 1), ("immuno", 1), ("hiv", 1), ("onset", 1),
           ("adm_n", 1), ("adm_t", MAX_ADMISSIONS), ("care", 1), ("transfer", 1), ("cath", 1), ("ppi", 1),
           ("ep_n", 1), ("ep_set", MAX_EPISODES), ("ep_cls", MAX_EPISODES), ("ep_cls2", MAX_EPISODES),
           ("ep_2nd", MAX_EPISODES), ("ep_time", MAX_EPISODES), ("ep_cult", MAX_EPISODES), ("col", 4),
           ("reveal", 4), ("fp", 4), ("syn", 1), ("sev", 1), ("icu", 1), ("egfr", 1), ("bact", 1), ("org", 1),
           ("mech", 4), ("mic_shared", 1), ("mic_own", 6), ("mech_noise", 24), ("sev_obs", 1)]
SLOT, _pos = {}, 0
for _name, _size in _LAYOUT:
    SLOT[_name] = (_pos, _size)
    _pos += _size
N_UNIFORMS = _pos
# v1.1 variables draw from a separate extension stream (fold_in(patient_key, 1)), so every v1 draw is unchanged.
_EXT_LAYOUT = {"comm": 0}
N_EXT_UNIFORMS = 1
# Later variables use further streams (fold_in(patient_key, 2), ...), so earlier draws never change.
_EXT2_LAYOUT = {"bsi_source": 0}
N_EXT2_UNIFORMS = 1
_EXT3_LAYOUT = {"oxa1": 0, "pa_tzp": 1}
N_EXT3_UNIFORMS = 2
_OBS_LAYOUT = {"rec": (0, 8), "code": (8, 3), "ast": (11, 6)}
N_OBS_UNIFORMS = 17


class _U:
    def __init__(self, u):
        self.u = u

    def __call__(self, name):
        a, k = SLOT[name]
        return self.u[a] if k == 1 else self.u[a:a + k]

    def normal(self, name):
        return ndtri(self(name))


def _poisson_icdf(u, lam, kmax):
    """Inverse-CDF Poisson draw capped at kmax (exact for k < kmax)."""
    k, pk = jnp.zeros_like(u, dtype=jnp.int32), jnp.exp(-lam)
    c = pk
    for j in range(1, kmax + 1):
        k = k + (u > c)
        pk = pk * lam / j
        c = c + pk
    return k


def _categorical_icdf(u, probs):
    c = jnp.cumsum(probs / probs.sum(-1, keepdims=True), axis=-1)
    return (u[..., None] > c[..., :-1]).sum(-1)


def _patient(u, u_ext, u_ext2, u_ext3, p):
    U = _U(u)
    age = jnp.clip(62 + 17 * U.normal("age"), 18, 95)
    age_z = (age - 62) / 17
    female = (U("sex") < 0.52)
    dm = (U("dm") < expit(p["dm_base"] + p["dm_age"] * age_z))
    ckd = (U("ckd") < expit(p["ckd_base"] + p["ckd_age"] * age_z + p["ckd_dm"] * dm))
    immuno = (U("immuno") < expit(p["immuno_base"] + 0.3 * age_z))
    hiv_shape = jnp.exp(-((age - p["hiv_peak_age"]) / p["hiv_age_sd"]) ** 2)
    hiv = (U("hiv") < jnp.clip(p["hiv_prevalence"] * hiv_shape / p["hiv_shape_mean"], 0, 0.95))
    comorb = dm.astype(int) + ckd + immuno + hiv
    hai = (U("onset") < p["p_hai"])
    # Admissions in last 12 months (Poisson, capped) and whether any fell in the last 90 days.
    adm_n = _poisson_icdf(U("adm_n"), jnp.exp(
        p["adm_base"] + p["adm_age"] * age_z + p["adm_comorb"] * comorb + p["adm_hai"] * hai), MAX_ADMISSIONS)
    adm_recent = (jnp.arange(MAX_ADMISSIONS) < adm_n) & (U("adm_t") < p["p_adm_within_90d"])
    hosp_90d = adm_recent.any()
    care = (U("care") < expit(p["care_base"] + p["care_age"] * age_z + p["care_comorb"] * comorb))
    transfer = (U("transfer") < expit(p["transfer_base"] + p["transfer_hai"] * hai + p["transfer_care"] * care))
    cath = (U("cath") < expit(p["cath_base"] + p["cath_age"] * age_z + p["cath_care"] * care
                                    + p["cath_hai"] * hai + p["cath_male"] * (~female) + p["cath_adm"] * adm_n))
    ppi = (U("ppi") < expit(p["ppi_base"] + p["ppi_age"] * age_z + p["ppi_comorb"] * comorb))
    # Prior infection episodes (6 months): setting -> class -> culture; timing for the 3-month window.
    lam = jnp.exp(jnp.log(p["ep_base"]) + p["ep_age"] * age_z + p["ep_comorb"] * comorb + p["ep_adm"] * adm_n
                  + p["ep_female"] * female + p["ep_care"] * care + p["ep_cath"] * cath)
    ep_n = _poisson_icdf(U("ep_n"), lam, MAX_EPISODES)
    exists = jnp.arange(MAX_EPISODES) < ep_n
    in_hosp = exists & (U("ep_set") < expit(p["ep_hosp_base"] + p["ep_hosp_adm"] * (adm_n > 0)))
    probs = jnp.where(in_hosp[:, None], p["mix_hosp"], p["mix_comm"])
    cls1 = _categorical_icdf(U("ep_cls"), probs)
    cls2 = _categorical_icdf(U("ep_cls2"), probs)
    has2 = in_hosp & (U("ep_2nd") < p["second_course_hosp"])
    within3m = U("ep_time") < 0.5
    ncls = p["mix_comm"].shape[0]
    onehot1 = jax.nn.one_hot(cls1, ncls) * exists[:, None]
    onehot2 = jax.nn.one_hot(cls2, ncls) * has2[:, None]
    exposed = (onehot1 + onehot2).sum(0) > 0
    bl_fq = jnp.zeros(ncls).at[jnp.array(BL_FQ)].set(1.0)
    n_courses_3m = ((onehot1 + onehot2) @ bl_fq * within3m).sum()
    cultured = (exists & (U("ep_cult") < jnp.where(in_hosp, p["ep_cult_hosp"], p["ep_cult_comm"]))).any()
    # Carriage by mechanism; ESBL first, others co-travel with it.
    base = jnp.array([logit(p["rho_esbl"]), logit(p["rho_carb"]), logit(p["rho_fqr"]), logit(p["rho_agr"])])
    common = (p["col_courses"] * n_courses_3m + p["col_adm"] * adm_n + p["col_hai"] * hai + p["col_care"] * care
              + p["col_cath"] * cath + p["col_ppi"] * ppi + p["col_age"] * age_z + p["col_age_step"] * (age >= 70)
              + p["col_transfer"] * transfer
              + p["col_ckd"] * ckd)
    select = p["col_select"] * exposed[jnp.array(SELECTING_CLASS)]
    # v1.1: acquisition the record never shows (community, household, travel, care elsewhere), as a latent
    # per-patient exposure z with a loading per mechanism. comm_sd = 0 reproduces v1 exactly.
    z_comm = ndtri(u_ext[_EXT_LAYOUT["comm"]])
    unrec = p["comm_sd"] * z_comm
    # Mechanism-specific age slopes on top of the shared one (zeros reproduce earlier versions).
    age_mech = p["col_age_mech"] * age_z
    u_col = U("col")
    c_esbl = u_col[0] < expit(base[0] + common + select[0] + p["col_fq_esbl"] * exposed[3] + unrec[0] + age_mech[0])
    co = jnp.array([0.0, p["co_carb"], p["co_fqr"], p["co_agr"]])
    col = jnp.concatenate([c_esbl[None], u_col[1:] < expit(base[1:] + common + select[1:] + co[1:] * c_esbl
                                                            + unrec[1:] + age_mech[1:])])
    shown = jnp.where(col, U("reveal") < p["prior_reveal"], U("fp") < p["prior_false_pos"])
    prior_r = cultured & shown
    # Syndrome, severity, ICU.
    p_bsi = jnp.where(hai, p["p_bsi_hai"], p["p_bsi_ca"])
    bsi = (U("syn") < expit(logit(p_bsi) + p["bsi_male"] * (~female) + p["bsi_cath"] * cath))
    # v1.1b: source of a bloodstream infection (urinary or other); non-urinary source and hospital onset shift
    # presenting severity. Offsets of 0 reproduce earlier versions.
    # Organism (drawn before severity so that severity can depend on it; draws use fixed slots, so values match
    # earlier versions exactly).
    bact = (U("bact") < jnp.where(bsi, p["p_bact_bsi"], p["p_bact_uti"]))
    row = 2 * bsi + hai
    org = _categorical_icdf(U("org"), p["org_probs"][row])
    org = jnp.where(bact, org, 3)
    oi = jnp.minimum(org, 2)
    # v1.1b source of a bloodstream infection; v1.1c: P. aeruginosa has a different urinary-source share
    # (log-OR p_bsi_urinary_pa_logor; 0 reproduces v1.1b).
    p_uro = jnp.where(hai, p["p_bsi_urinary_hai"], p["p_bsi_urinary_ca"])
    p_uro = jnp.where(org == 2, expit(logit(p_uro) + p["p_bsi_urinary_pa_logor"]), p_uro)
    bsi_urinary = bsi & (u_ext2[_EXT2_LAYOUT["bsi_source"]] < p_uro)
    # v1.1c: P. aeruginosa infection is more severe at presentation (offset 0 reproduces earlier versions).
    sev = jnp.where(bsi, p["sev_bsi"], p["sev_uti"]) + p["sev_comorb"] * comorb + p["sev_age"] * age_z \
        + U.normal("sev") + p["sev_bsi_nonurinary"] * (bsi & ~bsi_urinary) + p["sev_hai"] * hai \
        + p["sev_pa"] * (org == 2)
    icu = (U("icu") < expit(p["icu_base"] + p["icu_sev"] * sev + p["icu_hai"] * hai))
    egfr_low = ckd | ((age > 75) & (U("egfr") < 0.3))
    # Microbiology.
    # P. aeruginosa carbapenem resistance outside carriage (porin loss, efflux) is selected by carbapenem exposure
    # and ICU stay in v1.1; both log-ORs default to 0, which reproduces v1's flat rate.
    psi_pa = expit(logit(p["psi_pa_carb"]) + p["pa_carb_prior_carb"] * exposed[2] + p["pa_carb_icu"] * icu)
    psi = p["psi"].at[1].set(jnp.where(org == 2, psi_pa, p["psi"][1]))
    pm = jnp.where(col, p["tau"], psi)
    odds = pm[0] / (1 - pm[0]) * jnp.exp(p["esbl_kp_logor"])
    pm = pm.at[0].set(jnp.where(org == 1, odds / (1 + odds), pm[0]))
    # v1.1: organism-specific log-odds for each mechanism appearing in the infecting strain (species differ in how
    # often they acquire and express each mechanism). Zero leaves pm untouched, so v1 is reproduced exactly.
    lor = p["org_mech_logor"][oi]
    pm_adj = jnp.exp(lor) * pm / (1 - pm + jnp.exp(lor) * pm)
    pm = jnp.where(lor == 0, pm, pm_adj)
    mech = bact & jnp.asarray(COMPAT)[oi] & (U("mech") < pm)
    v11c = p["mic_tables_v11c"] > 0.5
    wt = jnp.where(v11c, jnp.asarray(WT_V11C)[oi], jnp.asarray(WT)[oi])
    wt_sd = jnp.where(v11c, jnp.asarray(SD_V11C)[oi], p["mic_sd"])
    total_sd = jnp.hypot(wt_sd[1], p["esbl_tzp_sd"])
    tzp_shift = jnp.where(org == 1, BP[0, 1, 1, 0] - wt[1] - ndtri(p["esbl_tzp_target_kp"]) * total_sd,
                          BP[0, 0, 1, 0] - wt[1] - ndtri(p["esbl_tzp_target_ec"]) * total_sd)
    delta = jnp.where(v11c, jnp.asarray(DELTA_V11C)[oi], jnp.asarray(DELTA)[oi])
    delta = delta.at[0, 1].set(jnp.where(oi < 2, tzp_shift, 0.0))
    msd = jnp.full((4, 6), p["mech_sd"]).at[0, 1].set(p["esbl_tzp_sd"])
    z_sh = U.normal("mic_shared")
    z_own = U.normal("mic_own")
    eps = wt_sd * jnp.where(jnp.asarray(BETA_LACTAM), jnp.sqrt(p["bl_shared"]) * z_sh
                                  + jnp.sqrt(1 - p["bl_shared"]) * z_own, z_own)
    z_m = U.normal("mech_noise").reshape(4, 6)
    shift = (mech[:, None] * (delta + msd * z_m) * (delta != 0)).sum(0)
    # v1.1c: OXA-1 co-carried with an ESBL (E. coli, K. pneumoniae) raises the piperacillin-tazobactam MIC; P.
    # aeruginosa non-carriage TZP resistance (AmpC derepression, efflux) is selected by prior beta-lactams.
    # Probabilities of 0 reproduce earlier versions.
    p_oxa1 = jnp.where(org == 1, p["p_oxa1_given_esbl_kp"], p["p_oxa1_given_esbl"])
    oxa1 = mech[0] & (oi < 2) & (u_ext3[_EXT3_LAYOUT["oxa1"]] < p_oxa1)
    bl_exposed = exposed[0] | exposed[1] | exposed[2]
    p_pa_tzp = expit(logit(jnp.maximum(p["p_pa_tzp"], 1e-9)) + p["pa_tzp_prior_bl"] * bl_exposed)
    pa_tzp = bact & (org == 2) & (p["p_pa_tzp"] > 0) & (u_ext3[_EXT3_LAYOUT["pa_tzp"]] < p_pa_tzp)
    shift = shift.at[1].add(p["delta_oxa1_tzp"] * oxa1 + p["delta_pa_tzp"] * pa_tzp)
    log2mic = jnp.where(jnp.asarray(INTRINSIC_R)[oi], INTRINSIC, wt + eps + shift)
    bp_tab = jnp.where(v11c, jnp.asarray(BP_V11C), jnp.asarray(BP))
    bp = jnp.where(p["breakpoints_clsi"] > 0.5, bp_tab[1, oi], bp_tab[0, oi])
    kk = jnp.ceil(log2mic)
    sir = jnp.where(kk <= bp[:, 0], 0, jnp.where(kk > bp[:, 1], 2, 1))
    sir = jnp.where(jnp.asarray(INTRINSIC_R)[oi], 2, sir)
    sir = jnp.where(bact, sir, 3)
    sev_obs = sev + p["sev_obs_sd"] * U.normal("sev_obs")
    return dict(age=age, age_z=age_z, female=female, diabetes=dm, ckd=ckd, immunosuppressed=immuno, hiv=hiv,
                comorbidity_count=comorb, onset_hai=hai, hosp_12m_count=adm_n, hosp_90d=hosp_90d, care_home=care,
                transfer_in=transfer, urinary_catheter=cath, ppi=ppi, n_prior_episodes=ep_n,
                prior_abx=exposed[:5], n_courses_3m=n_courses_3m, has_prior_culture=cultured, col=col,
                prior_r=prior_r, syndrome_bsi=bsi, severity=sev, severity_obs=sev_obs, icu=icu, egfr_low=egfr_low,
                bacterial=bact, organism=org, mech=mech, log2mic=log2mic, sir=sir, comm_exposure=z_comm,
                bp_s=bp[:, 0], bsi_urinary=bsi_urinary, oxa1=oxa1, pa_tzp_mech=pa_tzp)


@partial(jax.jit, static_argnames=("n",))
def _sample(base_key, start_id, n, p):
    pids = start_id + jnp.arange(n)
    u = jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(base_key, i), (N_UNIFORMS,),
                                              minval=1e-7, maxval=1 - 1e-7))(pids)
    u_ext = jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(jax.random.fold_in(base_key, i), 1),
                                                  (N_EXT_UNIFORMS,), minval=1e-7, maxval=1 - 1e-7))(pids)
    u_ext2 = jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(jax.random.fold_in(base_key, i), 2),
                                                   (N_EXT2_UNIFORMS,), minval=1e-7, maxval=1 - 1e-7))(pids)
    u_ext3 = jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(jax.random.fold_in(base_key, i), 3),
                                                   (N_EXT3_UNIFORMS,), minval=1e-7, maxval=1 - 1e-7))(pids)
    out = jax.vmap(_patient, in_axes=(0, 0, 0, 0, None))(u, u_ext, u_ext2, u_ext3, p)
    out["pid"] = pids
    return out


def _complete(params: dict) -> dict:
    """Fill keys added after a parameter set was saved (e.g. v1 calibration reports) with their defaults,
    which reproduce the version that saved them."""
    from .params import default_params
    return {**default_params(), **params}


def sample_world(seed: int, n: int, params: dict, start_id: int = 0) -> dict:
    p = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in _complete(params).items()}
    return _sample(jax.random.key(seed), start_id, n, p)


def _observe_one(u, w, p):
    """Observed record for one patient: record linkage and coding noise; AST misclassification."""
    rec = u[0:8] < p["record_completeness"]
    code = u[8:11] < p["comorb_coding"]
    u_ast = u[11:17]
    flip_to_s = (w["sir"] == 2) & (u_ast < p["ast_vme"])
    flip_to_r = (w["sir"] == 0) & (u_ast < p["ast_me"])
    sir_obs = jnp.where(flip_to_s, 0, jnp.where(flip_to_r, 2, w["sir"]))
    return dict(prior_r_obs=w["prior_r"] & rec[:4], prior_abx_obs=w["prior_abx"] & rec[3:8],
                diabetes_obs=w["diabetes"] & code[0], ckd_obs=w["ckd"] & code[1], hiv_obs=w["hiv"] & code[2],
                sir_obs=sir_obs)


@jax.jit
def _observe(base_key, world, p):
    okey = jax.random.fold_in(base_key, 7_777_777)
    u = jax.vmap(lambda i: jax.random.uniform(jax.random.fold_in(okey, i), (N_OBS_UNIFORMS,)))(world["pid"])
    return jax.vmap(_observe_one, in_axes=(0, 0, None))(u, world, p)


def observe(seed: int, world: dict, params: dict) -> dict:
    p = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in _complete(params).items()}
    return _observe(jax.random.key(seed), world, p)


def to_frame(world: dict, obs: dict | None = None):
    """pandas view with the column names used by the realism audit and predictor adapters."""
    import pandas as pd

    from .params import ABX_CLASSES, DRUGS, MECHANISMS, ORGANISMS, SIR
    w = {k: np.asarray(v) for k, v in world.items()}
    d = {k: w[k] for k in ("pid", "age", "age_z", "female", "diabetes", "ckd", "immunosuppressed", "hiv",
                           "comorbidity_count", "onset_hai", "hosp_12m_count", "hosp_90d", "care_home",
                           "transfer_in", "urinary_catheter", "ppi", "n_prior_episodes", "n_courses_3m",
                           "has_prior_culture", "syndrome_bsi", "severity", "severity_obs", "icu", "egfr_low",
                           "bacterial") if k in w}
    if "bsi_urinary" in w:
        d["bsi_urinary"] = w["bsi_urinary"]
    df = pd.DataFrame(d)
    for j, c in enumerate(ABX_CLASSES[:5]):
        df[f"prior_abx_{c}"] = w["prior_abx"][:, j]
    for j, m in enumerate(MECHANISMS):
        df[f"col_{m}"] = w["col"][:, j]
        df[f"mech_{m}"] = w["mech"][:, j]
    for j, dname in enumerate(("ceftriaxone", "meropenem", "ciprofloxacin", "amikacin")):
        df[f"prior_r_{dname}"] = w["prior_r"][:, j]
    df["prior_esbl_12m"] = df["prior_r_ceftriaxone"]
    df["organism"] = np.array(list(ORGANISMS) + ["none"], dtype=object)[w["organism"]]
    for j, dname in enumerate(DRUGS):
        df[f"log2mic_{dname}"] = np.where(w["bacterial"], w["log2mic"][:, j], np.nan)
        df[f"sir_{dname}"] = np.array(SIR, dtype=object)[w["sir"][:, j]]
    # Times consistent with onset (for the realism hard rules).
    base = pd.Timestamp("2025-01-01", tz="UTC")
    df["admit_time"] = base + pd.to_timedelta((df.pid % 365).astype(int), unit="D")
    df["t0"] = df["admit_time"] + pd.to_timedelta(np.where(df.onset_hai, 72, 3), unit="h")
    df["empiric_active"] = False
    df["regimen"] = "none"
    if obs is not None:
        o = {k: np.asarray(v) for k, v in obs.items()}
        for j, dname in enumerate(("ceftriaxone", "meropenem", "ciprofloxacin", "amikacin")):
            df[f"prior_r_{dname}_obs"] = o["prior_r_obs"][:, j]
        for k in ("diabetes_obs", "ckd_obs", "hiv_obs"):
            df[k] = o[k]
        for j, dname in enumerate(DRUGS):
            df[f"sir_obs_{dname}"] = np.array(SIR, dtype=object)[o["sir_obs"][:, j]]
    return df
