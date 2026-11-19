"""Parameters for the JAX simulator (ETTB v1). Plain dicts of floats so they are
JAX pytrees: they can be jit-compiled, vmapped over (for calibration) and overridden.

The defaults are placeholders chosen to be plausible, not calibrated estimates. calibrated_site() loads a
calibrated parameter set instead.
EVIDENCE records the few values that come from published sources.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.special import ndtri

ORGANISMS = ("e_coli", "k_pneumoniae", "p_aeruginosa")          # index 3 = none
MECHANISMS = ("ESBL", "CARB", "FQR", "AGR")
DRUGS = ("ceftriaxone", "piperacillin_tazobactam", "ertapenem", "meropenem", "amikacin", "ciprofloxacin")
ABX_CLASSES = ("cephalosporin", "bl_bli", "carbapenem", "fluoroquinolone", "aminoglycoside", "other")
BL_FQ = (0, 1, 2, 3)                      # classes counted in "beta-lactam or fluoroquinolone courses"
SELECTING_CLASS = (0, 2, 3, 4)            # class selecting each mechanism (ESBL<-ceph, CARB<-carb, FQR<-FQ, AGR<-AG)
SIR = ("S", "I", "R", "NA")
INTRINSIC = 10.0
MAX_EPISODES = 6
MAX_ADMISSIONS = 3


def default_params() -> dict:
    return {
        # --- context (site) ---
        "rho_esbl": 0.06, "rho_carb": 0.004, "rho_fqr": 0.10, "rho_agr": 0.03,
        "p_hai": 0.30, "hiv_prevalence": 0.05, "breakpoints_clsi": 0.0,
        # --- demography and comorbidity ---
        "dm_base": -1.9, "dm_age": 0.5, "ckd_base": -2.6, "ckd_age": 0.8, "ckd_dm": 1.2,
        "immuno_base": -2.6, "hiv_peak_age": 42.0, "hiv_age_sd": 16.0, "hiv_shape_mean": 0.343,
        # --- healthcare contact ---
        "adm_base": -1.6, "adm_age": 0.3, "adm_comorb": 0.35, "adm_hai": 0.9,      # log-rate, admissions in 12 mo
        "p_adm_within_90d": 0.25,
        "care_base": -3.2, "care_age": 1.0, "care_comorb": 0.3,
        "transfer_base": -3.0, "transfer_hai": 1.0, "transfer_care": 1.5,
        "cath_base": -3.0, "cath_age": 0.4, "cath_care": 1.2, "cath_hai": 1.0, "cath_male": 0.5, "cath_adm": 0.4,
        "ppi_base": -1.6, "ppi_age": 0.4, "ppi_comorb": 0.3,
        # --- prior infection episodes (6 months) ---
        "ep_base": 0.30, "ep_age": 0.1, "ep_comorb": 0.2, "ep_adm": 0.5, "ep_female": 0.3,
        "ep_care": 0.4, "ep_cath": 0.5,
        "ep_hosp_base": -2.0, "ep_hosp_adm": 1.6,
        "mix_comm": np.array([0.25, 0.35, 0.0, 0.25, 0.0, 0.15]),
        "mix_hosp": np.array([0.32, 0.30, 0.14, 0.08, 0.08, 0.08]),
        "second_course_hosp": 0.25, "ep_cult_comm": 0.25, "ep_cult_hosp": 0.7,
        # --- carriage (log-ORs) ---
        "col_select": 1.0,          # exposure to the selecting class
        "col_courses": 0.35,        # per beta-lactam/FQ course in last 3 months (any class)
        "col_fq_esbl": 0.6,         # co-selection of ESBL by fluoroquinolones
        "col_adm": 0.45,            # per admission in last 12 months
        "col_hai": 0.5, "col_care": 0.6, "col_cath": 0.7, "col_ppi": 0.4, "col_age": 0.25, "col_age_step": 0.0,
        "col_transfer": 0.6, "col_ckd": 0.3,
        "co_carb": 1.5, "co_fqr": 2.0, "co_agr": 1.5,
        # v1.1: loading of each mechanism (ESBL, CARB, FQR, AGR) on a latent exposure the record never shows
        # (community, household, travel, care elsewhere). Zero reproduces v1.
        "comm_sd": np.array([0.0, 0.0, 0.0, 0.0]),
        # Extra age slope per mechanism (ESBL, CARB, FQR, AGR), per 17 years, added to the shared col_age.
        "col_age_mech": np.array([0.0, 0.0, 0.0, 0.0]),
        "prior_reveal": 0.6, "prior_false_pos": 0.02,
        # --- syndrome, severity, ICU ---
        "p_bsi_ca": 0.35, "p_bsi_hai": 0.45, "bsi_male": 0.6, "bsi_cath": -0.5,
        "sev_bsi": 0.5, "sev_uti": -0.3, "sev_comorb": 0.25, "sev_age": 0.25,
        "icu_base": -3.0, "icu_sev": 1.2, "icu_hai": 0.5,
        # v1.1b: bloodstream-infection source and severity offsets (0 reproduces earlier versions).
        "p_bsi_urinary_ca": 0.55, "p_bsi_urinary_hai": 0.35,   # placeholders pending sourced targets
        "sev_bsi_nonurinary": 0.0, "sev_hai": 0.0,
        # v1.1c (0 reproduces earlier versions): P. aeruginosa severity offset; OXA-1 co-carriage with ESBL and its
        # TZP MIC shift (log2); P. aeruginosa non-carriage TZP resistance, its selection by prior beta-lactams
        # (log-OR) and its TZP MIC shift (log2).
        "sev_pa": 0.0, "p_bsi_urinary_pa_logor": 0.0, "p_oxa1_given_esbl": 0.0, "p_oxa1_given_esbl_kp": 0.0,
        "delta_oxa1_tzp": 0.0, "mic_tables_v11c": 0.0,
        "p_pa_tzp": 0.0, "pa_tzp_prior_bl": 0.0, "delta_pa_tzp": 0.0,
        # --- microbiology ---
        "p_bact_bsi": 0.65, "p_bact_uti": 0.85,
        "org_probs": np.array([[0.85, 0.10, 0.05], [0.62, 0.23, 0.15],     # UTI-CA, UTI-HAI
                               [0.70, 0.22, 0.08], [0.45, 0.35, 0.20]]),   # BSI-CA, BSI-HAI
        "tau": np.array([0.7, 0.7, 0.6, 0.5]),
        "psi": np.array([0.02, 0.002, 0.05, 0.02]),
        "psi_pa_carb": 0.08, "esbl_kp_logor": 1.0,
        "pa_carb_prior_carb": 0.0, "pa_carb_icu": 0.0,   # v1.1 log-ORs on P. aeruginosa non-carriage carbapenem R
        # v1.1: log-odds shift of each mechanism (ESBL, CARB, FQR, AGR) in the infecting strain, by organism
        # (rows: E. coli, K. pneumoniae, P. aeruginosa). Zero reproduces v1.
        "org_mech_logor": np.zeros((3, 4)),
        "mic_sd": 0.8, "bl_shared": 0.4, "mech_sd": 0.5, "esbl_tzp_sd": 1.0,
        "esbl_tzp_target_ec": 0.75, "esbl_tzp_target_kp": 0.483,
        # --- observation noise (POMDP switches; 1 = perfect) ---
        "record_completeness": 1.0,     # P(a true prior isolate / exposure appears in the record)
        "comorb_coding": 1.0,           # P(a true comorbidity is coded)
        "ast_vme": 0.0,                 # P(resistant reported susceptible)
        "ast_me": 0.0,                  # P(susceptible reported resistant)
        "sev_obs_sd": 0.5,
    }


EVIDENCE = {
    "esbl_tzp_target_ec/kp": "SAASP 2026 p.68 (GSH 2015), literature-associational",
    "carriage and contact log-ORs": "placeholders; calibrated sets fit them to published ESBL risk-factor adjusted ORs",
    "breakpoints": "v1 tables unverified; the v1.1c tables (mic_tables_v11c) were checked against EUCAST v16.1 and CLSI",
    "everything else": "toy placeholder",
}

# ----------------------------------------------------------------------------------------------
# Microbiology tables (arrays indexed [organism, ...]); identical numbers to the NumPy reference.
WT = np.array([[-4, 1, -5, -5, 1, -5],
               [-4, 2, -5, -5, 0, -4],
               [INTRINSIC, 2, INTRINSIC, 0, 1, -2]], dtype=float)
INTRINSIC_R = np.array([[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [1, 0, 1, 0, 0, 0]], dtype=bool)
DELTA = np.zeros((3, 4, 6))
for o in (0, 1):
    DELTA[o, 0, [0, 2]] = [8, 1]                 # ESBL: ceftriaxone, ertapenem (TZP set by calibration)
    DELTA[o, 1, [0, 1, 2, 3]] = [7, 7, 7, 8.5]   # carbapenemase
    DELTA[o, 2, 5] = 6                           # FQ resistance
    DELTA[o, 3, 4] = 5                           # aminoglycoside resistance
DELTA[2, 1, 3], DELTA[2, 2, 5], DELTA[2, 3, 4] = 3.5, 4, 4
COMPAT = np.array([[1, 1, 1, 1], [1, 1, 1, 1], [0, 1, 1, 1]], dtype=bool)
BETA_LACTAM = np.array([1, 1, 1, 1, 0, 0], dtype=bool)
# Breakpoints (log2 mg/L): [system(EUCAST, CLSI), organism, drug, (S<=, R>)]. UNVERIFIED.
_BP = {
    "EUCAST": {"ent": [(1, 2), (8, 8), (0.5, 0.5), (2, 8), (8, 8), (0.25, 0.5)],
               "pa": [(1, 1), (0.001, 16), (1, 1), (2, 8), (16, 16), (0.001, 0.5)]},
    "CLSI": {"ent": [(1, 2), (8, 16), (0.5, 1), (1, 2), (4, 8), (0.25, 0.5)],
             "pa": [(1, 1), (16, 64), (1, 1), (2, 4), (16, 32), (0.5, 1)]},
}
BP = np.zeros((2, 3, 6, 2))
for si, sysname in enumerate(("EUCAST", "CLSI")):
    for o, grp in enumerate(("ent", "ent", "pa")):
        BP[si, o] = np.log2(np.array(_BP[sysname][grp], dtype=float))


# v1.1c verified tables (switch params["mic_tables_v11c"]; 0 keeps the v1 tables above).
# Wild-type log2 MIC is drawn as a continuous normal and reported as 2^ceil(x), so the table holds the fitted
# continuous MEAN, not the modal MIC (fitted by interval-censored normal to EUCAST MIC distributions,
# mic.eucast.org, queried 2026-11-19). Columns: ceftriaxone, TZP,
# ertapenem, meropenem, amikacin, ciprofloxacin. K. pneumoniae ceftriaxone (n = 46) and P. aeruginosa meropenem
# SD are inferences.
WT_V11C = np.array([[-5.40, 0.20, -7.25, -6.20, 0.62, -6.67],
                    [-5.50, 0.80, -6.70, -4.80, 0.08, -5.40],
                    [INTRINSIC, 1.85, INTRINSIC, -1.60, 1.30, -3.12]], dtype=float)
SD_V11C = np.array([[0.78, 0.90, 0.54, 0.62, 0.80, 0.68],
                    [0.80, 0.94, 0.54, 0.66, 0.74, 0.88],
                    [1.00, 1.18, 1.00, 1.20, 1.06, 1.00]], dtype=float)
# Mechanism shifts under the v1.1c tables: a mechanism places the RESISTANT population where v1 placed it, so the
# shift absorbs the change in the wild-type mean (otherwise lowering the wild-type means would also lower the
# resistant populations, e.g. carbapenemase producers testing ertapenem-intermediate).
DELTA_V11C = DELTA.copy()
for _o in range(3):
    for _d in range(6):
        if not INTRINSIC_R[_o, _d]:
            DELTA_V11C[_o, :, _d] = np.where(DELTA[_o, :, _d] != 0, DELTA[_o, :, _d] + WT[_o, _d] - WT_V11C[_o, _d], 0.0)
# Breakpoints as in BP, with CLSI P. aeruginosa piperacillin-tazobactam corrected to S <= 16, R > 32 (CLSI 2023;
# FDA STIC rationale page; the v1 table used the 2012-2022 resistant cut-off). EUCAST v16.1 rows verified unchanged.
BP_V11C = BP.copy()
BP_V11C[1, 2, 1] = np.log2([16.0, 32.0])


def esbl_tzp_shift(target_s: float, org: int, mic_sd: float, esbl_tzp_sd: float) -> float:
    """ESBL shift on log2 TZP MIC giving P(S | ESBL only) = target under EUCAST breakpoints."""
    return float(BP[0, org, 1, 0] - WT[org, 1] - ndtri(target_s) * np.hypot(mic_sd, esbl_tzp_sd))


CALIBRATED_US_SITE = Path(__file__).with_name("calibrated_us_site.json")


def calibrated_site(theta1: float = 0.9, path=CALIBRATED_US_SITE) -> tuple[dict, dict]:
    """(world params, env params) for one outcome setting of a calibrated site.

    The default file is the v1.1d US-like site. The harm of an ineffective first drug is not identified, so the file
    holds three outcome settings (theta1 = 0.6, 0.9, 1.2) that pass every check; report results across all three.
    Pass world params to sample_world and observe, and env params to run_policy or run_loop.
    """
    rep = json.loads(Path(path).read_text())
    base = {k: np.array(v) if isinstance(v, list) else v for k, v in rep["calibrated_params"].items()}
    member = next((f for f in rep["outcome_family"] if abs(f["theta1"] - theta1) < 1e-9), None)
    if member is None:
        raise ValueError(f"theta1 must be one of {[f['theta1'] for f in rep['outcome_family']]}, got {theta1}")
    return dict(base, **member["world"]), dict(member["env"])
