"""Predictors trained on real records (ARMD), loaded with NumPy, and applied to simulated patients.

The weights are in a GitHub release named in models/release.json (see models/README.md); `fetch()` downloads
and checks them.
Every model predicts P(susceptible) of the first isolate in a culture order to one drug, in one cell:
setting (inpatient or outpatient) x specimen (urine or blood) x drug.

    import armd_models as AM
    root = AM.fetch()                                       # once
    m = AM.load("logistic", "inpatient_urine_ceftriaxone")  # or mlp_ensemble, bayes_logistic, xgboost
    X = AM.shared_features(world, obs, "ceftriaxone")       # simulator record -> ARMD feature columns
    p_susceptible = m.predict(X)

Logistic, MLP ensemble and Bayesian models need only NumPy. XGBoost models need the `xgboost` package, which
the simulator does not otherwise require. "shared" models use only features the simulator also records; "full"
models need ARMD's own feature tables.
"""
from __future__ import annotations

import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
RELEASE = json.loads((MODELS / "release.json").read_text())
FAMILIES = ("logistic", "mlp_ensemble", "bayes_logistic", "xgboost")

# ARMD age bands 18-24 (reference), 25-34, ..., 85-89, 90+.
AGE_EDGES = [25, 35, 45, 55, 65, 75, 85, 90]
PRIOR_R = ("ceftriaxone", "meropenem", "ciprofloxacin", "amikacin")
SAME_R = {"ceftriaxone": "ceftriaxone", "ertapenem": "meropenem", "meropenem": "meropenem",
          "ciprofloxacin": "ciprofloxacin", "amikacin": "amikacin"}
ABX = ("cephalosporin", "bl_bli", "carbapenem", "fluoroquinolone", "aminoglycoside")   # ettb ABX_CLASSES[:5]
COMORB = ("diabetes_obs", "ckd_obs", "hiv_obs", "immunosuppressed")


# ---------------------------------------------------------------------------------------------- download
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(dest: Path | str = MODELS / RELEASE["name"]) -> Path:
    """Download the release tarball (if not already here), check its checksum and every file's, and unpack."""
    dest = Path(dest)
    if (dest / "manifest.json").exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tar = dest.parent / RELEASE["asset"]
    if not tar.exists() or _sha256(tar) != RELEASE["sha256"]:
        urllib.request.urlretrieve(RELEASE["url"], tar)
    if _sha256(tar) != RELEASE["sha256"]:
        raise ValueError(f"{tar}: checksum does not match models/release.json")
    with tarfile.open(tar) as t:
        t.extractall(dest.parent, filter="data")
    manifest = json.loads((dest / "manifest.json").read_text())
    bad = [e["file"] for e in manifest["models"] if _sha256(dest / e["file"]) != e["sha256"]]
    if bad:
        raise ValueError(f"checksum mismatch: {bad}")
    return dest


def manifest(root: Path | str = MODELS / RELEASE["name"]) -> dict:
    return json.loads((Path(root) / "manifest.json").read_text())


def test_metrics(root: Path | str = MODELS) -> pd.DataFrame:
    """Test-split statistics, one row per model: point estimates and 95% patient-bootstrap intervals (`_lo`,
    `_hi`). The same table ships in this folder and in the release."""
    root = Path(root)
    return pd.read_csv(root / "test_metrics.csv")


# ---------------------------------------------------------------------------------------------- models
class _Prep:
    """Median imputation with missing indicators, log1p of heavy-tailed non-negative columns, standardisation."""

    def __init__(self, a):
        self.columns = [str(c) for c in a["prep_columns"]]
        self.median, self.missing, self.log = a["prep_median"], a["prep_missing"], a["prep_log"]
        self.mean, self.sd = a["prep_mean"], a["prep_sd"]

    def __call__(self, X: pd.DataFrame) -> np.ndarray:
        A = X[self.columns].to_numpy(np.float32)
        nan = np.isnan(A)
        ind = nan[:, self.missing].astype(np.float32)
        A = np.where(nan, self.median, A)
        A[:, self.log] = np.log1p(np.maximum(A[:, self.log], 0))
        A = (A - self.mean) / self.sd
        return np.concatenate([A, ind], axis=1) if len(self.missing) else A


class Logistic:
    def __init__(self, a):
        self.prep, self.w = _Prep(a), a["w"]
        self.columns = self.prep.columns

    def predict(self, X):
        return expit(self.prep(X) @ self.w[1:] + self.w[0])


class MLPEnsemble:
    """Deep ensemble: each member is Linear-ReLU-Linear-ReLU-Linear-sigmoid. Draws are the members."""

    def __init__(self, a):
        self.prep = _Prep(a)
        self.columns = self.prep.columns
        self.members = []
        while f"member{len(self.members)}_W0" in a:
            i = len(self.members)
            self.members.append([(a[f"member{i}_W{j}"], a[f"member{i}_b{j}"]) for j in range(3)])

    def predict_draws(self, X):
        A = self.prep(X)
        out = []
        for layers in self.members:
            h = A
            for j, (W, b) in enumerate(layers):
                h = h @ W.T + b
                if j < len(layers) - 1:
                    h = np.maximum(h, 0)
            out.append(expit(h[:, 0]))
        return np.stack(out)

    def predict(self, X):
        return self.predict_draws(X).mean(0)


class BayesLogistic:
    """Hierarchical logistic regression with site intercepts; 200 posterior draws. A site not seen in training
    (such as simulated patients) gets intercepts drawn from the fitted population of sites."""

    def __init__(self, a):
        self.prep = _Prep(a)
        self.columns = self.prep.columns
        self.beta, self.mu_a, self.tau_a = a["beta"], a["mu_a"], a["tau_a"]
        self.intercept_site, self.sites = a["intercept_site"], [str(s) for s in a["sites"]]

    def intercept(self, site=None, seed=0):
        if site in self.sites:
            return self.intercept_site[:, self.sites.index(site)]
        return self.mu_a + self.tau_a * np.random.default_rng(seed).normal(size=len(self.mu_a))

    def predict_draws(self, X, site=None):
        return expit(self.intercept(site)[:, None] + self.beta @ self.prep(X).T)

    def predict(self, X, site=None):
        return self.predict_draws(X, site).mean(0)


class XGBoost:
    def __init__(self, path):
        import xgboost as xgb
        self.booster = xgb.Booster()
        self.booster.load_model(str(path))
        self.columns = list(self.booster.feature_names)

    def predict(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X[self.columns].to_numpy(np.float32), feature_names=self.columns)
        return self.booster.predict(d).astype(float)


def load(family: str, cell: str, features: str = "shared", root: Path | str = MODELS / RELEASE["name"]):
    """A model from the unpacked release. `cell` is e.g. "inpatient_urine_ceftriaxone"."""
    root = Path(root)
    if family == "xgboost":
        return XGBoost(root / "xgboost" / features / f"{cell}.json")
    cls = {"logistic": Logistic, "mlp_ensemble": MLPEnsemble, "bayes_logistic": BayesLogistic}[family]
    with np.load(root / family / features / f"{cell}.npz", allow_pickle=False) as a:
        return cls({k: a[k] for k in a.files})


# ---------------------------------------------------------------------------------------------- simulator -> ARMD
def shared_features(world, obs, drug: str) -> pd.DataFrame:
    """The simulator's observed record as ARMD's shared feature columns, for predicting susceptibility to `drug`.
    Every simulated patient is admitted, so care_ward is 1 unless in ICU."""
    w = {k: np.asarray(v) for k, v in world.items()}
    o = {k: np.asarray(v) for k, v in obs.items()}
    X = {}
    band = np.digitize(w["age"], AGE_EDGES)
    for b in range(1, len(AGE_EDGES) + 1):
        X[f"age_band_{b}"] = band == b
    X["female"] = w["female"]
    X["care_icu"] = w["icu"]
    X["care_ward"] = ~w["icu"].astype(bool)
    pr = o["prior_r_obs"].astype(bool)
    X["prior_r_any"], X["prior_r_count"] = pr.any(1), pr.sum(1)
    X["prior_r_same"] = pr[:, PRIOR_R.index(SAME_R[drug])] if drug in SAME_R else np.zeros(len(pr), bool)
    X["prior_positive_culture"] = w["has_prior_culture"]
    abx = o["prior_abx_obs"][:, :len(ABX)].astype(bool)
    for j, c in enumerate(ABX):
        X[f"prior_abx_{c}"] = abx[:, j]
    X["prior_abx_any"] = abx.any(1)
    X["prior_abx_within_90d"] = abx.any(1) & (w["n_courses_3m"] > 0)
    X["prior_abx_courses_90d"] = w["n_courses_3m"] * abx.any(1)
    X["procedure_urinary_catheter"] = w["urinary_catheter"]
    X["nursing_home"] = w["care_home"]
    comorb = {"diabetes_obs": o["diabetes_obs"], "ckd_obs": o["ckd_obs"], "hiv_obs": o["hiv_obs"],
              "immunosuppressed": w["immunosuppressed"]}
    for k in COMORB:
        X[f"comorb_{k}"] = comorb[k]
    X["comorb_count"] = sum(np.asarray(comorb[k], int) for k in COMORB)
    return pd.DataFrame(X).astype(float)


def p_drug_a_fails(world, obs, family: str = "logistic", root: Path | str = MODELS / RELEASE["name"]) -> np.ndarray:
    """P(drug A tests resistant) for each simulated patient from the ARMD-trained model for their cell: urine or
    blood (bloodstream infection), ceftriaxone (community onset) or piperacillin-tazobactam (hospital onset)."""
    w = {k: np.asarray(v) for k, v in world.items()}
    spec = np.where(w["syndrome_bsi"], "blood", "urine")
    drug = np.where(w["onset_hai"], "piperacillin_tazobactam", "ceftriaxone")
    out = np.full(len(spec), np.nan)
    for s in ("urine", "blood"):
        for d in ("ceftriaxone", "piperacillin_tazobactam"):
            k = (spec == s) & (drug == d)
            if k.any():
                m = load(family, f"inpatient_{s}_{d}", "shared", root)
                X = shared_features(world, obs, d)[k]
                out[k] = 1 - m.predict(X)
    return out
