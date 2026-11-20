# Predictors trained on ARMD

Trained models that predict, from a patient's record, whether a culture isolate will be susceptible to an
antibiotic. They were trained on real hospital records so that the decision examples can use a predictor that has
never seen simulated data. The weights are in a GitHub release; this folder holds the model card, the manifest
(files, columns, checksums), the test statistics (`test_metrics.csv`) and `release.json` (which release to download
and its checksum).

**Current release: [`models-v0.9`](https://github.com/sgrimbly/amr-patient-simulator/releases/tag/models-v0.9),
interim.** It holds the shared-feature models only. The XGBoost models are final. The logistic, MLP ensemble and
Bayesian models come from an earlier training run on an earlier version of the ARMD feature tables, and their
statistics are computed on the current tables, so treat them as preliminary. `models-v1` will replace all of them,
add the full-feature models, and keep the same interface; `fetch()` follows `release.json`.

```python
import sys; sys.path.insert(0, "examples")
import armd_models as AM

AM.fetch()                                                   # downloads and checks the release
m = AM.load("logistic", "inpatient_urine_ceftriaxone")       # P(susceptible)
X = AM.shared_features(world, obs, "ceftriaxone")            # simulated records as ARMD feature columns
p_resistant = 1 - m.predict(X)
p_fail = AM.p_drug_a_fails(world, obs, "mlp_ensemble")       # drug A by onset, cell by syndrome
```

Logistic, MLP ensemble and Bayesian models load with NumPy alone. XGBoost models need the `xgboost` package.
Nothing in the release is a pickle.

## What the models are

- **Data.** The Antibiotic Resistance Microbiology Dataset (ARMD), three public sites: Stanford, ECUH and UTSW,
  2.37 million culture orders. CC0. Nateghi Haredasht et al., *Sci Data* 12:1299 (2025),
  doi:10.1038/s41597-025-05649-7; Cooper et al., *Sci Data* (2026), doi:10.1038/s41597-026-07892-y; Dryad
  doi:10.5061/dryad.jq2bvq8kp, 10.5061/dryad.7sqv9s55x and 10.5061/dryad.0rxwdbsd5.
- **Task.** P(susceptible) of the first isolate in a culture order to one drug, for 13 cells: inpatient urine and
  blood, each for ceftriaxone, piperacillin-tazobactam, meropenem, ciprofloxacin, gentamicin and amikacin, and
  outpatient urine ciprofloxacin. The setup follows ARMD-Predict (Nateghi Haredasht et al. 2026,
  arXiv:2610.32998). These are our own independent retrains; the ARMD-Predict authors did not release weights, and
  these are not theirs.
- **Split.** Patients are split 60/20/20 into train, validation and test, within each site, and the sites are
  pooled. Validation sets the penalty, early stopping and similar choices. Metrics in `manifest.json` are on the
  pooled test split.
- **Features.**
  - `shared`: 30 features the simulator also records (age band, sex, ICU or ward, prior resistant isolates, prior
    positive culture, prior antibiotics by class, urinary catheter, nursing home, four comorbidities and their
    count). These models run on simulated patients through `shared_features`.
  - `full`: ARMD-Predict's feature blocks (including laboratory values and every comorbidity component). These need
    ARMD's own feature tables, which this repository does not build.
- **Families.** XGBoost (`shared`, `full`); L2 logistic regression (`shared`, `full`); a deep ensemble of five MLPs
  (`shared`, `full`); a hierarchical Bayesian logistic regression with site intercepts and 200 posterior draws
  (`shared`). The ensemble and the Bayesian model give a distribution over P(susceptible) through
  `predict_draws`.

## Test statistics and intervals

`test_metrics.csv` (here and in the release) has one row per model, computed by running the released files on
the pooled test split of their cell. Each statistic has a point estimate on the whole test split and a 95%
percentile interval (`_lo`, `_hi`) from 1,000 bootstrap resamples of patients, so a patient's orders stay
together. All models in a cell share the same resamples, so differences between them are paired.

| Column | Meaning |
|---|---|
| `auroc`, `auprc` | discrimination; the positive class is *susceptible*, as in ARMD-Predict |
| `spec90` | specificity at 90% sensitivity |
| `brier`, `ece` | Brier score; expected calibration error over 10 equal-width bins |
| `cal_slope`, `cal_intercept` | logistic recalibration y ~ a + b logit(p); 1 and 0 are ideal, a slope below 1 means overconfident |
| `oe_ratio` | observed over expected share susceptible |
| `auroc_vs_xgb_shared` | paired AUROC difference from the shared-feature XGBoost model of the same cell |
| `auroc_<site>` | AUROC on one site's test patients |
| `draw_mean_sd`, `draw_mean_width90` | for models with draws: mean sd of P(susceptible) across draws, and the mean width of its 90% interval |

These intervals describe sampling uncertainty in the test split of these three sites. They say nothing about
performance at another hospital, which can be much worse (leave-one-site-out costs about 0.03 AUROC here).

Shared-feature models, pooled test split: AUROC (95% interval) and calibration slope.

| Cell | Test orders | Susceptible | XGBoost | Logistic | MLP ensemble | Bayesian |
|---|---|---|---|---|---|---|
| inpatient urine ceftriaxone | 24,849 | 89.8% | 0.732 (0.717–0.744); slope 1.03 | 0.725 (0.710–0.738); slope 1.04 | 0.725 (0.710–0.737); slope 1.10 | 0.742 (0.728–0.755); slope 1.05 |
| inpatient urine piperacillin-tazobactam | 24,574 | 96.1% | 0.662 (0.643–0.681); slope 1.02 | 0.663 (0.644–0.683); slope 1.00 | 0.658 (0.638–0.678); slope 1.17 | 0.672 (0.653–0.690); slope 0.93 |
| inpatient urine meropenem | 18,461 | 98.9% | 0.781 (0.740–0.819); slope 0.94 | 0.771 (0.729–0.810); slope 0.93 | 0.778 (0.737–0.815); slope 0.89 | 0.738 (0.692–0.778); slope 0.80 |
| inpatient urine ciprofloxacin | 31,893 | 77.9% | 0.722 (0.714–0.731); slope 1.01 | 0.719 (0.710–0.727); slope 0.99 | 0.720 (0.712–0.729); slope 0.98 | 0.724 (0.716–0.733); slope 0.99 |
| inpatient urine gentamicin | 31,161 | 89.8% | 0.629 (0.615–0.645); slope 1.04 | 0.630 (0.615–0.646); slope 1.03 | 0.630 (0.615–0.645); slope 0.99 | 0.637 (0.622–0.652); slope 1.01 |
| inpatient urine amikacin | 13,151 | 99.1% | 0.761 (0.717–0.810); slope 0.99 | 0.765 (0.723–0.811); slope 0.89 | 0.765 (0.721–0.812); slope 0.89 | 0.736 (0.686–0.787); slope 0.81 |
| inpatient blood ceftriaxone | 4,568 | 83.9% | 0.670 (0.645–0.695); slope 0.91 | 0.676 (0.651–0.701); slope 0.99 | 0.680 (0.656–0.704); slope 0.91 | 0.703 (0.679–0.726); slope 1.03 |
| inpatient blood piperacillin-tazobactam | 3,216 | 94.9% | 0.631 (0.584–0.680); slope 0.83 | 0.628 (0.582–0.673); slope 0.85 | 0.630 (0.583–0.676); slope 0.83 | 0.627 (0.579–0.674); slope 0.82 |
| inpatient blood meropenem | 4,184 | 95.1% | 0.716 (0.668–0.766); slope 1.07 | 0.739 (0.694–0.780); slope 1.11 | 0.744 (0.698–0.788); slope 1.00 | 0.782 (0.737–0.823); slope 1.13 |
| inpatient blood ciprofloxacin | 4,640 | 76.9% | 0.672 (0.649–0.694); slope 0.94 | 0.649 (0.626–0.673); slope 0.92 | 0.657 (0.633–0.681); slope 0.95 | 0.649 (0.627–0.672); slope 0.95 |
| inpatient blood gentamicin | 6,048 | 89.7% | 0.616 (0.587–0.642); slope 0.96 | 0.612 (0.583–0.638); slope 0.82 | 0.611 (0.580–0.637); slope 0.90 | 0.616 (0.585–0.641); slope 0.80 |
| inpatient blood amikacin | 2,067 | 97.9% | 0.638 (0.530–0.733); slope 1.44 | 0.613 (0.501–0.708); slope 0.81 | 0.623 (0.513–0.721); slope 0.79 | 0.714 (0.594–0.818); slope 0.93 |
| outpatient urine ciprofloxacin | 15,896 | 79.9% | 0.725 (0.713–0.738); slope 0.99 | 0.729 (0.716–0.742); slope 1.00 | 0.728 (0.716–0.741); slope 0.90 | 0.698 (0.683–0.711); slope 0.90 |

## Limits

- Research software. Not validated for clinical use and not guidance for treating anyone.
- Trained on three US health systems. Resistance rates and the meaning of record fields differ elsewhere; expect
  the intercept to need re-estimating on local isolates before any use outside these sites.
- The target is the laboratory result for a cultured isolate. It is not whether the drug would fail: tests miss
  some failures (for example piperacillin-tazobactam against OXA-1 co-producers), and patients without a positive
  culture are not in the training data.
- On simulated patients the models are less accurate than on ARMD (AUROC roughly 0.59 to 0.64 against 0.65 to 0.75),
  because the simulator's records differ from real ones in age, sex and history depth. That gap is under work.

## Files in the release

`<family>/<features>/<cell>.npz` for logistic, MLP and Bayesian models; `xgboost/<features>/<cell>.json` (native
XGBoost JSON) for trees; `manifest.json` with each file's SHA-256, input columns and test metrics. `fetch()`
checks the tarball against `release.json` and every file against the manifest.

| Family | Arrays |
|---|---|
| all `.npz` | `prep_columns`, `prep_median`, `prep_missing`, `prep_log`, `prep_mean`, `prep_sd`: median imputation with missing indicators appended, `log1p` on the `prep_log` columns, then standardisation |
| logistic | `w` (intercept first) |
| mlp_ensemble | `member{i}_W{j}`, `member{i}_b{j}`, j = 0, 1, 2; ReLU after layers 0 and 1, sigmoid output |
| bayes_logistic | `beta` (draws × inputs), `intercept_site` (draws × sites), `sites`, `mu_a`, `tau_a` (population of site intercepts, used for a new site) |
