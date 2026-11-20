# Predictors trained on ARMD

Trained models that predict, from a patient's record, whether a culture isolate will be susceptible to an
antibiotic. They were trained on real hospital records so that the decision examples can use a predictor that has
never seen simulated data. The weights are in a GitHub release; this folder holds the model card, the manifest
(files, columns, checksums), the test statistics (`test_metrics.csv`) and `release.json` (which release to download
and its checksum).

**Current release: [`models-v1`](https://github.com/sgrimbly/amr-patient-simulator/releases/tag/models-v1).**
All families are trained on the final ARMD feature tables: 52 shared-feature models (XGBoost, logistic, MLP
ensemble and Bayesian, 13 cells each) and 39 full-feature models (XGBoost, logistic and MLP ensemble). It replaces
the interim `models-v0.9`, with the same interface.

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
performance at another hospital. When a site is left out of training and used as the test site, AUROC falls by
only about 0.005, but calibration error roughly doubles (about 0.017 to 0.03), so expect to recalibrate on local
isolates.

Shared-feature models, pooled test split: AUROC (95% interval) and calibration slope.

| Cell | Test orders | Susceptible | XGBoost | Logistic | MLP ensemble | Bayesian |
|---|---|---|---|---|---|---|
| inpatient urine ceftriaxone | 24,849 | 89.8% | 0.732 (0.717–0.744); slope 1.03 | 0.731 (0.716–0.743); slope 1.02 | 0.733 (0.719–0.745); slope 1.06 | 0.743 (0.730–0.755); slope 1.03 |
| inpatient urine piperacillin-tazobactam | 24,574 | 96.1% | 0.662 (0.643–0.681); slope 1.02 | 0.665 (0.645–0.684); slope 0.99 | 0.666 (0.646–0.685); slope 1.05 | 0.676 (0.657–0.694); slope 1.03 |
| inpatient urine meropenem | 18,461 | 98.9% | 0.781 (0.740–0.819); slope 0.94 | 0.775 (0.733–0.814); slope 0.93 | 0.778 (0.736–0.816); slope 0.93 | 0.789 (0.748–0.826); slope 0.97 |
| inpatient urine ciprofloxacin | 31,893 | 77.9% | 0.722 (0.714–0.731); slope 1.01 | 0.723 (0.714–0.731); slope 1.00 | 0.724 (0.716–0.733); slope 1.01 | 0.729 (0.720–0.737); slope 1.00 |
| inpatient urine gentamicin | 31,161 | 89.8% | 0.629 (0.615–0.645); slope 1.04 | 0.632 (0.617–0.648); slope 1.03 | 0.632 (0.617–0.648); slope 0.98 | 0.640 (0.624–0.655); slope 1.00 |
| inpatient urine amikacin | 13,151 | 99.1% | 0.761 (0.717–0.810); slope 0.99 | 0.764 (0.721–0.809); slope 0.89 | 0.766 (0.718–0.814); slope 0.92 | 0.800 (0.756–0.846); slope 0.97 |
| inpatient blood ceftriaxone | 4,568 | 83.9% | 0.670 (0.645–0.695); slope 0.91 | 0.676 (0.651–0.702); slope 0.97 | 0.681 (0.657–0.706); slope 0.97 | 0.704 (0.681–0.727); slope 1.02 |
| inpatient blood piperacillin-tazobactam | 3,216 | 94.9% | 0.631 (0.584–0.680); slope 0.83 | 0.632 (0.587–0.678); slope 0.88 | 0.632 (0.585–0.678); slope 0.83 | 0.640 (0.597–0.689); slope 0.93 |
| inpatient blood meropenem | 4,184 | 95.1% | 0.716 (0.668–0.766); slope 1.07 | 0.743 (0.697–0.786); slope 1.12 | 0.744 (0.700–0.787); slope 1.13 | 0.787 (0.744–0.828); slope 1.11 |
| inpatient blood ciprofloxacin | 4,640 | 76.9% | 0.672 (0.649–0.694); slope 0.94 | 0.668 (0.644–0.688); slope 0.93 | 0.676 (0.652–0.698); slope 0.98 | 0.667 (0.644–0.688); slope 0.96 |
| inpatient blood gentamicin | 6,048 | 89.7% | 0.616 (0.587–0.642); slope 0.96 | 0.621 (0.592–0.648); slope 0.86 | 0.619 (0.590–0.647); slope 0.88 | 0.629 (0.601–0.654); slope 0.84 |
| inpatient blood amikacin | 2,067 | 97.9% | 0.638 (0.530–0.733); slope 1.44 | 0.625 (0.516–0.721); slope 0.88 | 0.624 (0.516–0.718); slope 0.86 | 0.725 (0.601–0.834); slope 1.02 |
| outpatient urine ciprofloxacin | 15,896 | 79.9% | 0.725 (0.713–0.738); slope 0.99 | 0.728 (0.715–0.741); slope 0.99 | 0.728 (0.715–0.741); slope 0.95 | 0.729 (0.717–0.742); slope 0.99 |

Full-feature models (pooled test split, mean over the 13 cells): XGBoost AUROC 0.734, logistic 0.718, MLP ensemble 0.717; all with calibration error below 0.01. The Bayesian model's higher AUROC on shared features comes partly from its site intercepts: on ARMD it knows each patient's hospital, which the other families do not.

## Limits

- Research software. Not validated for clinical use and not guidance for treating anyone.
- Trained on three US health systems. Resistance rates and the meaning of record fields differ elsewhere; expect
  the intercept to need re-estimating on local isolates before any use outside these sites.
- The target is the laboratory result for a cultured isolate. It is not whether the drug would fail: tests miss
  some failures (for example piperacillin-tazobactam against OXA-1 co-producers), and patients without a positive
  culture are not in the training data.
- On simulated patients the shared-feature models reach an AUROC of about 0.66 against culture results and 0.62
  against true failure of drug *A*, below their AUROC on ARMD. The simulator's records differ from real ones in
  age, sex and history depth; that gap is under work. Models that do not know the site also predict too little
  resistance for simulated patients, which a local intercept correction fixes.

## Comparison with ARMD-Predict

On the nine cells that ARMD-Predict reports (Table 1, four sites including MGB), full-feature XGBoost averages
0.730 AUROC against the paper's 0.731, and is within about 0.025 in every cell. Shared-feature models, which run
on simulated patients, average about 0.685. The two are not evaluated on the same patients (MGB is not public and
our feature construction is our own), so differences in single cells are not evidence that either is better.

## Planned improvements

Changing model family on the same features has moved AUROC by about 0.01; adding features moves it by 0.02 to
0.13. The plan therefore puts information and targets ahead of architectures. Each candidate must beat XGBoost on
paired patient resamples, hold up when a site is left out, and not worsen calibration.

- **Tabular foundation models** (TabPFN, TabICL and similar in-context learners), first on the blood cells, which
  have 6,000 to 18,000 training orders and are where in-context learners are strongest.
- **Blending** XGBoost with the event-history transformer, which ties XGBoost on urine cells.
- **Joint models across drugs and specimens**, so that blood cells borrow from the larger urine cells and
  cross-resistance is shared.
- **Tuned gradient boosting**: the current settings are ARMD-Predict's defaults.
- **Predictions that update as microbiology arrives** (organism identification before susceptibility), matching
  the daily decision loop.
- **Pretraining on all culture orders** (masked or next-event prediction) before fine-tuning each cell.
- **Transport**: site-specific effects, recalibration on a few hundred local isolates, and priors from a local
  antibiogram.
- **Uncertainty with guarantees**: class-conditional conformal prediction, with coverage checked on a left-out site.
- **The decision's target**: P(drug *A* fails) rather than the laboratory result, and training weighted toward the
  high-sensitivity end where step-up decisions are made.

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
