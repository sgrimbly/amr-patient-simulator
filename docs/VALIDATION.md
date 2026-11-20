# Validation

How the two calibrated sites in `ettb/` compare with published data. Each check says whether the target was used
in fitting or held out, and whether the simulator passes it. Failures are listed with the passes: they mark where
the simulator should not be trusted. What each layer assumes is in [ASSUMPTIONS.md](ASSUMPTIONS.md).

## US-like site (`ettb/calibrated_us_site.json`, v1.1d)

### Resistance risk factors (fitted)

A standard multivariable analysis of simulated patients is compared with published adjusted odds ratios for ESBL
risk factors. Simulated values are posterior predictive medians with 90% intervals. A check passes when the
published value lies inside the interval, widened by two of its standard errors.

| Risk factor or summary | Published | Simulated (90% interval) | Result |
|---|---|---|---|
| Prior ESBL isolate in 12 months (OR) | 26.8 | 10.1 (3.4 to 22.4) | fail |
| Beta-lactam or fluoroquinolone course in 3 months (OR) | 3.7 | 2.2 (1.5 to 3.8) | pass |
| Hospital admission in 12 months (OR) | 5.7 | 2.9 (1.4 to 5.0) | pass |
| Transfer from another facility (OR) | 5.6 | 2.7 (1.3 to 6.1) | pass |
| Hospital onset (OR) | 2.1 | 1.7 (1.3 to 2.5) | pass |
| Urinary catheter (OR) | 3.1 | 2.1 (1.2 to 3.4) | pass |
| Proton-pump inhibitor (OR) | 1.6 | 1.5 (1.1 to 1.8) | pass |
| Chronic kidney disease (OR) | 2.2 | 1.7 (1.2 to 2.3) | pass |
| ESBL prevalence | 0.13 | 0.13 (0.11 to 0.17) | pass |
| Community carriage | 0.035 | 0.03 (0.02 to 0.05) | pass |
| AUROC of a published clinical ESBL score | 0.71 | 0.69 (0.65 to 0.73) | pass |
| Age 70 or over (OR) | 3.2 | 0.6 (0.4 to 0.9) | not fitted: conflicts with crude age gradients in admitted cohorts, which were fitted instead |

Resistance prevalences are fitted to US surveillance. Most match their targets; four do not: ceftriaxone
resistance in *E. coli* (12.6% against 15%) and *K. pneumoniae* (20.6% against 16%), piperacillin-tazobactam
non-susceptibility in *E. coli* (3.9% against 8%) and *P. aeruginosa* (0.7% against 16%; no mechanism for it is
modelled), and amikacin non-susceptibility in *E. coli* (12% against 2%; the wild-type MIC spread alone exceeds the
target at the CLSI breakpoint).

### Outcomes (fitted)

| Quantity | Target | Simulated | Result |
|---|---|---|---|
| 30-day mortality, UTI without bloodstream infection | 4% | 3.8% | pass |
| 30-day mortality, urinary-source bloodstream infection | 6% | 6.1% | pass |
| 30-day mortality, other bloodstream infection | 15% | 17.2% | pass |
| 30-day mortality, *P. aeruginosa* bloodstream infection | 22% | 25.2% | pass |
| ICU share, bloodstream infection | 20% | 19.9% | pass |
| ICU share, UTI | 12% | 11.8% | pass |
| Odds ratio of death, inactive against active empiric therapy (all three outcome settings) | 1.6 (1.37 to 1.86) | 1.60 | pass |

Mortality strata are for outcome setting theta1 = 0.9 under usual care.

### Treatment effects: trials emulated on simulated patients

Each trial's eligibility and arms are reproduced on simulated patients, for each outcome setting.

| Trial | Published | Simulated (theta1 0.6 / 0.9 / 1.2) | Role | Result |
|---|---|---|---|---|
| MERINO: 30-day mortality, piperacillin-tazobactam minus meropenem | 0.086 (0.030 to 0.150) | 0.088 / 0.073 / 0.069 | fitted | pass |
| Shiber: risk ratio of death, early broad against narrow | 0.98 (0.79 to 1.20) | 0.99 / 0.99 / 0.99 | held out | pass |
| Mezzadri: 14-day risk difference | -0.074 (-0.122 to -0.025) | -0.021 / -0.018 / -0.018 | held out | fail |

### Prediction on simulated records (held out)

- **ARMD-Predict replica.** The published model's design, trained and tested on simulated records, against its
  reported AUROC (within 0.05 counts as a pass): passes for urine ceftriaxone (0.79 against 0.77), urine
  piperacillin-tazobactam (0.70 against 0.70) and blood piperacillin-tazobactam (0.68 against 0.66); fails for
  meropenem (0.67-0.69 against 0.83), urine ciprofloxacin (0.67 against 0.74), blood ceftriaxone (0.76 against
  0.70), and amikacin, where the simulator has too little aminoglycoside resistance for the model to learn from.
- **Models trained on real records.** The released ARMD-trained models (`models/`) average 0.68 AUROC on ARMD's
  test patients and 0.61 on simulated patients. Simulated records carry less predictive signal than real ones.

## South African anchor site (`ettb/sa_anchor_site.json`)

A public tertiary hospital fitted to *E. coli* and *K. pneumoniae* bloodstream infections at Tygerberg Hospital,
Cape Town (Dramowski et al. 2022), with a 3.1-day culture turnaround (Rule et al. 2021). Observed values are
proportions with exact 95% intervals; a check passes when the simulated value falls inside the interval.

| Quantity | Observed (95% interval) | Simulated | Role | Result |
|---|---|---|---|---|
| 3GC non-susceptible, *E. coli* | 0.24 (0.16 to 0.32) | 0.22 | fitted | pass |
| 3GC non-susceptible, *K. pneumoniae* | 0.52 (0.41 to 0.63) | 0.41 | fitted | pass |
| 3GC non-susceptible, community onset | 0.18 (0.08 to 0.34) | 0.11 | fitted | pass |
| 3GC non-susceptible, healthcare-associated | 0.35 (0.22 to 0.50) | 0.27 | fitted | pass |
| 3GC non-susceptible, hospital onset | 0.41 (0.31 to 0.50) | 0.41 | fitted | pass |
| Share hospital onset | 0.57 (0.50 to 0.64) | 0.53 | fitted | pass |
| Share healthcare-associated | 0.24 (0.18 to 0.31) | 0.17 | fitted | fail |
| Share *K. pneumoniae* | 0.40 (0.33 to 0.47) | 0.39 | fitted | pass |
| Share urinary source | 0.20 (0.15 to 0.27) | 0.18 | fitted | pass |
| HIV, among known status | 0.18 (0.12 to 0.26) | 0.19 | fitted | pass |
| 30-day mortality, 3GC-susceptible | 0.29 (0.21 to 0.37) | 0.29 | fitted | pass |
| 30-day mortality, 3GC-non-susceptible | 0.44 (0.32 to 0.56) | 0.35 | held out | pass |

Two further South African checks fail when the same structure is varied across plausible sites: mortality from
carbapenem-resistant bloodstream infection is too high (about 50% against 31-37% in-hospital case fatality
nationally), and amikacin susceptibility of ESBL producers is too low (about 91% against 95-97%). The first
suggests that the assumption of fully active therapy after the result, or the harm of a long inactive window, is
wrong for organisms with few treatment options.

## Main limits these checks show

1. Simulated records are less predictive than real ones, by about 0.06 AUROC for models trained on real data.
2. The effect of early broad-spectrum therapy in one held-out trial (Mezzadri) is smaller in the simulator than
   observed.
3. Carbapenem-resistant infections are more lethal in the simulator than in South African surveillance.
4. Aminoglycoside resistance is not well calibrated at either site.
