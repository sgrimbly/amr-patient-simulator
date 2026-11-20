# Assumption register

What the simulator assumes, why, how well each assumption is supported, and how to change it. It covers v1.1d,
the calibrated default in `ettb/calibrated_us_site.json`. Keep it current: a change to the model changes this file.

**Type.** *Causal* links are the ones a treatment decision acts on; they must be right for the decision to be
right. *Association* layers only have to reproduce observed data; their internal mechanism can be wrong without
biasing decisions, as long as the record they produce is realistic. *Scope* entries are deliberate limits.

**Support.** *Calibrated*: fitted to published data. *Checked*: compared with data held out from fitting.
*Placeholder*: a plausible value without a source. *Known misfit*: checked and found wrong.

## Population and records

| Assumption | Type | Support | Switch or parameter |
|---|---|---|---|
| Adults admitted with suspected urinary tract infection or gram-negative bloodstream infection | Scope | — | — |
| 15% of suspected UTI and 35% of suspected BSI have no bacterial infection, and recover whatever they receive | Scope / causal | Placeholder | `p_bact_uti`, `p_bact_bsi`; `lam_nb` |
| Age, sex, comorbidities, admissions, care-home residence, catheters and prior antibiotics drive carriage of resistance | Association | Calibrated to published adjusted odds ratios for ESBL risk factors (ABC-PMC); one, prior ESBL isolate, is fitted well below its published value | `col_*` parameters |
| Part of the exposure that drives resistance never appears in the record (community, household, care elsewhere) | Association | Calibrated (community carriage target) | `comm_sd` |
| The record shows a true prior resistant isolate with probability `prior_reveal`, plus false positives | Association | Calibrated | `prior_reveal`, `prior_false_pos`, `record_completeness`, `comorb_coding` |
| Record features are fixed at admission | Scope | — | — |

## Microbiology

| Assumption | Type | Support | Switch or parameter |
|---|---|---|---|
| Three organisms: *E. coli*, *K. pneumoniae*, *P. aeruginosa* | Scope | Mix by syndrome and onset from surveillance | `org_probs` |
| Four acquired mechanisms (ESBL, carbapenemase, fluoroquinolone and aminoglycoside resistance), plus OXA-1 with ESBL and *P. aeruginosa* piperacillin-tazobactam resistance | Association | Prevalences calibrated to surveillance; several miss their targets (see the validation report) | `rho_*`, `tau`, `psi`, `org_mech_logor` |
| One carbapenemase: OXA-48-like and NDM are not distinguished | Scope | — | — |
| MICs drawn from wild-type distributions shifted by mechanisms; S/I/R from EUCAST or CLSI breakpoints | Association | Tables checked against EUCAST v16.1 | `mic_tables_v11c`, `breakpoints_clsi` |
| Laboratory susceptibility testing is correct unless error rates are set | Association | Default 0 | `ast_vme`, `ast_me` |

## The daily process (causal core)

| Assumption | Type | Support | Switch or parameter |
|---|---|---|---|
| A drug is active if the isolate tests S or I; piperacillin-tazobactam has reduced efficacy against OXA-1 co-producers; amikacin counts only for urinary sources | Causal | OXA-1 efficacy fitted to MERINO; graded PK/PD activity available but off | `graded_activity`, `tzp_oxa1_eff`, `amikacin_urinary_only` |
| One latent severity per patient, moving daily: back towards baseline, up by `kappa` while the drug is inactive, down by `lam` while active | Causal | `kappa` fitted per outcome setting to an observational odds ratio of 1.6 for inactive empiric therapy (Paul 2010); three settings reported because the causal effect is not identified | `kappa`, `lam`, `phi`, `sigma` |
| Daily death probability depends only on severity | Causal | Calibrated to 30-day mortality by syndrome and source | `theta0`, `theta1` |
| Decision: everyone starts drug A (ceftriaxone, or piperacillin-tazobactam if hospital onset) and may switch once, irreversibly, to drug B before the result | Scope | — | policy interface (`ettb/loop.py`) |
| The culture result arrives on day 1 to 4 (US-like), or on a lognormal day with a chosen mean | Causal | US-like timing from published turnaround; South African blood cultures about 3 days (one laboratory) | `p_result_day*`, `result_mean` |
| After the result, therapy is directed and active if any formulary drug is active | Causal | Not checked; probably optimistic for organisms with no good option | `directed` argument |
| Vital signs are the severity plus noise, measured daily | Scope | Not calibrated to any measured sign | `sigma_v`, `m0`, `m1` |
| Antibiotics cause no harm other than counting days on drug B | Scope | — | — |
| Clinicians follow the policy exactly | Scope | — | — |
| Patients are independent; transmission is not modelled here | Scope | — | — |
| The only health outcome is death by day 30 | Scope | — | — |

## Kept deliberately

- All randomness is keyed by patient, day and stream, never by the action taken, so any two policies can be
  compared on exactly the same patients.
- The truth (organism, mechanisms, activity, severity) is stored beside the record for every patient, cultured or
  not.
- Association layers are calibrated to associations; only the causal links the decision acts on are checked
  against trials and trial emulations.

## Known misfits

Listed in the validation reports, which the code regenerates. Main ones for v1.1d: the prior ESBL isolate odds
ratio (10 against 26.8); several resistance prevalences; the Mezzadri trial emulation (risk difference -0.02
against -0.074); and ARMD-trained models score about 0.06 AUROC lower on simulated patients than on real ones, a
gap the v1.2 record layer did not close.
