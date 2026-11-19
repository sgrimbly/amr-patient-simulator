# The full loop: simulator, predictor and decision model

`full_loop.py` runs the whole pipeline on simulated patients. A predictor turns each patient's record into the probability that drug *A* will fail, a decision model uses that probability and the daily vital signs to decide whether to step up to drug *B*, and the simulator runs each patient's course. Because the simulator knows the truth, both stages are scored against it.

```bash
.venv/bin/python -I examples/full_loop.py                        # three outcome settings, 50,000 patients each
.venv/bin/python -I examples/full_loop.py --n 10000 --theta1 0.9  # one setting, quicker
```

The default run takes about ten seconds on a laptop CPU.

## What does the script do?

For each of the three outcome settings in `params/ettb_v1_1d.json` it does four things.

1. **Trains the predictor on a historical cohort.** The labels are what a hospital would have, namely the laboratory's susceptibility result for drug *A*, available only for patients with a bacterial infection.
2. **Predicts for new patients** from their records alone.
3. **Runs each decision model on the same new patients with the same noise.** Differences between rules are therefore counterfactual, not sampling noise.
4. **Scores everything against the truth.** The predictor is scored against culture results (what a hospital could check) and against true failure of *A* in every patient (what it cannot). The decisions are compared per 1000 patients on deaths, deaths averted or caused relative to never stepping up, days on *B*, unnecessary step-ups and days on a failing drug.

It ends by printing one patient's course day by day.

## What goes in and out each day?

Each patient has the following fixed on day 0.

| Variable | Can the models see it? | Comes from |
|---|---|---|
| Record (age, prior antibiotics, prior resistant isolate, care setting and so on) | yes | simulator |
| P(drug *A* fails) | yes | predictor, from the record |
| Does *A* work? Does *B* work? | no | simulator |
| Severity | only as a noisy reading at presentation | simulator |
| Day the culture result arrives | no, until it arrives | simulator |

Then each day *t* runs five steps in order.

| Step | Goes in | Comes out |
|---|---|---|
| 1. Observe | severity *S<sub>t</sub>* (hidden) | vital signs *v<sub>t</sub>* |
| 2. Decide | P(*A* fails), the vital signs so far, the day | step up today, yes or no |
| 3. Treat | stepped up so far? result back yet? | whether today's drug works |
| 4. Death | *S<sub>t</sub>* | died today, yes or no |
| 5. Next day | *S<sub>t</sub>*, whether today's drug works | *S<sub>t+1</sub>* |

By day 30 each patient has an outcome (died or not, and the day), whether and when they stepped up, days on *B*, days on a failing drug and whether the step-up was unnecessary.

A step-up on day 2 cannot change that day's risk of death, which depends on the severity that morning. It changes the move from day 2 to day 3, so its benefit starts on day 3. A step-up is possible only before the culture result arrives, which is on day 2 for most patients.

## How do I plug in my own model?

A predictor needs `fit(table, labels)` and `predict(table)`, returning one probability per patient. `predictors.py` has a logistic regression and a one-variable rule, and `predictors.record_table` builds the table of recorded variables. Any model with those two methods works. A scikit-learn classifier needs a small wrapper whose `predict` returns `predict_proba(table)[:, 1]`.

A decision model needs `reset(first_look)` and `act(day)`. `act` is called every morning and returns one yes or no per patient. The `day` dictionary holds only what a clinician could see.

| Key | Meaning |
|---|---|
| `t` | the day, 0 to 29 |
| `vitals` | the vital-sign reading, NaN on day 0 and when not measured |
| `can_step` | alive, still on *A*, and the culture result not yet back |
| `result_back` | the result has arrived and therapy is now directed |
| `alive` | alive this morning |

Requests where `can_step` is false are ignored, and a step-up cannot be undone. `first_look["presenting_severity"]` is the severity reading at presentation. Start from `MyPolicy` in `policies.py`, then add it to the `rules` dictionary in `full_loop.py`. `ettb/loop.py` describes the interface in full.

The decision model can be written in numpy, PyTorch or anything else that returns an array, because `run_loop` calls it once per day from Python. The simulator itself stays compiled.

## What do the built-in rules show?

`policies.py` has five rules for comparison. They are reference points, not recommendations, and their thresholds are illustrative.

| Rule | What it does |
|---|---|
| `Never` | stays on *A* until the result |
| `AlwaysB` | starts everyone on *B* |
| `Threshold` | steps up on day 0 when P(*A* fails) is above a cut-off |
| `Belief` | combines P(*A* fails) with the vital signs through two Kalman filters, and steps up once the updated probability passes a cut-off |
| `Oracle` | steps up exactly when it helps. It sees the truth, so no real rule can do this |

The script does not weigh deaths against days on *B*. That trade-off belongs to the decision model, and the table gives what is needed to apply one.

## Caveats

The parameters describe a synthetic US-like site calibrated to published figures. They are not estimates for any real hospital, and the harm of an ineffective first drug is not pinned down, which is why the script reports three outcome settings. Results should be compared across all three. Nothing here is guidance on prescribing.
