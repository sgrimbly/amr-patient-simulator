"""The full loop on simulated patients: record -> predictor -> decision model -> outcomes, scored against the truth.

    .venv/bin/python -I examples/full_loop.py            # all three outcome settings, 50,000 patients each
    .venv/bin/python -I examples/full_loop.py --n 10000 --theta1 0.9
    .venv/bin/python -I examples/full_loop.py --armd mlp_ensemble   # predictor trained on real records (ARMD)

Steps, for each outcome setting of the calibrated US-like site (ettb/calibrated_us_site.json):
  1. Simulate a historical cohort and train the predictor on its culture results.
  2. Simulate a new cohort and predict P(drug A fails) from each patient's record.
  3. Run each decision model on the same patients, with the same noise, so differences are counterfactual.
  4. Score the predictor and the decisions against the hidden truth.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import numpy as np  # noqa: E402

import policies as pol  # noqa: E402
import predictors as pred  # noqa: E402
from ettb.loop import run_loop  # noqa: E402
from ettb.params import CALIBRATED_US_SITE, DRUGS, calibrated_site  # noqa: E402
from ettb.world import observe, sample_world  # noqa: E402


def summarise(name, r, never, n):
    per1000 = 1000 / n
    return {"rule": name, "stepped up (%)": 100 * r["stepped"].mean(), "deaths": per1000 * r["died"].sum(),
            "averted vs never": per1000 * (never["died"] & ~r["died"]).sum(),
            "caused vs never": per1000 * (~never["died"] & r["died"]).sum(),
            "days on B": per1000 * r["b_days"].sum(), "unnecessary step-ups": per1000 * r["unnecessary_b"].sum(),
            "days on a failing drug": per1000 * r["inactive_days"].sum()}


def print_table(rows):
    cols = list(rows[0])
    print("  " + "  ".join(f"{c:>{max(len(c), 10)}}" for c in cols))
    for row in rows:
        cells = [f"{row[c]:>{max(len(c), 10)}}" if isinstance(row[c], str) else f"{row[c]:>{max(len(c), 10)}.1f}"
                 for c in cols]
        print("  " + "  ".join(cells))


def print_trace(i, r, p_fail, truth, world):
    """One patient's course, day by day, until two days after the culture result."""
    print(f"\n  Patient {i}: P(A fails) from the record {p_fail[i]:.2f}; truth: A "
          f"{'fails' if truth['a_fails'][i] else 'works'}, first drug {DRUGS[pred.drug_a(world)[i]]}, "
          f"result on day {r['result_day'][i]}")
    print("    day  vitals  step up  drug working  severity (hidden)")
    for t in range(min(r["result_day"][i] + 2, 30)):
        v = r["vitals"][i, t]
        print(f"    {t:>3}  {'  -  ' if np.isnan(v) else f'{v:5.2f}'}  {'yes' if r['step_up'][i, t] else 'no':>7}"
              f"  {'yes' if r['alpha'][i, t] > 0.5 else 'no':>12}  {r['severity_path'][i, t]:17.2f}")
    print(f"    died by day 30: {'yes' if r['died'][i] else 'no'}")


def simulate_site(theta1, n, params=CALIBRATED_US_SITE, armd=None) -> dict:
    """Steps 1 and 2: train the predictor on a historical cohort's culture results, then simulate new patients and
    predict P(drug A fails) from their records. Returns the new cohort with its predictions and hidden truth.
    With `armd` set to a model family, step 1 is skipped and a predictor trained on real records is used instead
    (models/README.md)."""
    world_params, env_params = calibrated_site(theta1, params)
    world = sample_world(2, n, world_params, start_id=n)
    obs = observe(2, world, world_params)
    if armd:
        import armd_models as AM
        AM.fetch()
        p_fail = AM.p_drug_a_fails(world, obs, armd)
    else:
        hist = sample_world(1, n, world_params)
        hist_obs = observe(1, hist, world_params)
        y, labelled = pred.culture_labels(hist, hist_obs)
        model = pred.Logistic().fit(pred.record_table(hist, hist_obs)[labelled], y[labelled])
        p_fail = model.predict(pred.record_table(world, obs))
    y_new, lab_new = pred.culture_labels(world, obs)
    return dict(world=world, obs=obs, env_params=env_params, p_fail=p_fail, truth=pred.truth(world, env_params),
                auroc_culture=pred.auroc(p_fail[lab_new], y_new[lab_new]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--theta1", type=float, nargs="*", default=[0.6, 0.9, 1.2])
    ap.add_argument("--params", default=str(CALIBRATED_US_SITE))
    ap.add_argument("--armd", choices=["logistic", "mlp_ensemble", "bayes_logistic", "xgboost"],
                    help="use a predictor trained on ARMD instead of one trained on simulated culture results")
    args = ap.parse_args()

    for theta1 in args.theta1:
        print(f"\n=== Outcome setting theta1 = {theta1} ===")
        s = simulate_site(theta1, args.n, args.params, args.armd)
        world, obs, env_params, p_fail, truth = s["world"], s["obs"], s["env_params"], s["p_fail"], s["truth"]
        print(f"  Predictor: AUROC {s['auroc_culture']:.3f} against culture results "
              f"(cultured patients only), {pred.auroc(p_fail, truth['a_fails']):.3f} against true failure of A "
              f"(all patients). Mean P(A fails) {p_fail.mean():.3f}; true rate {truth['a_fails'].mean():.3f}.")

        # 3. Decision models, all on the same patients and noise.
        rules = {"never": pol.Never(), "always B": pol.AlwaysB(), "threshold 0.3": pol.Threshold(p_fail, 0.3),
                 "belief 0.5": pol.Belief(p_fail, 0.5), "oracle": pol.Oracle(truth["b_helps"])}
        runs = {name: run_loop(13, world, m, env_params, obs=obs) for name, m in rules.items()}

        # 4. Score against the truth (per 1000 patients).
        print("\n  Per 1000 patients:")
        print_table([summarise(name, r, runs["never"], args.n) for name, r in runs.items()])

        r = runs["belief 0.5"]
        late = np.flatnonzero(truth["b_helps"] & (r["step_day"] >= 1))
        if late.size:
            print_trace(late[0], r, p_fail, truth, world)


if __name__ == "__main__":
    main()
