"""The full loop on simulated patients: record -> predictor -> decision model -> outcomes, scored against the truth.

    .venv/bin/python -I examples/full_loop.py            # all three outcome settings, 50,000 patients each
    .venv/bin/python -I examples/full_loop.py --n 10000 --theta1 0.9

Steps, for each outcome setting of the calibrated v1.1d site:
  1. Simulate a historical cohort and train the predictor on its culture results.
  2. Simulate a new cohort and predict P(drug A fails) from each patient's record.
  3. Run each decision model on the same patients, with the same noise, so differences are counterfactual.
  4. Score the predictor and the decisions against the hidden truth.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import numpy as np  # noqa: E402

import policies as pol  # noqa: E402
import predictors as pred  # noqa: E402
from ettb.loop import run_loop  # noqa: E402
from ettb.params import DRUGS  # noqa: E402
from ettb.world import observe, sample_world  # noqa: E402


def load_site(path, theta1):
    """World and environment parameters for one outcome setting of a calibrated site."""
    rep = json.loads(Path(path).read_text())
    base = {k: np.array(v) if isinstance(v, list) else v for k, v in rep["calibrated_params"].items()}
    member = next(f for f in rep["outcome_family"] if abs(f["theta1"] - theta1) < 1e-9)
    return dict(base, **member["world"]), dict(member["env"])


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=50_000)
    ap.add_argument("--theta1", type=float, nargs="*", default=[0.6, 0.9, 1.2])
    ap.add_argument("--params", default=str(ROOT / "params/ettb_v1_1d.json"))
    args = ap.parse_args()

    for theta1 in args.theta1:
        world_params, env_params = load_site(args.params, theta1)
        print(f"\n=== Outcome setting theta1 = {theta1} ===")

        # 1. A historical cohort: the predictor learns from culture results, as a hospital would.
        hist = sample_world(1, args.n, world_params)
        hist_obs = observe(1, hist, world_params)
        y, labelled = pred.culture_labels(hist, hist_obs)
        model = pred.Logistic().fit(pred.record_table(hist, hist_obs)[labelled], y[labelled])

        # 2. New patients: predict from the record only.
        world = sample_world(2, args.n, world_params, start_id=args.n)
        obs = observe(2, world, world_params)
        p_fail = model.predict(pred.record_table(world, obs))
        truth = pred.truth(world, env_params)
        y_new, lab_new = pred.culture_labels(world, obs)
        print(f"  Predictor: AUROC {pred.auroc(p_fail[lab_new], y_new[lab_new]):.3f} against culture results "
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
