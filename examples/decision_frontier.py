"""Where published decision rules sit on the trade-off between the patient and the community.

    .venv/bin/python -I examples/decision_frontier.py               # outcome setting theta1 = 0.9, 50,000 patients
    .venv/bin/python -I examples/decision_frontier.py --theta1 0.6 --n 20000

The `Belief` rule steps up once P(A fails), updated daily from vital signs, passes a threshold t. Sweeping t traces
deaths averted against days on drug B, on the same patients throughout. Each rule in decision_rules.py is a threshold,
so it is a point on that curve; the bar below shows the days on B its plausible settings imply (5th-95th
percentile).
Writes examples/output/decision_frontier.csv, and decision_frontier.png if matplotlib is installed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "examples")]

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import decision_rules as dr  # noqa: E402
import policies as pol  # noqa: E402
from ettb.loop import run_loop  # noqa: E402
from full_loop import simulate_site  # noqa: E402

OUT = ROOT / "examples" / "output"
GRID = np.r_[0.0, np.arange(0.02, 1.0, 0.02), 1.0]


def frontier(theta1=0.9, n=50_000, draws=300) -> pd.DataFrame:
    site = simulate_site(theta1, n)
    world, obs, env, p = site["world"], site["obs"], site["env_params"], site["p_fail"]

    def run(model, curve, t=np.nan):
        r = run_loop(13, world, model, env, obs=obs)
        return {"curve": curve, "t": t, "died": r["died"].sum(), "b_days": 1000 / n * r["b_days"].sum()}

    rows = [run(pol.Never(), "never"), run(pol.Oracle(site["truth"]["b_helps"]), "oracle")]
    rows += [run(pol.Belief(p, t), "predictor", t) for t in GRID]
    rows += [run(pol.Belief(np.full_like(p, p.mean()), t), "antibiogram", t) for t in GRID[::2]]
    for name in dr.RULES:
        lo, hi = np.percentile(dr.threshold_draws(name, draws), [5, 95])
        rows.append(run(pol.Belief(p, dr.effective_threshold(name)), name, dr.effective_threshold(name))
                    | {"t_low": lo, "t_high": hi})
    df = pd.DataFrame(rows)
    df["deaths_averted"] = 1000 / n * (df.loc[df.curve == "never", "died"].iloc[0] - df.died)
    return df.drop(columns="died").assign(theta1=theta1)


def plot(df, path):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 5))
    for curve, style in (("predictor", dict(color="#2a78d6", lw=2, label="Predictor + vital signs")),
                         ("antibiogram", dict(color="#888780", lw=1.5, ls=":", label="Antibiogram + vital signs"))):
        c = df[df.curve == curve]
        ax.plot(c.b_days, c.deaths_averted, **style)
    o = df[df.curve == "oracle"].iloc[0]
    ax.scatter(o.b_days, o.deaths_averted, marker="*", s=120, color="#444441", label="Oracle")
    c = df[df.curve == "predictor"]
    for k, (name, colour) in enumerate(zip(dr.RULES, ["#eb6834", "#1baf7a", "#eda100", "#6250d6"])):
        r = df[df.curve == name].iloc[0]
        span = np.interp([r.t_high, r.t_low], c.t, c.b_days)       # B-days fall as the threshold rises
        y = -0.08 * (k + 1)
        ax.plot(span, [y, y], color=colour, lw=5, alpha=0.5, solid_capstyle="butt")
        ax.scatter(r.b_days, r.deaths_averted, s=50, color=colour, edgecolor="white", zorder=5, label=name)
        ax.scatter(r.b_days, y, s=20, color=colour, zorder=5)
    ax.axhline(0, color="#c3c2b7", lw=0.8)
    ax.text(1000, -0.08 * (len(dr.RULES) + 1), "Bars: days on B across plausible settings (5th-95th percentile)",
            ha="right", va="center", fontsize=8, color="#5f5e5a")
    ax.set_xlim(0, 1000)
    ax.set_xlabel("Days on drug B per 1000 patients")
    ax.set_ylabel("Deaths averted per 1000 patients")
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--theta1", type=float, default=0.9)
    ap.add_argument("--n", type=int, default=50_000)
    args = ap.parse_args()
    df = frontier(args.theta1, args.n)
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "decision_frontier.csv", index=False)
    rules = df[df.curve.isin(dr.RULES)][["curve", "t", "t_low", "t_high", "b_days", "deaths_averted"]]
    print(rules.round(2).to_string(index=False))
    try:
        plot(df, OUT / "decision_frontier.png")
        print(f"Figure: {OUT / 'decision_frontier.png'}")
    except ImportError:
        print("matplotlib is not installed; the CSV holds the results.")


if __name__ == "__main__":
    main()
