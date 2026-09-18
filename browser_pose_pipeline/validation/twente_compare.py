"""Compare extract.js measures (outputs/twente/ours) with the Twente ground truth (outputs/twente/truth).

1. Time offset per trial: video time = Qualisys time + offset, found by
   matching our contacts to force-plate contacts (leg labels ignored),
   searched within half a step (±0.17 s) of zero.
2. Leg labels: share of our steps whose matching force-plate contact is on
   the same leg.  The side camera films the left side: L = near, R = far.
3. Per step (same leg, within 60 ms): contact / toe-off / contact-time error.
4. Per trial and leg: median of our per-step values vs median of the truth
   over the same steps (what the app reports), across trials: bias, SD, MAE, r.

Usage: python3 twente_compare.py [ours-suffix]   (e.g. ".raw" for untracked legs)
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TRUTH, OURS = HERE / "outputs/twente/truth", HERE / "outputs/twente/ours"
SUFFIX = sys.argv[1] if len(sys.argv) > 1 else ""
# KF variants are all compared with the truth at mid-stance (the spec's KF).
TRUTH_KEY = {"KF_max_seen": "KF", "KF_max_all": "KF", "KF_mid_seen": "KF"}
ANGLES = ["KF", "KF_max_seen", "KF_max_all", "KF_mid_seen", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride"]


def offset(ours, truth):
    t_on = np.array([s["on"] for s in truth]); o_on = np.array([s["on"] for s in ours])
    # Video and Qualisys start together (same length); a wider search can lock
    # onto a contact one step away, which swaps left and right.
    best = max(np.arange(-.17, .17, 0.005), key=lambda d: (np.abs(o_on[:, None] - (t_on[None, :] + d)).min(1) < .04).sum())
    diff = o_on[:, None] - (t_on[None, :] + best)
    near = np.abs(diff).min(1) < .06
    return best + np.median(diff[np.arange(len(o_on)), np.abs(diff).argmin(1)][near]), near.mean()


def main():
    trials, steps = [], []
    for tf in sorted(TRUTH.glob("*.json")):
        of = OURS / (tf.stem + SUFFIX + ".json")
        if not of.exists():
            continue
        t, o = json.loads(tf.read_text()), json.loads(of.read_text())
        truth = [s for s in t["steps"] if .12 <= s["off"] - s["on"] <= .5]
        if len(o["steps"]) < 20:
            print(f"{tf.stem}: only {len(o['steps'])} steps detected, skipped"); continue
        d, matched_share = offset(o["steps"], truth)
        same = 0; pairs = []
        for s in o["steps"]:
            c = min(truth, key=lambda x: abs(s["on"] - (x["on"] + d)))
            if abs(s["on"] - (c["on"] + d)) > .06:
                continue
            same += c["leg"] == s["leg"]
            if c["leg"] == s["leg"]:
                pairs.append((s, c))
        n = sum(abs(s["on"] - (min(truth, key=lambda x: abs(s["on"] - (x["on"] + d)))["on"] + d)) <= .06 for s in o["steps"])
        row = {"trial": tf.stem, "offset_s": d, "matched": matched_share, "leg_ok": same / max(1, n), "n": len(pairs)}
        for leg in "LR":
            P = [(s, c) for s, c in pairs if s["leg"] == leg]
            row[leg] = {k: (np.nanmedian([s[k] for s, _ in P if s[k] is not None]), np.nanmedian([c[TRUTH_KEY.get(k, k)] for _, c in P])) for k in ANGLES} if len(P) >= 5 else None
            row[leg + "_contact"] = (np.median([s["off"] - s["on"] for s, _ in P]), np.median([c["off"] - c["on"] for _, c in P])) if len(P) >= 5 else None
            for s, c in P:
                steps.append({"leg": leg, "ic": (s["on"] - c["on"] - d) * 1000, "to": (s["off"] - c["off"] - d) * 1000,
                              "contact": ((s["off"] - s["on"]) - (c["off"] - c["on"])) * 1000})
        trials.append(row)

    print(f"\n{len(trials)} trials. Offset video - Qualisys: median {np.median([r['offset_s'] for r in trials]) * 1000:.0f} ms "
          f"(range {min(r['offset_s'] for r in trials) * 1000:.0f} to {max(r['offset_s'] for r in trials) * 1000:.0f}); "
          f"steps matched to a force-plate contact: {np.mean([r['matched'] for r in trials]) * 100:.0f}%")
    print("Leg labels agreeing with the force plate, per trial: " + " ".join(f"{r['trial'][4:6]}{r['trial'][-2:]}:{r['leg_ok'] * 100:.0f}%" for r in trials))
    print(f"  mean {np.mean([r['leg_ok'] for r in trials]) * 100:.1f}%, trials >= 95%: {sum(r['leg_ok'] >= .95 for r in trials)}/{len(trials)}")
    print("\nPer step, same leg (ms): mean ± SD")
    for leg, name in (("L", "near (L)"), ("R", "far (R)")):
        S = [s for s in steps if s["leg"] == leg]
        f = lambda k: f"{np.mean([s[k] for s in S]):+.0f} ± {np.std([s[k] for s in S]):.0f}"
        print(f"  {name}: n={len(S)}  contact {f('ic')}  toe-off {f('to')}  contact time {f('contact')}")
    print("\nPer trial (median of steps), ours - truth across trials: bias, SD, MAE, r")
    for leg, name in (("L", "near (L)"), ("R", "far (R)")):
        R = [r for r in trials if r[leg]]
        print(f"  {name}, {len(R)} trials")
        for k in ANGLES + ["contact_s"]:
            if k == "contact_s":
                a = np.array([r[leg + "_contact"][0] for r in R]) * 1000; b = np.array([r[leg + "_contact"][1] for r in R]) * 1000; unit = "ms"
            else:
                a = np.array([r[leg][k][0] for r in R]); b = np.array([r[leg][k][1] for r in R]); unit = "" if k == "overstride" else "°"
            e = a - b; rr = np.corrcoef(a, b)[0, 1] if len(a) > 2 else np.nan
            print(f"    {k:20s} ours {np.mean(a):7.2f}  truth {np.mean(b):7.2f}  bias {np.mean(e):+7.2f}{unit}  SD {np.std(e, ddof=1):6.2f}  MAE {np.mean(np.abs(e)):6.2f}  r {rr:5.2f}")
    (HERE / f"outputs/twente/compare{SUFFIX}.json").write_text(json.dumps(trials, default=float, indent=1))


if __name__ == "__main__":
    main()
