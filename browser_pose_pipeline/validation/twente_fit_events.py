"""Refit the toe-off offset on MediaPipe points of the Twente trials.

extract.js: toe-off = moment the toe is furthest behind the pelvis
- K * stride.  K was fitted on Fukuchi markers (MT heads); here it is fitted
on MediaPipe's toe tip against the force plates, leaving one subject out at
a time (fit on the others, test on the one left out).
Usage: python3 twente_fit_events.py [tag]   (after twente_ours.mjs; tag e.g. "heavy")
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T, O = HERE / "outputs/twente/truth", HERE / "outputs/twente/ours"


def pairs():
    out = []
    for tf in sorted(T.glob("*.json")):
        of = O / (tf.stem + (("." + sys.argv[1]) if len(sys.argv) > 1 else "") + ".json")
        if not of.exists():
            continue
        truth = [s for s in json.loads(tf.read_text())["steps"] if .12 <= s["off"] - s["on"] <= .5]
        ours = json.loads(of.read_text())["steps"]
        if len(ours) < 20:
            continue
        on = np.array([s["on"] for s in truth])
        d = max(np.arange(-.17, .17, .005), key=lambda d: (np.abs(np.array([s["on"] for s in ours])[:, None] - on[None] - d).min(1) < .04).sum())
        for s in ours:
            c = min(truth, key=lambda x: abs(s["on"] - x["on"] - d))
            if abs(s["on"] - c["on"] - d) <= .06 and c["leg"] == s["leg"]:
                out.append({"subj": tf.stem[:6], "trial": tf.stem, "leg": s["leg"], "stride": s["stride_s"], "back": s["back_s"] - d,
                            "on": s["on"] - d, "t_on": c["on"], "t_off": c["off"]})
    return out


def main():
    P = pairs()
    subjects = sorted({p["subj"] for p in P})
    k_all = np.median([(p["back"] - p["t_off"]) / p["stride"] for p in P])
    ms_all = np.median([p["back"] - p["t_off"] for p in P])
    print(f"{len(P)} steps, {len(subjects)} subjects.  Fitted on all: K = {k_all:.4f} (stride fraction), or {ms_all * 1000:.0f} ms")
    for name, pred in (("stride fraction", lambda p, k: p["back"] - k * p["stride"]), ("constant ms", lambda p, k: p["back"] - k)):
        err_step, trial_rows = [], {}
        for subj in subjects:
            train = [p for p in P if p["subj"] != subj]
            k = np.median([(p["back"] - p["t_off"]) / p["stride"] for p in train]) if name == "stride fraction" else np.median([p["back"] - p["t_off"] for p in train])
            for p in (p for p in P if p["subj"] == subj):
                c_est, c_true = pred(p, k) - p["on"], p["t_off"] - p["t_on"]
                err_step.append((c_est - c_true) * 1000)
                trial_rows.setdefault((p["trial"], p["leg"]), []).append((c_est, c_true))
        tr = np.array([(np.median([a for a, _ in v]), np.median([b for _, b in v])) for v in trial_rows.values()])
        e = (tr[:, 0] - tr[:, 1]) * 1000
        print(f"  leave-one-subject-out, {name}: contact time per step {np.mean(err_step):+.0f} ± {np.std(err_step):.0f} ms;"
              f" per trial and leg {np.mean(e):+.1f} ± {np.std(e, ddof=1):.1f} ms, MAE {np.mean(np.abs(e)):.1f} ms, r {np.corrcoef(tr[:, 0], tr[:, 1])[0, 1]:.2f} (n={len(e)})")


if __name__ == "__main__":
    main()
