"""Knee flexion from MediaPipe's 3-D world landmarks, compared with the truth.

For each of our steps (side-view events), take the frame at mid-stance
(KF) and at initial contact (knee_flex_contact) in the front video, and in
the side video, and compute the 3-D hip-knee-ankle angle from the world
landmarks.  Front video: left/right labels are reliable.  Side video: world
labels follow the raw 2-D labels, so it is only usable where those are right.
Usage: python3 twente_kf3d.py   (after twente_ours.mjs and twente_compare.py)
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
T, O, MP = HERE / "outputs/twente/truth", HERE / "outputs/twente/ours", HERE / "outputs/mp"
IDX = {"L": (23, 25, 27), "R": (24, 26, 28)}


def flex(w, leg):
    if w is None:
        return np.nan
    h, k, a = (np.array(w[i][:3]) for i in IDX[leg])
    u, v = h - k, a - k
    return 180 - np.degrees(np.arccos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1)))


def main():
    cmp = {r["trial"]: r for r in json.loads((HERE / "outputs/twente/compare.json").read_text())}
    rows = []
    for tf in sorted(T.glob("*.json")):
        t = tf.stem
        if t not in cmp:
            continue
        d = cmp[t]["offset_s"]
        truth = [s for s in json.loads(tf.read_text())["steps"] if .12 <= s["off"] - s["on"] <= .5]
        ours = json.loads((O / tf.name).read_text())["steps"]
        views = {v: json.loads((MP / f"{t}_{v}.json").read_text()) for v in ("front", "side")}
        for leg in "LR":
            vals = {k: [] for k in ("t_KF", "t_kfc", "front_KF", "front_kfc", "side_KF", "side_kfc")}
            for s in ours:
                if s["leg"] != leg:
                    continue
                c = min(truth, key=lambda x: abs(s["on"] - x["on"] - d))
                if abs(s["on"] - c["on"] - d) > .06 or c["leg"] != leg:
                    continue
                vals["t_KF"].append(c["KF"]); vals["t_kfc"].append(c["knee_flex_contact"])
                for v, V in views.items():
                    at = lambda sec: V["world"][min(len(V["world"]) - 1, int(round(sec * V["fps"])))]
                    vals[f"{v}_KF"].append(flex(at((s["on"] + s["off"]) / 2), leg))
                    vals[f"{v}_kfc"].append(flex(at(s["on"]), leg))
            if len(vals["t_KF"]) >= 5:
                rows.append({"trial": t, "leg": leg, **{k: float(np.nanmedian(v)) for k, v in vals.items()}})
    for leg, name in (("L", "near (L)"), ("R", "far (R)")):
        R = [r for r in rows if r["leg"] == leg]
        print(f"{name}: {len(R)} trials")
        for est, ref, label in (("front_KF", "t_KF", "KF, front 3-D"), ("side_KF", "t_KF", "KF, side 3-D"),
                                ("front_kfc", "t_kfc", "knee_flex_contact, front 3-D"), ("side_kfc", "t_kfc", "knee_flex_contact, side 3-D")):
            a, b = np.array([r[est] for r in R]), np.array([r[ref] for r in R]); e = a - b
            print(f"  {label:30s} ours {a.mean():6.1f}  truth {b.mean():6.1f}  bias {e.mean():+6.1f}°  SD {e.std(ddof=1):5.1f}  r {np.corrcoef(a, b)[0, 1]:5.2f}")
    print(f"truth KF across trials: SD {np.std([r['t_KF'] for r in rows], ddof=1):.1f}°, knee_flex_contact SD {np.std([r['t_kfc'] for r in rows], ddof=1):.1f}°")


if __name__ == "__main__":
    main()
