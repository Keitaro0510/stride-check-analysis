"""Ground truth for the Twente running trials (Zenodo 6457662), same videos as Running_mp4.

Time base: seconds from the first Qualisys frame.  The Qualisys recording
(128 Hz, e.g. 7789 frames = 60.85 s) has the same length as the videos
(2008 frames at 33 fps = 60.85 s), so video time is assumed to start there;
twente_compare.py checks the offset per trial from the contact events.

- Contact / toe-off: vertical force of each treadmill belt (2048 Hz) > 50 N.
- Angles: markers projected on the sagittal plane (X forward, Y up) and put
  through the same 2-D formulas as extract.js sideValues():
  hip = hip joint centre (Harrington, SACR as the PSIS midpoint),
  knee = mid(lateral, medial epicondyle), ankle = mid(malleoli),
  heel = CAL, toe = mid(1st, 5th metatarsal head).
  MediaPipe's hip point is not the joint centre, so part of any difference
  is the pose model's landmark definition, not measurement error.

Usage: python3 twente_truth.py   -> outputs/twente/truth/SubjXX_run_YY.json
"""

import json
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.signal import butter, filtfilt

HERE = Path(__file__).resolve().parent
SRC = HERE.parents[1] / "data/twente_locomotion_6sensors_2022/Processed_data"
OUT = HERE / "outputs/twente/truth"
FORCE_N = 50.0
MIN_CONTACT_S = 0.1


def lowpass(x, fs, fc):
    b, a = butter(2, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def contacts(fy, fs):
    on = lowpass(fy, fs, 50) > FORCE_N
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    starts, ends = edges[on[edges]], edges[~on[edges]]
    out = []
    for s in starts:
        e = ends[ends > s]
        if len(e) and (e[0] - s) / fs >= MIN_CONTACT_S:
            out.append((s / fs, e[0] / fs))
    return out


def hip_centres(m):
    """Harrington 2007 in the pelvis frame; SACR stands in for the PSIS midpoint."""
    rasi, lasi, sacr = m["RASI"], m["LASI"], m["SACR"]
    pw = np.nanmedian(np.linalg.norm(rasi - lasi, axis=1))
    mid = (rasi + lasi) / 2
    pd = np.nanmedian(np.linalg.norm(mid - sacr, axis=1))
    z = (rasi - lasi) / np.linalg.norm(rasi - lasi, axis=1, keepdims=True)
    y = np.cross(z, mid - sacr); y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    lx, ly, lz = -0.24 * pd - 0.0099, -0.30 * pw - 0.0109, 0.33 * pw + 0.0073
    return {"R": mid + lx * x + ly * y + lz * z, "L": mid + lx * x + ly * y - lz * z}


def angle(a, b, c):
    u, v = a - b, c - b
    return np.degrees(np.arccos(np.clip((u * v).sum(-1) / (np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)), -1, 1)))


def side_values(p, leg, pelvis):
    """extract.js sideValues() on sagittal points given as image coords (x forward, y down)."""
    hip, knee, ankle, heel, toe = (p[leg][k] for k in ("hip", "knee", "ankle", "heel", "toe"))
    leg_len = np.linalg.norm(hip - ankle, axis=-1)
    return {
        "KF": 180 - angle(hip, knee, ankle),
        "shank_angle_contact": np.degrees(np.arctan2(knee[..., 0] - ankle[..., 0], ankle[..., 1] - knee[..., 1])),
        "foot_angle_contact": np.degrees(np.arctan2(heel[..., 1] - toe[..., 1], toe[..., 0] - heel[..., 0])),
        "overstride": (heel[..., 0] - pelvis[..., 0]) / leg_len,
    }


def trial(subj, run):
    f = SRC / subj / f"{subj}_{run}.mat"
    d = sio.loadmat(f, squeeze_me=True, struct_as_record=False)["Datastr"]
    lab = list(d.Marker.DataLabel)
    X = np.transpose(d.Marker.MarkerData, (2, 0, 1)).astype(float)  # frames, markers, xyz
    fs_m = float(d.Marker.FrameRate)
    X = np.where(X == 0, np.nan, X)
    m = {k: X[:, i, :] for i, k in enumerate(lab)}
    for k, v in m.items():  # fill short gaps, then 12 Hz low-pass
        ok = np.isfinite(v).all(1)
        if ok.sum() > 10:
            idx = np.arange(len(v))
            m[k] = lowpass(np.column_stack([np.interp(idx, idx[ok], v[ok, j]) for j in range(3)]), fs_m, 12)
    hc = hip_centres(m)
    img = lambda P: np.column_stack([P[:, 0], -P[:, 1]])  # sagittal: x forward, y down
    p = {s: {"hip": img(hc[s]), "knee": img((m[f"{s}LFE"] + m[f"{s}MFE"]) / 2), "ankle": img((m[f"{s}LM"] + m[f"{s}MM"]) / 2),
             "heel": img(m[f"{s}CAL"]), "toe": img((m[f"{s}1MT"] + m[f"{s}5MT"]) / 2)} for s in "LR"}
    pelvis = (p["L"]["hip"] + p["R"]["hip"]) / 2
    vals = {s: side_values(p, s, pelvis) for s in "LR"}
    fs_f = float(d.Force.FrameRate)
    ev = {"R": contacts(d.Force.RightForceData[:, 1], fs_f), "L": contacts(d.Force.LeftForceData[:, 1], fs_f)}
    steps = []
    for leg in "LR":
        for on, off in ev[leg]:
            i_on, i_mid = int(round(on * fs_m)), int(round((on + off) / 2 * fs_m))
            if i_mid >= len(pelvis):
                continue
            v = vals[leg]
            steps.append({"leg": leg, "on": on, "off": off,
                          "KF": float(v["KF"][i_mid]), "knee_flex_contact": float(v["KF"][i_on]),
                          "shank_angle_contact": float(v["shank_angle_contact"][i_on]),
                          "foot_angle_contact": float(v["foot_angle_contact"][i_on]), "overstride": float(v["overstride"][i_on])})
    steps.sort(key=lambda s: s["on"])
    return {"subject": subj, "run": run, "duration_s": len(pelvis) / fs_m, "marker_hz": fs_m,
            "mass_kg": float(d.Info.subjMass), "height_m": float(d.Info.subjHeight), "steps": steps}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in sorted(SRC.glob("Subj*/Subj*_run_*.mat")):
        subj, run = f.stem.split("_", 1)
        try:
            t = trial(subj, run)
        except Exception as e:  # one broken trial should not stop the rest
            print(f"{subj} {run}: {e}"); continue
        (OUT / f"{subj}_{run}.json").write_text(json.dumps(t))
        n = {s: sum(x["leg"] == s for x in t["steps"]) for s in "LR"}
        print(f"{subj} {run}: {t['duration_s']:.2f} s, steps L{n['L']} R{n['R']}")


if __name__ == "__main__":
    main()
