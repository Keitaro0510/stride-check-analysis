"""Fukuchi 2017 の床反力とマーカーから、接地・離地の検出を確かめるデータを作る。

姿勢推定の誤差を除いて、extract.js の「接地・離地の検出」だけの誤差を測るためのもの。
- 正解：床反力（300Hz、Fy > 50N）の接地・離地。左右は圧力中心（COPz）と踵の位置で決める
- 入力：マーカーから作った側方の2D座標（X = 前、Y = 上）を、MediaPipe の点の番号に並べたもの
  股関節 = 股関節中心（Harrington）、膝・足首 = 内外の中点、踵 = Heel.Bottom、つま先 = MT1・MT5 の中点
  鼻は無いので、骨盤の前上に置く（extract.js が進行方向を決めるのに使うだけ）

出力：validation/outputs/fukuchi_events/RBDS###_T##.json（150Hz、最初の SECONDS 秒）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "analysis" / "step4_load"))
from load_common import FS_FORCE, FS_MARKER, SPEEDS, SRC, hip_centers, lowpass, read_markers, reconstruct, static_positions  # noqa: E402

force_contacts = __import__("41_marker_features").force_contacts

OUT = HERE / "outputs" / "fukuchi_events"
SECONDS = 15.0
# MediaPipe Pose の点の番号
MP = {"nose": 0, "LHip": 23, "RHip": 24, "LKnee": 25, "RKnee": 26, "LAnkle": 27, "RAnkle": 28,
      "LHeel": 29, "RHeel": 30, "LToe": 31, "RToe": 32}


def trial(subject: int, code: str, stat: dict) -> dict | None:
    mpath = SRC / f"RBDS{subject:03d}runT{code}markers.txt"
    fpath = SRC / f"RBDS{subject:03d}runT{code}forces.txt"
    if not mpath.exists() or not fpath.exists():
        return None
    dyn = read_markers(mpath)
    n = len(next(iter(dyn.values())))
    m = reconstruct(dyn, stat, n)
    need = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.Knee", "R.Knee.Medial", "L.Knee", "L.Knee.Medial",
            "R.Ankle", "R.Ankle.Medial", "L.Ankle", "L.Ankle.Medial", "R.Heel.Bottom", "L.Heel.Bottom", "R.MT1", "R.MT5", "L.MT1", "L.MT5"]
    if any(k not in m or np.isfinite(m[k]).all(axis=1).mean() < 0.8 for k in need):
        return None
    m = {k: lowpass(v, FS_MARKER, 10) for k, v in m.items() if k in need}
    hjc = dict(zip("RL", hip_centers(m, stat)))
    pts = {}
    for s in "RL":
        pts[f"{s}Hip"] = hjc[s]
        pts[f"{s}Knee"] = (m[f"{s}.Knee"] + m[f"{s}.Knee.Medial"]) / 2
        pts[f"{s}Ankle"] = (m[f"{s}.Ankle"] + m[f"{s}.Ankle.Medial"]) / 2
        pts[f"{s}Heel"] = m[f"{s}.Heel.Bottom"]
        pts[f"{s}Toe"] = (m[f"{s}.MT1"] + m[f"{s}.MT5"]) / 2
    pelvis = (m["R.ASIS"] + m["L.ASIS"] + m["R.PSIS"] + m["L.PSIS"]) / 4
    pts["nose"] = pelvis + np.array([150.0, 650.0, 0.0])

    frames = min(n, int(SECONDS * FS_MARKER))
    # 側方の画像の座標：x = 前（X）、y = 下（-Y）。単位は mm のまま（検出は相対的な閾値だけを使う）
    series = {name: [[round(float(p[f, 0]), 1), round(float(-p[f, 1]), 1)] for f in range(frames)] for name, p in pts.items()}

    contacts, _ = force_contacts(fpath)
    events = []
    for fs, fe, copz in contacts:
        ms, me = int(round(fs * FS_MARKER / FS_FORCE)), int(round(fe * FS_MARKER / FS_FORCE))
        if me >= frames:
            break
        dz = {s: abs(copz - np.nanmean(m[f"{s}.Heel.Bottom"][ms:me, 2])) for s in "RL"}
        if abs(dz["R"] - dz["L"]) < 20:  # 圧力中心が左右の真ん中で判定できない
            continue
        events.append({"leg": min(dz, key=dz.get), "on_s": fs / FS_FORCE, "off_s": fe / FS_FORCE})
    return {"subject": subject, "speed_ms": SPEEDS[code], "fs": FS_MARKER, "landmarks": MP, "series": series, "events": events}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    done = 0
    for subject in range(1, 43):
        spath = SRC / f"RBDS{subject:03d}static.txt"
        if not spath.exists():
            continue
        stat = static_positions(spath)
        for code in SPEEDS:
            try:
                d = trial(subject, code, stat)
            except Exception as e:  # 1試行の失敗で全体を止めない
                print(f"RBDS{subject:03d} T{code}: {e}")
                continue
            if d and len(d["events"]) >= 10:
                (OUT / f"RBDS{subject:03d}_T{code}.json").write_text(json.dumps(d, separators=(",", ":")))
                done += 1
    print(f"{done} trials -> {OUT}")


if __name__ == "__main__":
    main()
