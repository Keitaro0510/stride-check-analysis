"""ステップ4で共通に使うパス、設定、マーカー処理の部品。"""
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
ROOT = HERE.parents[1]
SRC = ROOT / "analysis_data" / "fukuchi2017_running"
STEP1 = HERE.parent / "step1_data" / "outputs"

SEED = 20260917
FS_MARKER = 150.0
FS_FORCE = 300.0
SPEEDS = {"25": 2.5, "35": 3.5, "45": 4.5}
TARGET_SPEED_MS = 10 / 3.6

# 座標: X = 前、Y = 上、Z = 右（ASIS と PSIS、左右の ASIS の位置から確認）
SEGMENTS = {
    "pelvis": ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.Iliac.Crest", "L.Iliac.Crest"],
    **{f"thigh_{s}": [f"{s}.Thigh.Top.Lateral", f"{s}.Thigh.Bottom.Lateral", f"{s}.Thigh.Top.Medial", f"{s}.Thigh.Bottom.Medial"] for s in "RL"},
    **{f"shank_{s}": [f"{s}.Shank.Top.Lateral", f"{s}.Shank.Bottom.Lateral", f"{s}.Shank.Top.Medial", f"{s}.Shank.Bottom.Medial"] for s in "RL"},
    **{f"foot_{s}": [f"{s}.Heel.Top", f"{s}.Heel.Bottom", f"{s}.Heel.Lateral", f"{s}.MT1", f"{s}.MT5"] for s in "RL"},
}
# 静止姿勢にだけある解剖学的な目印を、どのかたまりから再構成するか
LANDMARKS = {
    **{f"{s}.Knee": f"thigh_{s}" for s in "RL"}, **{f"{s}.Knee.Medial": f"thigh_{s}" for s in "RL"},
    **{f"{s}.Ankle": f"shank_{s}" for s in "RL"}, **{f"{s}.Ankle.Medial": f"shank_{s}" for s in "RL"},
}


def setup_font():
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def read_markers(path: Path) -> dict:
    d = pd.read_csv(path, sep="\t")
    names = sorted({c[:-1] for c in d.columns if c != "Time" and c[-1] in "XYZ"})
    out = {}
    for n in names:
        cols = [n + a for a in "XYZ"]
        if all(c in d.columns for c in cols):
            arr = d[cols].to_numpy(float)
            if np.isfinite(arr).any():
                out[n] = arr
    return out


def static_positions(path: Path) -> dict:
    return {k: np.nanmean(v, axis=0) for k, v in read_markers(path).items() if np.isfinite(v).any()}


def kabsch(A: np.ndarray, B: np.ndarray):
    """A（静止姿勢の点群）を B（走行中の点群）に重ねる回転 R と並進 t（B ≈ A R^T + t）。"""
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1, 1, d])
    R = Vt.T @ D @ U.T
    return R, cb - ca @ R.T


def lowpass(x: np.ndarray, fs: float, fc: float, order: int = 4) -> np.ndarray:
    """欠けを直線で補ってから、前後両方向のバターワースフィルタ。"""
    x = np.asarray(x, float).copy()
    idx = np.arange(len(x))
    for j in range(x.shape[1] if x.ndim > 1 else 1):
        col = x[:, j] if x.ndim > 1 else x
        ok = np.isfinite(col)
        if ok.sum() < 10:
            continue
        col[~ok] = np.interp(idx[~ok], idx[ok], col[ok])
    b, a = butter(order, fc / (fs / 2))
    return filtfilt(b, a, x, axis=0)


def reconstruct(dyn: dict, stat: dict, n_frames: int) -> dict:
    """かたまりごとに剛体として重ね、欠けたマーカーと解剖学的な目印を毎フレーム再構成する。"""
    out = {k: v.copy() for k, v in dyn.items()}
    for seg, members in SEGMENTS.items():
        members = [m for m in members if m in stat]
        extra = [lm for lm, s in LANDMARKS.items() if s == seg and lm in stat]
        targets = members + extra
        for t in targets:
            if t not in out:
                out[t] = np.full((n_frames, 3), np.nan)
        A_all = np.array([stat[m] for m in members])
        T_all = np.array([stat[t] for t in targets])
        for f in range(n_frames):
            vis = [i for i, m in enumerate(members) if m in dyn and np.isfinite(dyn[m][f]).all()]
            if len(vis) < 3:
                continue
            R, tr = kabsch(A_all[vis], np.array([dyn[members[i]][f] for i in vis]))
            est = T_all @ R.T + tr
            for j, t in enumerate(targets):
                if not np.isfinite(out[t][f]).all():
                    out[t][f] = est[j]
    return out


def hip_centers(m: dict, stat: dict) -> tuple:
    """Harrington ほか 2007 の回帰式で股関節中心を推定する（骨盤座標系: x 前、y 上、z 右、単位 mm）。
    PW = 左右 ASIS の距離、PD = ASIS の中点と PSIS の中点の距離（静止姿勢で計算）。"""
    PW = np.linalg.norm(stat["R.ASIS"] - stat["L.ASIS"])
    PD = np.linalg.norm((stat["R.ASIS"] + stat["L.ASIS"]) / 2 - (stat["R.PSIS"] + stat["L.PSIS"]) / 2)
    asis_mid = (m["R.ASIS"] + m["L.ASIS"]) / 2
    psis_mid = (m["R.PSIS"] + m["L.PSIS"]) / 2
    z = m["R.ASIS"] - m["L.ASIS"]
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    xp = asis_mid - psis_mid
    y = np.cross(z, xp)
    y /= np.linalg.norm(y, axis=1, keepdims=True)
    x = np.cross(y, z)
    local_x, local_y, local_z = -0.24 * PD - 9.9, -0.30 * PW - 10.9, 0.33 * PW + 7.3
    R = asis_mid + local_x * x + local_y * y + local_z * z
    L = asis_mid + local_x * x + local_y * y - local_z * z
    return R, L


def angle_between(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    c = np.sum(v1 * v2, axis=-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1))
    return np.degrees(np.arccos(np.clip(c, -1, 1)))
