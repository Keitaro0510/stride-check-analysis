"""ステップ4-2：Fukuchi のマーカーから「動画で測れる値」を、2D に投影して歩ごとに計算する。

入力: analysis_data/fukuchi2017_running/RBDSxxxstatic.txt, RBDSxxxrunT{25,35,45}markers.txt, forces.txt
      step1_data/outputs/fukuchi_subjects.csv
出力: outputs/marker_steps.csv（1歩 = 1行）, outputs/marker_features.csv（人×速度×脚の中央値）, outputs/marker_check.txt

処理:
  1. 静止姿勢のマーカーの平均位置から、かたまり（骨盤・太もも・すね・足）と解剖学的な目印（膝・足首の内外側）の位置関係を求める
  2. 走行中の各フレームで、見えているマーカー（3個以上）に剛体として重ね、欠けたマーカーと目印を再構成 → 10 Hz の低域通過フィルタ
  3. 関節中心: 股関節 = Harrington ほか 2007 の回帰式、膝 = 膝の内外側の中点、足首 = くるぶしの内外側の中点、踵 = Heel.Bottom、つま先 = MT1 と MT5 の中点
  4. 接地と離地: 床反力（300 Hz、50 Hz の低域通過）の鉛直成分が 50 N を超える区間。左右: 接地中の圧力中心の左右位置（COPz）が、
     どちらの踵の左右位置に近いか（トレッドミルは1枚なので、床反力だけでは左右が分からないため）
  5. 歩ごとの値（後方 = Z-Y 平面、側方 = X-Y 平面に投影。立脚中期 = 立脚期の50%）:
     後方  CPD  骨盤の傾き: 左右の PSIS を結ぶ線の水平からの角度。反対側が下がると正
           HADD 股関節の内転（Loh に近い定義）: 立脚側の股関節中心から反対側の股関節中心へのベクトルと、太もも（股関節中心→膝中心）のなす角。
                内転が大きいほど小さくなる（約90°−内転）
           KA   膝の外反（2D）: 股関節中心・膝中心・足首中心の角度の 180° からのずれ。膝が外側（内反）にずれると正
     側方  KF   膝の屈曲: 太ももとすねのなす角の 180° からのずれ（立脚中期）
           knee_flex_contact 接地時の膝の屈曲、shank_angle_contact 接地時のすねの傾き（膝が足首より前なら正）、
           foot_angle_contact 接地時の足の角度（つま先が上なら正 = 踵接地）、
           overstride 接地時の踵の、骨盤の中心（ASIS と PSIS の中点）からの前方距離 ÷ 脚の長さ（静止姿勢の股関節中心〜足首中心）、
           pelvis_vertical_osc 1歩（接地〜次の接地）の骨盤の中心の上下の幅（mm）
     時間  contact_s, flight_s, step_s, cadence_spm（60 ÷ 1歩の時間）, duty（接地時間 ÷ 1歩の時間）
  6. 品質の確認（人×速度×脚ごとに除外。全件を計算したあと、値の分布を見てから決めた基準）:
     - 歩数が20未満（左右の割り当てや再構成がうまくいっていない）
     - 角度などの中央値が、全体の中央値から 5×MAD（頑健な標準偏差）以上離れている
     - 骨盤の ASIS・PSIS のどれかが、試行の半分以上で欠けている（股関節中心の推定がずれる）
     21番（3.5・4.5 m/s）と26番（4.5 m/s）は再構成が壊れていて（膝の屈曲 100°超など）、35番（3.5 m/s）は右 ASIS が全体で欠けていた
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from load_common import FS_FORCE, FS_MARKER, OUT, SPEEDS, SRC, STEP1, angle_between, hip_centers, lowpass, read_markers, reconstruct, static_positions

FORCE_THRESH_N = 50.0
MIN_CONTACT_S = 0.08


def force_contacts(path):
    f = pd.read_csv(path, sep="\t")
    fy = lowpass(f[["Fy"]].to_numpy(), FS_FORCE, 50)[:, 0]
    copz = f["COPz"].to_numpy(float)
    on = fy > FORCE_THRESH_N
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    starts, ends = edges[on[edges]], edges[~on[edges]]
    out = []
    for s in starts:
        e = ends[ends > s]
        if len(e) and (e[0] - s) / FS_FORCE >= MIN_CONTACT_S:
            out.append((s, e[0], float(np.nanmedian(copz[s:e[0]]))))
    return out, fy


def process_trial(subject, speed_code, mass, stat):
    mpath = SRC / f"RBDS{subject:03d}runT{speed_code}markers.txt"
    fpath = SRC / f"RBDS{subject:03d}runT{speed_code}forces.txt"
    if not mpath.exists() or not fpath.exists():
        return []
    dyn = read_markers(mpath)
    n = len(next(iter(dyn.values())))
    m = reconstruct(dyn, stat, n)
    need = ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS", "R.Knee", "R.Knee.Medial", "L.Knee", "L.Knee.Medial",
            "R.Ankle", "R.Ankle.Medial", "L.Ankle", "L.Ankle.Medial", "R.Heel.Bottom", "L.Heel.Bottom", "R.MT1", "R.MT5", "L.MT1", "L.MT5"]
    if any(k not in m or np.isfinite(m[k]).all(axis=1).mean() < 0.8 for k in need):
        return []
    pelvis_missing = max(1 - (np.isfinite(dyn[k]).all(axis=1).mean() if k in dyn else 0) for k in ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"])
    m = {k: lowpass(v, FS_MARKER, 10) for k, v in m.items() if k in need}
    hjc = dict(zip("RL", hip_centers(m, stat)))
    kjc = {s: (m[f"{s}.Knee"] + m[f"{s}.Knee.Medial"]) / 2 for s in "RL"}
    ajc = {s: (m[f"{s}.Ankle"] + m[f"{s}.Ankle.Medial"]) / 2 for s in "RL"}
    heel = {s: m[f"{s}.Heel.Bottom"] for s in "RL"}
    toe = {s: (m[f"{s}.MT1"] + m[f"{s}.MT5"]) / 2 for s in "RL"}
    pelvis = (m["R.ASIS"] + m["L.ASIS"] + m["R.PSIS"] + m["L.PSIS"]) / 4
    # 脚の長さ（静止姿勢）
    s_h = dict(zip("RL", hip_centers({k: stat[k][None, :] for k in ["R.ASIS", "L.ASIS", "R.PSIS", "L.PSIS"]}, stat)))
    leg = {s: float(np.linalg.norm(s_h[s][0] - (stat[f"{s}.Ankle"] + stat[f"{s}.Ankle.Medial"]) / 2)) for s in "RL"}

    contacts, _ = force_contacts(fpath)
    rows = []
    ratio = FS_MARKER / FS_FORCE
    for i, (fs, fe, copz) in enumerate(contacts[:-1]):
        ms, me = int(round(fs * ratio)), int(round(fe * ratio))
        mn = int(round(contacts[i + 1][0] * ratio))
        if me >= n or mn >= n or me - ms < 5:
            continue
        mid = int(round(ms + 0.5 * (me - ms)))
        dz = {s: abs(copz - np.nanmean(heel[s][ms:me, 2])) for s in "RL"}
        side = min(dz, key=dz.get)
        other = "L" if side == "R" else "R"
        if abs(dz["R"] - dz["L"]) < 20:  # 圧力中心が左右の真ん中で判定できない
            continue
        lat = 1.0 if side == "R" else -1.0  # 立脚側の外側の向き（Z）

        def rear(p):  # 後方: (Z, Y)
            return p[..., [2, 1]]

        def sagittal(p):  # 側方: (X, Y)
            return p[..., [0, 1]]

        # 後方
        pr, pl = m[f"{side}.PSIS"][mid], m[f"{other}.PSIS"][mid]
        cpd = np.degrees(np.arctan2(pr[1] - pl[1], abs(pr[2] - pl[2])))
        hadd = angle_between(rear(hjc[other][mid] - hjc[side][mid]), rear(kjc[side][mid] - hjc[side][mid]))
        h, k, a = rear(hjc[side][mid]), rear(kjc[side][mid]), rear(ajc[side][mid])
        dev = 180 - angle_between(h - k, a - k)
        # 膝が股関節と足首を結ぶ線より外側なら正
        t = np.dot(k - h, a - h) / np.dot(a - h, a - h)
        proj = h + t * (a - h)
        ka = dev * np.sign(lat * (k[0] - proj[0])) if abs(k[0] - proj[0]) > 1e-9 else 0.0
        # 側方
        def kflex(fr):
            return 180 - angle_between(sagittal(hjc[side][fr]) - sagittal(kjc[side][fr]), sagittal(ajc[side][fr]) - sagittal(kjc[side][fr]))
        shank = np.degrees(np.arctan2(kjc[side][ms][0] - ajc[side][ms][0], kjc[side][ms][1] - ajc[side][ms][1]))
        foot = np.degrees(np.arctan2(toe[side][ms][1] - heel[side][ms][1], toe[side][ms][0] - heel[side][ms][0]))
        overstride = (heel[side][ms][0] - pelvis[ms][0]) / leg[side]
        osc = float(np.nanmax(pelvis[ms:mn, 1]) - np.nanmin(pelvis[ms:mn, 1]))
        step_s = (contacts[i + 1][0] - fs) / FS_FORCE
        contact_s = (fe - fs) / FS_FORCE
        rows.append({"subject": subject, "speed_ms": SPEEDS[speed_code], "side": side, "step": i,
                     "CPD": cpd, "HADD": float(hadd), "KA": float(ka), "KF": float(kflex(mid)),
                     "knee_flex_contact": float(kflex(ms)), "shank_angle_contact": float(shank), "foot_angle_contact": float(foot),
                     "overstride": float(overstride), "pelvis_vertical_osc": osc,
                     "contact_s": contact_s, "flight_s": step_s - contact_s, "step_s": step_s,
                     "cadence_spm": 60 / step_s, "duty": contact_s / step_s,
                     "force_start": int(fs), "force_end": int(fe), "leg_length_mm": leg[side], "pelvis_marker_missing": pelvis_missing})
    return rows


def process_subject(subject, mass):
    spath = SRC / f"RBDS{subject:03d}static.txt"
    if not spath.exists():
        return []
    stat = static_positions(spath)
    rows = []
    for code in SPEEDS:
        try:
            rows += process_trial(subject, code, mass, stat)
        except Exception as e:  # 1試行の失敗で全体を止めない
            print(f"RBDS{subject:03d} T{code}: {e}")
    return rows


def main():
    subj = pd.read_csv(STEP1 / "fukuchi_subjects.csv")
    rows = Parallel(n_jobs=12)(delayed(process_subject)(int(r.subject), float(r.Mass)) for r in subj.itertuples())
    steps = pd.DataFrame([x for r in rows for x in r])
    steps.to_csv(OUT / "marker_steps.csv", index=False)
    feats = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride",
             "pelvis_vertical_osc", "contact_s", "flight_s", "step_s", "cadence_spm", "duty", "leg_length_mm"]
    g = steps.groupby(["subject", "speed_ms", "side"])
    agg = g[feats].median()
    agg["n_steps"] = g.size()
    agg["pelvis_marker_missing"] = g.pelvis_marker_missing.first()
    agg = agg.reset_index()

    # 品質の確認
    qc_cols = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride", "pelvis_vertical_osc"]
    med = agg[qc_cols].median()
    mad = (agg[qc_cols] - med).abs().median() * 1.4826
    robust_z = ((agg[qc_cols] - med) / mad).abs()
    agg["qc_few_steps"] = agg.n_steps < 20
    agg["qc_outlier"] = (robust_z > 5).any(axis=1)
    agg["qc_pelvis_missing"] = agg.pelvis_marker_missing > 0.5
    agg["qc_exclude"] = agg.qc_few_steps | agg.qc_outlier | agg.qc_pelvis_missing
    agg.to_csv(OUT / "marker_features.csv", index=False)

    # 左右の割り当てが交互になっているか（連続する歩で、左右が入れ替わっている割合）
    alt = []
    for (s, v), d in steps.groupby(["subject", "speed_ms"]):
        d = d.sort_values("step")
        consecutive = (d.step.diff() == 1).to_numpy()
        changed = (d.side != d.side.shift()).to_numpy()
        if consecutive.sum():
            alt.append(float(changed[consecutive].mean()))
    lines = [
        "== マーカーから計算した動画相当の値 ==",
        f"歩: {len(steps)}、人×速度×脚: {len(agg)}、人: {agg.subject.nunique()}（全39名）",
        f"処理できなかった人: {sorted(set(subj.subject) - set(agg.subject))}",
        f"人×速度ごとの歩数: 中央値 {g.size().groupby(level=[0, 1]).sum().median():.0f}",
        f"左右の割り当てが交互になっている割合（連続する歩）: 中央値 {np.median(alt):.2f}、最小 {np.min(alt):.2f}",
        "",
        "除外（人×速度×脚）: " + ", ".join(f"{int(r.subject)}番 {r.speed_ms} m/s {r.side}（{'歩数' if r.qc_few_steps else ''}{'外れ値' if r.qc_outlier else ''}{'骨盤マーカー欠け' if r.qc_pelvis_missing else ''}）" for r in agg[agg.qc_exclude].itertuples()),
        f"残り: {int((~agg.qc_exclude).sum())} 件（{agg[~agg.qc_exclude].subject.nunique()} 名）",
        "",
        "値の分布（除外後、人×速度×脚の中央値、速度別の平均 ± SD）:",
        agg[~agg.qc_exclude].groupby("speed_ms")[feats[:-1]].agg(["mean", "std"]).round(2).T.to_string(),
    ]
    text = "\n".join(lines)
    (OUT / "marker_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
