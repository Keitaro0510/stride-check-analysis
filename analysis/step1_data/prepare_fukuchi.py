"""Fukuchi ほか 2017（ランナーの走行の生体力学、figshare 4543435, CC BY 4.0）を整理する。

入力: analysis_data/fukuchi2017_running/
  RBDSinfo.txt               参加者情報（1ファイル1行。参加者ごとに重複）
  RBDSxxxprocessed.txt       歩行周期0〜100%で平均した左右の関節角度・モーメント・床反力・パワー（速度 2.5/3.5/4.5 m/s）
  RBDSxxxrunT{25,35,45}forces.txt  トレッドミルの床反力（300 Hz、30秒）
出力:
  outputs/fukuchi_subjects.csv   1人1行
  outputs/fukuchi_cycles.parquet 平均波形（縦持ち: subject, speed_ms, side, pct, variable, value）
  outputs/fukuchi_steps.csv      1人×速度ごとの接地時間・滞空時間・ピッチ・Duty factor・鉛直床反力のピーク
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "analysis_data/fukuchi2017_running"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

FS = 300.0          # 床反力のサンプリング周波数 [Hz]
THRESH_N = 50.0     # 接地の判定に使う鉛直床反力の閾値 [N]
MIN_CONTACT_S = 0.08
SPEEDS = {"25": 2.5, "35": 3.5, "45": 4.5}


def subjects() -> pd.DataFrame:
    info = pd.read_csv(SRC / "RBDSinfo.txt", sep="\t")
    s = info.drop_duplicates("Subject").drop(columns=["FileName"]).rename(columns={"Subject": "subject"})
    s["prior_injury"] = (s["Injury"] == "Yes").astype(int)
    return s.reset_index(drop=True)


def cycles() -> pd.DataFrame:
    frames = []
    for f in sorted(SRC.glob("RBDS*processed.txt")):
        subj = int(re.search(r"RBDS0*(\d+)processed", f.name).group(1))
        d = pd.read_csv(f, sep="\t")
        long = d.melt(id_vars="PercGcycle", var_name="col", value_name="value")
        m = long["col"].str.extract(r"^([LR])(\w+?)(25|35|45)$")
        long = long.assign(side=m[0], variable=m[1], speed_ms=m[2].map(SPEEDS), subject=subj).dropna(subset=["variable"])
        frames.append(long.rename(columns={"PercGcycle": "pct"})[["subject", "speed_ms", "side", "pct", "variable", "value"]])
    return pd.concat(frames, ignore_index=True)


def steps_from_forces(path: Path, mass_kg: float) -> dict:
    fy = pd.read_csv(path, sep="\t")["Fy"].to_numpy()
    on = fy > THRESH_N
    edges = np.flatnonzero(np.diff(on.astype(int))) + 1
    starts = edges[on[edges]]           # 離れていた → 接地
    ends = edges[~on[edges]]            # 接地 → 離地
    contacts = []
    for s in starts:
        e = ends[ends > s]
        if len(e) == 0:
            break
        dur = (e[0] - s) / FS
        if dur >= MIN_CONTACT_S:
            contacts.append((s, e[0]))
    contacts = np.array(contacts)
    if len(contacts) < 10:
        return {}
    contact_s = (contacts[:, 1] - contacts[:, 0]) / FS
    step_s = np.diff(contacts[:, 0]) / FS
    flight_s = (contacts[1:, 0] - contacts[:-1, 1]) / FS
    peaks = np.array([fy[s:e].max() for s, e in contacts]) / (mass_kg * 9.81)
    return {
        "n_steps": len(step_s),
        "contact_s": np.median(contact_s),
        "flight_s": np.median(flight_s),
        "step_s": np.median(step_s),
        "cadence_spm": 60.0 / np.median(step_s),
        "duty_factor": np.median(contact_s[:-1] / step_s),
        "peak_vgrf_bw": np.median(peaks),
        # 左右差: 1歩おきに並ぶ接地時間の奇数番目と偶数番目の差（どちらが左右かは床反力だけでは分からない）
        "contact_asym_pct": abs(np.median(contact_s[0::2]) - np.median(contact_s[1::2])) / np.median(contact_s) * 100,
    }


def steps(subj: pd.DataFrame) -> pd.DataFrame:
    rows = []
    mass = subj.set_index("subject")["Mass"]
    for f in sorted(SRC.glob("RBDS*runT*forces.txt")):
        m = re.search(r"RBDS0*(\d+)runT(\d\d)forces", f.name)
        sid, sp = int(m.group(1)), m.group(2)
        r = steps_from_forces(f, float(mass[sid]))
        if r:
            rows.append({"subject": sid, "speed_ms": SPEEDS.get(sp, int(sp) / 10), **r})
    return pd.DataFrame(rows)


def main():
    s = subjects()
    s.to_csv(OUT / "fukuchi_subjects.csv", index=False)
    c = cycles()
    c.to_parquet(OUT / "fukuchi_cycles.parquet", index=False)
    st = steps(s)
    st.to_csv(OUT / "fukuchi_steps.csv", index=False)

    print(f"参加者: {len(s)} 名（過去のケガあり {s.prior_injury.sum()} 名）")
    print(f"平均波形: {c.subject.nunique()} 名, 変数 {sorted(c.variable.unique())}")
    print("速度ごとの人数（平均波形）:", c.dropna(subset=['value']).groupby('speed_ms').subject.nunique().to_dict())
    print("\n床反力から計算したタイミング（速度ごとの平均 ± SD）:")
    cols = ["contact_s", "flight_s", "cadence_spm", "duty_factor", "peak_vgrf_bw", "contact_asym_pct"]
    print(st.groupby("speed_ms")[cols].agg(["mean", "std"]).round(3).T.to_string())
    print("\n速度ごとの人数（床反力）:", st.groupby("speed_ms").subject.nunique().to_dict())


if __name__ == "__main__":
    main()
