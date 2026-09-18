"""npj 2026 の 0〜1 に正規化された値を、元の単位に戻せるかを調べる。

正規化は列ごとの一次変換（x_norm = (x - min) / (max - min)）なので、x = a + b * x_norm の a, b が分かれば戻せる。

方法1（表1の平均と標準偏差）:
  補足資料の表1に参加時の平均±SD（142名、男女別と全体）がある項目は、
  b = SD_raw / SD_norm, a = mean_raw - b * mean_norm で求まる。
  男女別の値でも同じ a, b になるかで、妥当性を確かめる。

方法2（床反力のタイミングの物理的な関係）:
  補足資料の定義では、1歩 = 接地 + 滞空。したがって
    duty factor = 接地時間 / (接地時間 + 滞空時間)
    ピッチ[歩/分] = 60 / (接地時間 + 滞空時間)
  この2つの関係だけでは「時間を k 倍、ピッチを 1/k 倍」にしても成り立つので、絶対値が決まらない。
  そこで Fukuchi ほか 2017（ランナー39名、2.5/3.5/4.5 m/s）の平均を速度で内挿した値を基準（アンカー）にする。
    時速12km: Duty factor は表1から戻した値を使い、ピッチの平均だけを Fukuchi に合わせる
    時速10km: ピッチと Duty factor の平均と SD を Fukuchi に合わせる（Duty factor の絶対値の手がかりがないため）
  基準に使っていない接地時間・滞空時間の平均と SD を Fukuchi と比べて、妥当性を確かめる。

出力: outputs/npj_denorm_params.csv, outputs/npj_person_raw.csv, outputs/npj_denorm_check.txt
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
REPORT = OUT

# 補足資料 表1（Supplementary Table 1）。(男性 n=78, 女性 n=64, 全体 n=142) の (平均, SD)
TABLE1 = {
    "Age": ((32.0, 9.7), (30.9, 9.9), (31.5, 9.8)),
    "Mass": ((72.0, 8.6), (55.8, 6.2), (64.7, 11.1)),
    "BMI": ((22.3, 2.2), (20.4, 1.9), (21.4, 2.2)),
    "lower_limb_days_total": ((42.4, 91.4), (54.1, 91.7), (47.7, 91.4)),
    "EDEQ_total": ((0.46, 0.60), (0.65, 0.85), (0.55, 0.72)),
    "hip_abduction_peak_torque": ((116.4, 28.0), (87.9, 21.3), (103.6, 28.9)),
    "knee_extension_peak_torque": ((160.0, 38.5), (109.8, 25.6), (137.6, 41.6)),
    "knee_flexion_peak_torque": ((89.7, 20.6), (63.1, 13.2), (77.8, 22.1)),
    "navicular_drop": ((0.50, 0.20), (0.51, 0.25), (0.50, 0.23)),
    "Q_angle": ((12.0, 3.8), (15.0, 4.5), (13.3, 4.4)),
    "Impact_peak_12": ((2.66, 0.23), (2.66, 0.23), (2.66, 0.23)),
    "Duty_factor_12": ((0.63, 0.05), (0.61, 0.05), (0.62, 0.05)),
}
MALE, FEMALE = 0, 1  # prepare_npj.py の出力で sex=0 が78名（表1の男性78名と一致）


def fit_from_table1(p: pd.DataFrame, col: str):
    (mm, ms), (fm, fs), (am, asd) = TABLE1[col]
    x = p[col]
    rows = {}
    for label, sub, (m, s) in [("all", x, (am, asd)), ("male", x[p.sex == MALE], (mm, ms)), ("female", x[p.sex == FEMALE], (fm, fs))]:
        sd_n = sub.std(ddof=1)
        if sd_n == 0:
            rows[label] = (np.nan, np.nan)
            continue
        b = s / sd_n
        rows[label] = (m - b * sub.mean(), b)
    a, b = rows["all"]
    # 男女別に求めた a, b と全体の a, b のずれ（全体の値の幅に対する割合）
    rng = b  # 正規化前の値の幅（max - min）
    dev = max(abs(rows[k][0] - a) / rng + abs(rows[k][1] - b) / rng for k in ("male", "female")) if rng else np.nan
    return a, b, dev


def fukuchi_anchor(speed_kmh: float) -> dict:
    st = pd.read_csv(OUT / "fukuchi_steps.csv")
    g = st.groupby("speed_ms")[["contact_s", "flight_s", "cadence_spm", "duty_factor"]].agg(["mean", "std"])
    v = speed_kmh / 3.6
    xs = g.index.to_numpy()
    return {f"{c}_{s}": float(np.interp(v, xs, g[(c, s)].to_numpy())) for c, s in g.columns}


def fit_timing(p: pd.DataFrame, speed: int, duty_raw: np.ndarray | None, anchor: dict):
    ct, ft, sf, df = (p[f"{n}_{speed}"].to_numpy() for n in ("Contact_time", "Flight_time", "Step_frequency", "Duty_factor"))

    def unpack(theta):
        return theta[0:2], theta[2:4], theta[4:6], theta[6:8]

    def resid(theta):
        (ac, bc), (af, bf), (as_, bs), (ad, bd) = unpack(theta)
        c, f, s = ac + bc * ct, af + bf * ft, as_ + bs * sf
        d = ad + bd * df if duty_raw is None else duty_raw
        step = c + f
        r1 = (c / step - d)                 # duty factor の関係
        r2 = (s * step / 60.0 - 1.0)        # ピッチ × 1歩の時間 = 60
        n = len(ct)
        r3 = [np.sqrt(n) * (s.mean() / anchor["cadence_spm_mean"] - 1.0)]        # ピッチの平均を Fukuchi に合わせる
        if duty_raw is None:
            # 時速10kmは Duty factor の絶対値の手がかりがないので、ピッチと Duty factor の平均と SD を Fukuchi に合わせる
            r3.append(np.sqrt(n) * (d.mean() - anchor["duty_factor_mean"]) / anchor["duty_factor_std"])
            r3.append(np.sqrt(n) * (s.std() / anchor["cadence_spm_std"] - 1.0))
            r3.append(np.sqrt(n) * (d.std() / anchor["duty_factor_std"] - 1.0))
        return np.concatenate([r1, r2, r3])

    x0 = np.array([0.15, 0.15, 0.05, 0.15, 150, 50, 0.5, 0.2])
    res = least_squares(resid, x0, bounds=([0, 0.01, 0, 0.01, 60, 1, 0, 0.01], [1, 1, 1, 1, 250, 150, 1, 1]))
    (ac, bc), (af, bf), (as_, bs), (ad, bd) = unpack(res.x)
    r = resid(res.x)
    n = len(ct)
    r = r[: 2 * n]
    return {
        f"Contact_time_{speed}": (ac, bc), f"Flight_time_{speed}": (af, bf),
        f"Step_frequency_{speed}": (as_, bs), f"Duty_factor_{speed}": (ad, bd),
    }, float(np.sqrt(np.mean(r[:n] ** 2))), float(np.sqrt(np.mean(r[n:] ** 2)))


def main():
    p = pd.read_csv(OUT / "npj_person.csv")
    lines, params = [], []

    lines.append("== 方法1：表1の平均±SDから戻す ==")
    for col in TABLE1:
        if col not in p.columns:
            continue
        a, b, dev = fit_from_table1(p, col)
        params.append({"column": col, "a": a, "b": b, "method": "table1", "check": dev})
        lo, hi = a, a + b
        lines.append(f"{col:28s} 元の範囲 {lo:9.3f} 〜 {hi:9.3f}   男女別とのずれ {dev:.2f}")

    lines.append("")
    lines.append("== 方法2：タイミングの物理的な関係から戻す ==")
    duty12 = next(x for x in params if x["column"] == "Duty_factor_12")
    duty12_raw = duty12["a"] + duty12["b"] * p["Duty_factor_12"].to_numpy()
    for speed, duty in ((12, duty12_raw), (10, None)):
        anc = fukuchi_anchor(speed)
        fitted, r_duty, r_step = fit_timing(p, speed, duty, anc)
        lines.append(f"時速{speed}km: 残差 duty factor {r_duty:.3f}, ピッチ×1歩の時間 {r_step:.3%}")
        lines.append("  Fukuchi の内挿値（基準と確認用）: " + ", ".join(f"{k} {v:.3f}" for k, v in anc.items()))
        for col, (a, b) in fitted.items():
            if col == "Duty_factor_12":
                continue
            raw = a + b * p[col]
            params.append({"column": col, "a": a, "b": b, "method": "timing_physics", "check": r_duty if "Duty" in col else r_step})
            lines.append(f"  {col:22s} 元の範囲 {a:8.3f} 〜 {a + b:8.3f}   参加者の平均 {raw.mean():8.3f} ± {raw.std():.3f}")

    lines.append("")
    lines.append("== 戻せない項目 ==")
    done = {x["column"] for x in params}
    rest = [c for c in p.columns if c.endswith(("_10", "_12")) and c not in done] + ["Alt_strike"]
    lines.append("表1にも物理的な関係にもない（パーセンタイルで扱う）: " + ", ".join(rest))

    pd.DataFrame(params).to_csv(OUT / "npj_denorm_params.csv", index=False)
    raw = p.copy()
    for x in params:
        raw[x["column"]] = x["a"] + x["b"] * p[x["column"]]
    raw.to_csv(OUT / "npj_person_raw.csv", index=False)

    text = "\n".join(lines)
    (REPORT / "npj_denorm_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
