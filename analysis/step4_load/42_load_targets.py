"""ステップ4-1：負担の指標（予測する目標）を、人×速度×脚ごとに作る。

入力: step1_data/outputs/fukuchi_cycles.parquet（処理済みの平均波形）, fukuchi_subjects.csv,
      outputs/marker_steps.csv（左右を割り当てた接地の区間）, analysis_data/fukuchi2017_running/*forces.txt
出力: outputs/load_targets.csv, outputs/load_targets_check.txt

指標（立脚期 = 処理済みの波形で鉛直床反力が 1 N/kg を超える区間、歩行周期の約0〜35%）:
  （最初は「床反力の値がある区間」にしたが、29〜39番の人は遊脚期にも 0 付近の値が入っていて使えなかった）
  knee_ext_moment   膝のお皿の裏   膝の伸展モーメントのピーク（kneeMomZ の最大、Nm/kg）
  ankle_pf_moment   アキレス腱     足首の底屈モーメントのピーク（ankleMomZ の最大、Nm/kg）
  hip_abd_moment    股関節の外側   股関節の外転モーメントのピーク（hipMomX の最大、Nm/kg）
  vgrf_peak_bw      すね           鉛直床反力のピーク（grfY の最大 ÷ 9.81、体重の倍数）
  loading_rate_bw_s すね           負荷率（体重/秒）。生の床反力（300 Hz、50 Hz の低域通過）から歩ごとに計算して中央値:
                                   立脚期の最初の25%に衝撃のピーク（はっきりした山）があれば、その20%〜80%の平均の傾き（踵接地の一般的な定義）。
                                   なければ、立脚期の25%の時点の力の20%〜80%の平均の傾き（踵接地でない人も同じ尺度で比べるための、この分析での定義）
軸の向きの確認: 角度 Z が屈曲・伸展（膝は立脚期の15%で約44°、遊脚期で約107°）、モーメント Z の正 = 伸展・底屈、X の正 = 外転
（3.5 m/s の平均のピーク: 膝 3.1、足首 2.2、股関節 1.9 Nm/kg で、ランニングの一般的な値の範囲）
品質の確認: 次の範囲を外れる人×速度×脚は除外する（qc_exclude）。36〜39番の人の左脚は、値が1000倍以上ずれ符号も逆で、形式が違うため除外される
  膝の伸展 0.5〜6、足首の底屈 0.5〜5、股関節の外転 0.3〜4 Nm/kg、鉛直床反力のピーク 1.2〜4 体重倍
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from load_common import FS_FORCE, OUT, SRC, STEP1, lowpass


def curve_targets():
    c = pd.read_parquet(STEP1 / "fukuchi_cycles.parquet")
    rows = []
    for (s, v, side), g in c.groupby(["subject", "speed_ms", "side"]):
        w = g.pivot_table(index="pct", columns="variable", values="value")
        if "grfY" not in w or w.grfY.notna().sum() < 10:
            continue
        stance = w[w.grfY > 1.0]
        rows.append({"subject": s, "speed_ms": v, "side": side,
                     "knee_ext_moment": stance.kneeMomZ.max(), "ankle_pf_moment": stance.ankleMomZ.max(),
                     "hip_abd_moment": stance.hipMomX.max(), "vgrf_peak_bw": stance.grfY.max() / 9.81,
                     "stance_pct": float(stance.index.max())})
    return pd.DataFrame(rows)


def loading_rates(mass):
    steps = pd.read_csv(OUT / "marker_steps.csv")
    rows = []
    for (s, v), g in steps.groupby(["subject", "speed_ms"]):
        code = f"{int(round(v * 10))}"
        f = pd.read_csv(SRC / f"RBDS{int(s):03d}runT{code}forces.txt", sep="\t")
        fy = lowpass(f[["Fy"]].to_numpy(), FS_FORCE, 50)[:, 0] / (mass[int(s)] * 9.81)
        for r in g.itertuples():
            seg = fy[r.force_start:r.force_end]
            n = len(seg)
            early = seg[: max(3, int(0.25 * n))]
            peaks, props = find_peaks(early, prominence=0.05)
            rearfoot = len(peaks) > 0
            ref = early[peaks[0]] if rearfoot else seg[min(n - 1, int(0.25 * n))]
            i20 = np.argmax(seg >= 0.2 * ref)
            i80 = np.argmax(seg >= 0.8 * ref)
            if i80 <= i20:
                continue
            lr = (seg[i80] - seg[i20]) / ((i80 - i20) / FS_FORCE)
            rows.append({"subject": s, "speed_ms": v, "side": r.side, "loading_rate_bw_s": lr, "impact_peak_present": rearfoot})
    d = pd.DataFrame(rows)
    return d.groupby(["subject", "speed_ms", "side"]).agg(loading_rate_bw_s=("loading_rate_bw_s", "median"),
                                                        impact_peak_share=("impact_peak_present", "mean")).reset_index()


def main():
    subj = pd.read_csv(STEP1 / "fukuchi_subjects.csv")
    mass = dict(zip(subj.subject, subj.Mass))
    t = curve_targets().merge(loading_rates(mass), on=["subject", "speed_ms", "side"], how="left")
    ok = (t.knee_ext_moment.between(0.5, 6) & t.ankle_pf_moment.between(0.5, 5) & t.hip_abd_moment.between(0.3, 4)
          & t.vgrf_peak_bw.between(1.2, 4))
    t["qc_exclude"] = ~ok
    # 衝撃のピークの検出と、Fukuchi の記録の接地の仕方（3.5 m/s の RFSI/LFSI）の一致
    fsi = subj.set_index("subject")
    t["recorded_rearfoot"] = [fsi.loc[s, f"{side}FSI{int(round(v * 10))}"] == "Rearfoot" if f"{side}FSI{int(round(v * 10))}" in fsi.columns else np.nan
                              for s, v, side in zip(t.subject, t.speed_ms, t.side)]
    t.to_csv(OUT / "load_targets.csv", index=False)
    cols = ["knee_ext_moment", "ankle_pf_moment", "hip_abd_moment", "vgrf_peak_bw", "loading_rate_bw_s", "impact_peak_share", "stance_pct"]
    text = "\n".join([
        "== 負担の指標 ==",
        f"人×速度×脚: {len(t)}、人: {t.subject.nunique()}",
        "除外: " + ", ".join(f"{int(r.subject)}番 {r.speed_ms} m/s {r.side}" for r in t[t.qc_exclude].itertuples()),
        "速度別の平均 ± SD（除外後）:",
        t[~t.qc_exclude].groupby("speed_ms")[cols].agg(["mean", "std"]).round(2).T.to_string(),
        "",
        "衝撃のピークを検出した歩の割合（中央値）× 記録の接地の仕方:",
        t[~t.qc_exclude].groupby("recorded_rearfoot").impact_peak_share.describe().round(2).to_string(),
        "",
        "指標どうしの相関（3.5 m/s）:",
        t[(t.speed_ms == 3.5) & ~t.qc_exclude][cols[:5]].corr().round(2).to_string(),
    ])
    (OUT / "load_targets_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
