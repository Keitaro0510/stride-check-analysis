"""ステップ4-5：部位ごとの負担の推定を、プロトタイプ用の load.json と、結果のまとめ load_report.md に書き出す。

入力: outputs/load_model_metrics.csv, load_noise.csv, load_reference.csv, load_feature_reference.csv, load_cadence_sim.csv,
      load_final_models.json, marker_check.txt, load_targets_check.txt
出力: outputs/load.json, outputs/load_report.md

画面に出す値（結果を見る前に決めたもの）:
  - 推定: L3 のリッジ回帰
  - 平均的なランナーとの差（%）= (推定値 − 基準の平均) ÷ 基準の平均 × 100、誤差の幅 = 1人ずつ除く交差検証の RMSE ÷ 基準の平均 × 100
  - 信頼度の星: L3 リッジの R²_extra（速度・体格を超えて説明できた割合）が 0.3 以上 ★★★、0.1〜0.3 ★★、0.1 未満 ★
    （設計の段階では R² で決める案だったが、R² は速度の違いでほとんど決まってしまい、画面で見せたい「同じ速度での個人差」の精度を表さないので、R²_extra に変更）
  - すねは、鉛直床反力のピークと負荷率のうち R²_extra が高いほうを表示に使う
  - R²_extra が0以下の部位は individual_difference_estimable = false とし、画面では「走り方による違いは推定できない（平均的な値のみ）」と表示する
    （結果を見てから追加した表示の区別。アキレス腱が該当）
  - ピッチのシミュレーション: load_cadence_sim.csv の show_on_screen が真の部位だけ
"""
import json

import pandas as pd

from load_common import OUT

REGIONS = {
    "knee": {"label": "膝のお皿の裏", "targets": ["knee_ext_moment"], "unit": "Nm/kg"},
    "achilles": {"label": "アキレス腱", "targets": ["ankle_pf_moment"], "unit": "Nm/kg"},
    "hip": {"label": "股関節の外側", "targets": ["hip_abd_moment"], "unit": "Nm/kg"},
    "shin": {"label": "すね", "targets": ["vgrf_peak_bw", "loading_rate_bw_s"], "unit": "体重倍 / 体重/秒"},
}
TARGET_LABELS = {"knee_ext_moment": "膝の伸展モーメントのピーク", "ankle_pf_moment": "足首の底屈モーメントのピーク",
                 "hip_abd_moment": "股関節の外転モーメントのピーク", "vgrf_peak_bw": "鉛直床反力のピーク", "loading_rate_bw_s": "負荷率"}


def stars(r2x):
    return 3 if r2x >= 0.3 else 2 if r2x >= 0.1 else 1


def main():
    met = pd.read_csv(OUT / "load_model_metrics.csv")
    noise = pd.read_csv(OUT / "load_noise.csv")
    ref = pd.read_csv(OUT / "load_reference.csv").set_index("target")
    fref = pd.read_csv(OUT / "load_feature_reference.csv")
    sim = pd.read_csv(OUT / "load_cadence_sim.csv")
    final = json.loads((OUT / "load_final_models.json").read_text(encoding="utf-8"))
    l3 = met[(met.feature_set == "L3") & (met.method == "ridge")].set_index("target")

    regions = {}
    for key, info in REGIONS.items():
        cand = info["targets"]
        t = max(cand, key=lambda x: l3.loc[x, "r2_extra"])
        r = l3.loc[t]
        regions[key] = {
            "label": info["label"], "target": t, "target_label": TARGET_LABELS[t],
            "model": final["L3"][t],
            "reference_10kmh": {"mean": float(ref.loc[t, "mean"]), "sd": float(ref.loc[t, "sd"]), "p10": float(ref.loc[t, "p10"]), "p90": float(ref.loc[t, "p90"])},
            "cv": {"r2": float(r.r2), "rmse": float(r.rmse), "rel_rmse": float(r.rel_rmse), "spearman": float(r.spearman),
                   "r2_extra": float(r.r2_extra), "r2_extra_ci95": [float(r.r2_extra_lo), float(r.r2_extra_hi)]},
            "error_pct_of_reference": float(r.rmse / ref.loc[t, "mean"] * 100),
            "stars": stars(r.r2_extra),
            "individual_difference_estimable": bool(r.r2_extra > 0),
            "cadence_sim": [{"cadence_change_pct": int(x.cadence_change_pct), "load_change_pct": float(x.load_change_pct), "ci95": [float(x.lo), float(x.hi)]}
                            for x in sim[(sim.target == t)].itertuples()],
            "show_cadence_sim": bool(sim[sim.target == t].show_on_screen.iloc[0]),
            "alternatives": [x for x in cand if x != t],
        }
    export = {
        "created": "2026-09-17（ステップ4）",
        "source": "Fukuchi ほか 2017（ランナー39名、トレッドミル 2.5/3.5/4.5 m/s、3D動作と床反力、CC BY 4.0）。品質の確認後 36名",
        "display": "平均的なランナー（時速10km相当）との差（%）と誤差の幅、信頼度の星",
        "inputs_note": "入力はマーカーを2Dに投影した動画相当の値。速度は m/s、体重 kg、身長 cm。角度の定義は 41_marker_features.py を参照",
        "limitations": ["参加者のほぼ全員が男性（38/39名）", "36名と少ない", "マーカーから作った値と姿勢推定の値は同じではない（ステップ2で確認）",
                        "負荷率は、衝撃のピークの検出と記録の接地の仕方の一致が弱い", "ピッチのシミュレーションは人どうしの違いから作った推定で、因果ではない"],
        "regions": regions,
        "angle_only_models_for_step5_1": final["Lang"],
    }
    (OUT / "load.json").write_text(json.dumps(export, ensure_ascii=False, indent=1), encoding="utf-8")

    L = ["# ステップ4 結果：部位ごとの負担の推定（Fukuchi 2017）", "", "## 結論", ""]
    for key, rg in regions.items():
        cv = rg["cv"]
        L.append(f"- **{rg['label']}**（{rg['target_label']}）：R²_extra {cv['r2_extra']:.2f}［{cv['r2_extra_ci95'][0]:.2f}〜{cv['r2_extra_ci95'][1]:.2f}］、"
                 f"R² {cv['r2']:.2f}、誤差 {rg['error_pct_of_reference']:.0f}%、信頼度 {'★' * rg['stars']}{'☆' * (3 - rg['stars'])}"
                 + ("" if rg["individual_difference_estimable"] else "、**走り方による違いは推定できない**")
                 + (f"、ピッチ+10%で {next(x for x in rg['cadence_sim'] if x['cadence_change_pct'] == 10)['load_change_pct']:+.1f}%（画面に出す）" if rg["show_cadence_sim"] else "、ピッチのシミュレーションは画面に出さない"))
    L += ["", "R²_extra = 1 − 誤差²(モデル) ÷ 誤差²(速度・体重・身長だけのモデル)。0なら「走り方を見ても、速度と体格から分かる以上のことは分からない」", ""]

    L += ["## データ", "", "```", (OUT / "marker_check.txt").read_text(encoding="utf-8").split("値の分布")[0].strip(), "```", "",
          "```", (OUT / "load_targets_check.txt").read_text(encoding="utf-8").split("速度別の平均")[0].strip(), "```", ""]

    L += ["## マーカーから計算した角度と、Loh 2025 の比較（時速10km相当）", "",
          "| 値 | Fukuchi 平均 ± SD | Loh 2025 ケガなし 平均 ± SD |", "|---|---|---|"]
    for r in fref.itertuples():
        loh = f"{r.loh2025_mean:.1f} ± {r.loh2025_sd:.1f}" if pd.notna(r.loh2025_mean) else "—"
        L.append(f"| {r.feature} | {r.mean:.2f} ± {r.sd:.2f} | {loh} |")
    L += ["", "- 股関節の内転（HADD）を「反対側の股関節への線と太ももの線のなす角」として計算すると約83°で、Loh の約80°に近い値になった（差は Loh の SD の約0.8倍）。この定義では内転が大きいほど値が小さくなるので、ステップ3-2の「値が大きいほど内転が小さい」という解釈を支持する",
          "- 骨盤の傾き・膝の屈曲・膝の外反も、Loh との差は SD の0.5〜0.9倍程度。関節中心の定義（Loh は動画上の目視、ここはマーカーからの推定）の違いによるずれが含まれる", ""]

    L += ["## モデルの比較（1人ずつ除く交差検証）", "", "![](load_model_comparison.png)", "",
          "| 指標 | 入力 | 方法 | R² | 誤差（÷平均） | 順位相関 | R²_extra（95%CI） |", "|---|---|---|---|---|---|---|"]
    for r in met.itertuples():
        L.append(f"| {TARGET_LABELS[r.target]} | {r.feature_set} | {r.method} | {r.r2:.2f} | {r.rel_rmse:.1%} | {r.spearman:.2f} | {r.r2_extra:.2f}（{r.r2_extra_lo:.2f}〜{r.r2_extra_hi:.2f}） |")
    L += ["", "入力：L0 = 速度・体重・身長、L1 = L0＋タイミング、L2 = L0＋角度など、L3 = すべて、Lang = L0＋Loh の4つの角度", ""]

    L += ["## 動画並みの誤差を入力に足した場合（L3 リッジ、R²_extra）", "", "| 指標 | 誤差なし | 誤差1倍 | 誤差2倍 |", "|---|---|---|---|"]
    for t, g in noise.groupby("target", sort=False):
        g = g.set_index("noise_level")
        L.append(f"| {TARGET_LABELS[t]} | {g.loc[0.0, 'r2_extra_mean']:.2f} | {g.loc[1.0, 'r2_extra_mean']:.2f} ± {g.loc[1.0, 'r2_extra_sd']:.2f} | {g.loc[2.0, 'r2_extra_mean']:.2f} ± {g.loc[2.0, 'r2_extra_sd']:.2f} |")
    L += ["", "誤差1倍：角度 3°、オーバーストライド 0.03、骨盤の上下動 10 mm、ピッチ 3 歩/分、Duty factor 0.02、接地時間 0.015 秒（仮の値。ステップ2で置き換える）", ""]

    L += ["## ピッチを変えたときのシミュレーション（平均的なランナー、時速10km）", "",
          "| 指標 | ピッチの変化 | 負担の変化（95%CI） | 先行研究の向き | 画面に出す |", "|---|---|---|---|---|"]
    for r in sim.itertuples():
        L.append(f"| {TARGET_LABELS[r.target]} | +{r.cadence_change_pct}% | {r.load_change_pct:+.1f}%（{r.lo:+.1f}〜{r.hi:+.1f}） | {r.expected_direction} | {'はい' if r.show_on_screen else ''} |")
    L += ["", "- 人どうしの違いから作った推定で、同じ人がピッチを変えた実験ではない", ""]

    L += ["## 平均的なランナー（時速10km相当、2.5 と 3.5 m/s から内挿）", "", "| 指標 | 平均 ± SD | 10%点〜90%点 | 脚の数（人） |", "|---|---|---|---|"]
    for t, r in ref.iterrows():
        L.append(f"| {TARGET_LABELS[t]} | {r['mean']:.2f} ± {r['sd']:.2f} | {r.p10:.2f}〜{r.p90:.2f} | {int(r.n_legs)}（{int(r.n_subjects)}） |")
    L += ["", "## 限界", ""] + [f"- {x}" for x in export["limitations"]] + [""]
    (OUT / "load_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:10]))


if __name__ == "__main__":
    main()
