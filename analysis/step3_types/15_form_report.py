"""ステップ3-2：結果のまとめ（outputs/form_report.md）を、成果物の数値から作る。"""
import json

import pandas as pd

from common import FORM_LABELS, FORM_SELECTED_VARIANT, FORM_VARIANTS, OUT


def pct(x):
    return f"{x * 100:.0f}%"


def main():
    feats = pd.read_csv(OUT / "form_features.csv")
    sel = pd.read_csv(OUT / "form_cluster_selection.csv")
    out = pd.read_csv(OUT / "form_outcomes.csv")
    tests = pd.read_csv(OUT / "form_tests.csv")
    cont = pd.read_csv(OUT / "form_continuous.csv")
    sens = pd.read_csv(OUT / "form_sensitivity.csv")
    power = pd.read_csv(OUT / "form_power.csv")
    scale = pd.read_csv(OUT / "form_scale_check.csv")
    corr = pd.read_csv(OUT / "form_angle_correlations.csv")
    models = json.loads((OUT / "form_models.json").read_text(encoding="utf-8"))
    export = json.loads((OUT / "types_form.json").read_text(encoding="utf-8"))
    main_out = out[out.analysis == "主な分析"]
    SV = FORM_SELECTED_VARIANT
    vS = export["variants"][SV]

    L = ["# ステップ3-2 結果：フォームのタイプ分け（Loh 2025）", "", "## 結論", ""]
    inc_txt = "、".join(f"{t['id']} {t['name']}（{t['n_person_equiv']:.0f}人分）{pct(t['injury_12m']['est'])}（{pct(t['injury_12m']['lo'])}〜{pct(t['injury_12m']['hi'])}）" for t in vS["types"])
    kf = cont[(cont.model == "膝の屈曲だけ")].iloc[0]
    kf_all = cont[(cont.model == "4つの角度を一緒に") & (cont.term == "膝の屈曲")].iloc[0]
    L += [f"- **タイプ間でケガの割合に統計的な差はなかった。** 組み合わせ{SV}（{', '.join(FORM_LABELS[c] for c in FORM_VARIANTS[SV])}、{vS['method']}・{vS['k']}タイプ）の12か月でケガをした割合：{inc_txt}。並べ替え検定 p={vS['p_perm']:.2f}。組み合わせB、調整、感度分析でも同じ。",
          f"- タイプに分けずに見ると、**膝の屈曲**だけ、立脚中期に膝が深く曲がっているほどケガが少ない傾向があった（1SDあたりのオッズ比 {kf.OR_per_SD:.2f}［{kf.lo:.2f}〜{kf.hi:.2f}］、p={kf.p_boot:.3f}。4角度を一緒に入れても {kf_all.OR_per_SD:.2f}［{kf_all.lo:.2f}〜{kf_all.hi:.2f}］）。ただし4つの角度を試した中の1つで、多重比較を考えると偶然の可能性がある（Bonferroni 補正後は p≈{min(1, kf.p_boot * 4):.2f}）。**探索的な傾向**として扱う",
          "- **ケガの前とケガの後で、股関節の内転の値の特徴が違った。** Loh & Kong 2026 の受傷群（ケガの後）は健常群より約0.9SD大きいが、Loh 2025 で後でケガをした人（ケガの前）はほぼ差がない（下の尺度の表）。ケガの後の走り方の違いは、原因ではなく、かばった結果である可能性を示す",
          "- 人数が少なく（ケガ26名）、検出できるのは大きな差だけ（下表）。「差がない」ではなく「大きな差はない」と解釈する", ""]

    L += ["## データと重み", "",
          f"- Loh 2025 の81名、角度がある脚 {len(feats)} 脚（ケガなし55名 110脚、ケガあり26名 27脚）",
          "- ケガをした人は1名を除き片脚だけに角度があるので、脚を単位にし、重み = 1 ÷ その人の脚の数（ケガなしの人の脚は0.5）とした。信頼区間と安定性は人単位のブートストラップ",
          "- 1人1行・重みなし（角度のある脚の平均）で分けた場合は感度分析に示した", ""]

    L += ["## 角度の尺度と向き", "", "![](form_scale.png)", "", "| グループ | 骨盤の傾き | 股関節の内転 | 膝の屈曲 | 膝の外反 |", "|---|---|---|---|---|"]
    for g, s in scale.groupby("group", sort=False):
        s = s.set_index("angle")
        L.append(f"| {g} | " + " | ".join(f"{s.loc[a, 'mean']:.1f} ± {s.loc[a, 'sd']:.1f}（{s.loc[a, 'smd_vs_lk_control']:+.2f}SD）" for a in FORM_VARIANTS["A"]) + " |")
    L += ["", "（ ）内は Loh & Kong 2026 健常群との標準化平均差。膝の外反は健常群の符号の向きが逆の可能性があるので、比較に使えない。", "",
          "| グループ | 脚 | 人数 | 相関（骨盤の傾き・股関節の内転） | 相関（股関節の内転・膝の外反） |", "|---|---|---|---|---|"]
    for r in corr.itertuples():
        L.append(f"| {r.group} | {r.side} | {r.n_limbs} | {r.corr_CPD_HADD:+.2f} | {r.corr_HADD_KA:+.2f} |")
    L += ["",
          "- **Loh & Kong 2026 の健常群だけ、股関節の内転と膝の外反の相関が負**（他は正）。健常群のシートは膝の外反の符号の向きが逆の可能性が高い",
          "- 骨盤の傾きと股関節の内転の値は、すべてのデータで負の相関。反対側の骨盤が落ちるほど立脚側の内転は大きくなるはずなので、**股関節の内転の値は「大きいほど内転が小さい（約90°−内転）」、膝の外反の値は「正のとき膝が外に開く」と解釈してタイプに名前を付けた**（未確定。ステップ2で動画から角度の定義を合わせるときに確かめる）", ""]

    L += ["## 分け方の選択（ケガの結果を見る前に決めた基準）", "",
          "基準：人単位のブートストラップで全タイプの安定性0.6以上・重み付き15人分以上 → 重み付きシルエット係数が最も高いもの（差0.02以内ならタイプの数が少ないほう）→ なければルール。", "",
          "| 組み合わせ | 方法 | タイプの数 | シルエット係数（重み付き） | 最小（人分） | 安定性（最小） | 採用 |", "|---|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        L.append(f"| {r.variant} | {r.method} | {r.k} | {r.silhouette_w:.3f} | {r.min_size_w:.1f} | {r.min_jaccard:.2f} | {'✅' if r.selected else ''} |")
    L += ["", "採用した分け方：", ""]
    for v, m in models.items():
        L.append(f"- **{v}**（{', '.join(FORM_LABELS[c] for c in FORM_VARIANTS[v])}）：{m['method']}、{m['k']}タイプ。{m['reason']}")
        for t in export["variants"][v]["types"]:
            zs = ", ".join(f"{FORM_LABELS[c]} {z:+.2f}SD" for c, z in t["z_means"].items())
            L.append(f"  - {t['id']} {t['name']}（{t['n_person_equiv']:.0f}人分、{t['n_limbs']}脚）：{zs}")
    L += [""]

    L += ["## タイプごとのケガの割合", "", "![](form_forest.png)", "", "![](form_map.png)", "",
          "| 組み合わせ | タイプ | 人分 | 12か月でケガをした割合（95%CI、ブートストラップ） | 参考：Wilson 法 |", "|---|---|---|---|---|"]
    for r in main_out[main_out.metric == "injury_12m"].itertuples():
        L.append(f"| {r.variant} | {r.type} {models[r.variant]['types'][r.type]['name']} | {r.n_person_equiv:.1f} | {pct(r.est)}（{pct(r.lo)}〜{pct(r.hi)}） | {pct(r.wilson_lo)}〜{pct(r.wilson_hi)} |")
    L += ["", "| 組み合わせ | タイプ | 比較（調整） | オッズ比（95%CI） |", "|---|---|---|---|"]
    for r in main_out[main_out.metric.str.startswith("OR")].itertuples():
        L.append(f"| {r.variant} | {r.type} | {r.metric} | {r.est:.2f}（{r.lo:.2f}〜{r.hi:.2f}） |")
    L += ["", "| 組み合わせ | 検定 | p |", "|---|---|---|"]
    for r in tests.itertuples():
        L.append(f"| {r.variant} | {r.test} | {r.p:.3f} |")

    L += ["", "## タイプに分けずに見た場合（重み付きロジスティック回帰、1SDあたり）", "", "| モデル | 角度 | オッズ比（95%CI） | p（ブートストラップ） |", "|---|---|---|---|"]
    for r in cont.itertuples():
        L.append(f"| {r.model} | {r.term} | {r.OR_per_SD:.2f}（{r.lo:.2f}〜{r.hi:.2f}） | {r.p_boot:.3f} |")

    L += ["", f"## 感度分析（組み合わせ{SV}）", "", "| 分析 | 人分 | タイプごとのケガの割合 | 並べ替え検定 p |", "|---|---|---|---|"]
    for r in sens.itertuples():
        L.append(f"| {r.analysis} | {r.n_person_equiv} | {r.injury_12m_by_type} | {r.p_perm:.2f} |")
    L += ["", "- 左脚だけ・右脚だけの分析では、反対側の脚にだけ角度があるケガの人が入らないので、ケガの割合は低めに出る（タイプ間の向きを見るためのもの）", ""]

    L += ["## 検出できる差の大きさ（有意水準5%、検出力80%）", "", "| 組み合わせ | 比較 | 基準のタイプのケガの割合 | 検出できるもう一方の割合 |", "|---|---|---|---|"]
    for r in power.itertuples():
        L.append(f"| {r.variant} | {r.comparison} | {pct(r.ref_risk)} | {pct(r.detectable_risk_lower)} 以下 または {pct(r.detectable_risk_upper)} 以上 |")

    L += ["", "## 判定の準備（ステップ3-3）", "",
          "- `types_form.json` に、脚ごとの判定に必要な値（標準化の平均と SD、タイプの中心、近いタイプなしの閾値、速度の補正、角度の基準の分布、角度の向きの前提）を書き出した",
          "- 速度 1 km/h あたりの角度の変化：" + ", ".join(f"{FORM_LABELS[a]} {s:+.2f}°" for a, s in export["speed_adjust_per_kmh"].items()),
          "- 動画の角度は、Loh の定義と向きに合わせてから当てはめる必要がある（ステップ2）。Fukuchi の3D角度は定義が違うので、ここでは確認に使っていない", ""]

    L += ["## 限界", "",
          "- 81名（ケガ26名）と少なく、大きな差しか検出できない",
          "- ケガの部位・種類・時期は分からない",
          "- ケガをした人は片脚の角度しかなく、それがケガをした側かどうかは未確認",
          "- 股関節の内転と膝の外反の値の向きは、相関から推定したもので未確定",
          "- Loh & Kong 2026 健常群の膝の外反は、符号の向きが逆の可能性がある",
          "- 2D動画の角度は、カメラの位置や撮り方で変わる", ""]
    (OUT / "form_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:9]))


if __name__ == "__main__":
    main()
