"""ステップ3-1：結果のまとめ（outputs/rhythm_report.md）を、成果物の数値から作る。"""
import json

import pandas as pd

from common import OUT, SELECTED_VARIANT, VARIANTS


def pct(x):
    return f"{x * 100:.0f}%"


def main():
    feats = pd.read_csv(OUT / "rhythm_features.csv")
    sel = pd.read_csv(OUT / "rhythm_cluster_selection.csv")
    out = pd.read_csv(OUT / "rhythm_outcomes.csv")
    tests = pd.read_csv(OUT / "rhythm_tests.csv")
    cont = pd.read_csv(OUT / "rhythm_continuous.csv")
    sens = pd.read_csv(OUT / "rhythm_sensitivity.csv")
    power = pd.read_csv(OUT / "rhythm_power.csv")
    models = json.loads((OUT / "rhythm_models.json").read_text(encoding="utf-8"))
    export = json.loads((OUT / "types_rhythm.json").read_text(encoding="utf-8"))
    check = (OUT / "rhythm_export_check.txt").read_text(encoding="utf-8")
    main_out = out[out.analysis == "主な分析"]
    d = feats[feats.imputed_running == 0]

    L = []
    L += ["# ステップ3-1 結果：リズムのタイプ分け（npj 2026）", ""]
    L += ["## 結論", ""]
    SV = SELECTED_VARIANT
    pS = tests[(tests.variant == SV) & tests.test.str.startswith("log-rank")].p.iloc[0]
    hrS = main_out[(main_out.variant == SV) & (main_out.metric.str.contains("調整なし"))]
    typesS = export["variants"][SV]["types"]
    inc_txt = "、".join(f"{t['id']} {t['name']}（{t['n']}名）{pct(t['injury_52w']['est'])}" for t in typesS)
    hr_txt = "、".join(f"{r.type} {r.est:.2f}（{r.lo:.2f}〜{r.hi:.2f}）" for r in hrS.itertuples())
    L += [f"- **画面に使うのは組み合わせ{SV}**（{', '.join(VARIANTS[SV])}、{models[SV]['method']}・{models[SV]['k']}タイプ）。9/17 にユーザーの判断で A から変更した。",
          f"- **どの分け方でも、タイプ間でケガの多さに統計的な差はなかった。** 組み合わせ{SV}の52週以内に初めてケガをした割合：{inc_txt}。"
          f"ハザード比（基準＝{hrS.metric.iloc[0].split('_vs_')[1].split('（')[0]}）：{hr_txt}。log-rank p={pS:.2f}。",
          "- タイプに分けず、ピッチと Duty factor を連続値のまま入れても関係は見えなかった（下表）。ケガ歴・練習量で調整しても、感度分析でも結論は同じ。",
          f"- ただし、この人数（ケガ {int(d.event.sum())} 名）で検出できるのはハザード比が約1.7以上の差だけ。**「差がない」ではなく「大きな差はない」**と解釈する。",
          "- 参加者のケガの割合が非常に高い（52週以内に約84%）ため、タイプによる差が出にくい集団でもある。",
          "- プロトタイプでは、タイプとケガの割合を信頼区間つきで表示し、「タイプ間で差は見られなかった」ことを明記する。", ""]

    L += ["## データ", "",
          f"- npj 2026 の142名のうち、走り方の値を埋めたと思われる2名（ID 70・71。タイミングの全項目が完全に同じで、女性の中央値とほぼ一致）を除いた **{len(d)}名**",
          "  - 論文では「1名を中央値で埋めた」とあるが、データでは2名が該当した。2名とも除き、含めた場合は感度分析で確認した",
          "  - 論文によると、停電で2名は約4か月後に走り方を測定している。誰かは特定できない",
          f"- ケガ（追跡中に1回以上）：{int(d.event.sum())}名。追跡した週数の中央値：{d.n_weeks.median():.0f}週（{d.n_weeks.min()}〜{d.n_weeks.max()}）",
          "- 走り方は参加時に1回だけ測った値（時速10km）。ピッチと Duty factor は、ステップ1で Fukuchi の平均を基準に元の単位に戻した値", ""]

    L += ["## 分け方の選択（ケガの結果を見る前に決めた基準）", "",
          "基準：全タイプで安定性（Jaccard 係数）0.6以上・20名以上 → シルエット係数が最も高いもの（差が0.02以内ならタイプの数が少ないほう）。", "",
          "| 組み合わせ | 方法 | タイプの数 | シルエット係数 | 最小人数 | 安定性（最小） | 採用 |", "|---|---|---|---|---|---|---|"]
    for r in sel.itertuples():
        L.append(f"| {r.variant} | {r.method} | {r.k} | {r.silhouette:.3f} | {r.min_size} | {r.min_jaccard:.2f} | {'✅' if r.selected else ''} |")
    L += ["", "採用した分け方：", ""]
    for v, m in models.items():
        L.append(f"- **{v}**（{', '.join(VARIANTS[v])}）：{m['method']}、{m['k']}タイプ。{m['reason']}")
        for t in export["variants"][v]["types"]:
            mean = ", ".join(f"{k} {x:.3f}" if x < 2 else f"{k} {x:.1f}" for k, x in t["mean_raw"].items())
            L.append(f"  - {t['id']} {t['name']}（{t['n']}名）：平均 {mean}")
    L += ["", "- 組み合わせBは、主に「踵から接地するか」「左右で接地の仕方が違うか」で分かれた（ピッチと Duty factor はタイプ間でほぼ同じ）", ""]

    L += ["## タイプごとのケガの多さ", "", "![](rhythm_forest.png)", "", "![](rhythm_km.png)", "",
          "| 組み合わせ | タイプ | 人数 | ケガ | 52週以内のケガの割合（95%CI） | 100週あたりのケガの週（95%CI） |", "|---|---|---|---|---|---|"]
    for v in VARIANTS:
        for t in export["variants"][v]["types"]:
            i, r = t["injury_52w"], t["injury_weeks_per_100"]
            L.append(f"| {v} | {t['id']} {t['name']} | {t['n']} | {i['events']} | {pct(i['est'])}（{pct(i['lo'])}〜{pct(i['hi'])}） | {r['est']:.1f}（{r['lo']:.1f}〜{r['hi']:.1f}） |")
    L += ["", "### タイプ間の差の検定", "", "| 組み合わせ | 検定 | p |", "|---|---|---|"]
    for r in tests.itertuples():
        L.append(f"| {r.variant} | {r.test} | {r.p:.3f} |")
    L += ["", "### ハザード比（基準＝人数が最も多いタイプ）", "", "| 組み合わせ | タイプ | 調整 | ハザード比（95%CI） | p |", "|---|---|---|---|---|"]
    for r in main_out[main_out.metric.str.startswith("HR")].itertuples():
        L.append(f"| {r.variant} | {r.type} | {r.metric.split('（')[1].rstrip('）')} | {r.est:.2f}（{r.lo:.2f}〜{r.hi:.2f}） | {r.p:.3f} |")
    L += ["", "- 組み合わせBの R1（左右で接地の仕方が違う20名）は、ケガが少ない傾向（ハザード比 約0.55、p≈0.08）だったが、有意ではなく、人数も少ない。複数の比較をしているので偶然の可能性が高い。発表では「探索的な傾向」以上には扱わない", ""]

    L += ["## タイプに分けずに見た場合（連続値の Cox 回帰）", "", "| モデル | 項目 | 1SDあたりのハザード比（95%CI） | p |", "|---|---|---|---|"]
    for r in cont.itertuples():
        if pd.notna(r.HR_per_SD):
            L.append(f"| {r.model} | {r.term} | {r.HR_per_SD:.2f}（{r.lo:.2f}〜{r.hi:.2f}） | {r.p:.3f} |")
        else:
            L.append(f"| {r.model} | {r.term} | — | {r.p:.3f} |")
    L += [""]

    L += [f"## 感度分析（組み合わせ{SV}）", "", "| 分析 | 人数 | タイプごとの52週以内のケガの割合 | log-rank p |", "|---|---|---|---|"]
    for r in sens.itertuples():
        L.append(f"| {r.analysis} | {r.n} | {r.injury_52w_by_type} | {r.p_logrank:.2f} |")
    L += ["", "## 検出できる差の大きさ（有意水準5%、検出力80%）", "", "| 組み合わせ | 比較 | ケガの人数 | 検出できるハザード比 |", "|---|---|---|---|"]
    for r in power.itertuples():
        L.append(f"| {r.variant} | {r.comparison} | {r.events} | {r.detectable_HR:.2f} 以上 |")

    L += ["", "## 判定の準備（ステップ3-3）", "", "- `types_rhythm.json` に、組み合わせ A・B・C の判定に必要な値（標準化の平均と SD、中心や閾値、近いタイプなしの閾値、速度の補正、タイプごとのケガの割合）を書き出した",
          "", "```", check, "```", "",
          f"- Fukuchi のランナー（時速10km相当）を組み合わせ{SV}で判定した分布と、「近いタイプなし」の人数は上のとおり。npj のピッチと Duty factor は Fukuchi の平均を基準に元の単位に戻しているので、平均が近いのは当然で、独立した確認ではない",
          "- 組み合わせBは主に「踵接地か」「左右で接地が違うか」で分かれるので、ピッチと Duty factor の基準を ±1標準誤差ずらしても、判定はほとんど変わらない",
          "- **組み合わせBで判定するには、動画から接地の仕方（踵接地か、左右で違うか）と左右差を測る必要がある**（ステップ2）。左右差は npj では順位（パーセンタイル）でしか持っていないので、動画で測った左右差を Fukuchi のランナーの分布の中での順位に変換して当てはめる（同じような集団だという仮定）",
          "- Fukuchi の確認では、ピッチの左右差が床反力から分からないため中央（50）として判定した", ""]

    L += ["## 限界", "",
          "- ケガはすべての部位・種類をまとめたもの。部位ごとには分けられない",
          "- 走り方は参加時の1回の測定（床反力、時速10km）。動画で測った値とは測り方が違う",
          "- ピッチと Duty factor の絶対値は、Fukuchi の平均を基準に推定したもの",
          "- 参加者は競技志向の持久系ランナーで、ケガの割合が非常に高い。一般のアマチュアランナーにそのまま当てはまるとは限らない",
          "- 人数が限られ、検出できるのは大きな差だけ",
          ""]
    (OUT / "rhythm_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:12]))


if __name__ == "__main__":
    main()
