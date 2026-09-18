"""ステップ5：予測モデルの比較と論文の再現の結果をまとめる（outputs/prediction_report.md と図）。

決め方（結果を見る前に決めたもの）:
  1. 変数そのまま（R2・F2）とタイプ（R1・F1）の AUC の差の95%信頼区間
     - 下限 > 0 → 変数そのままが良い / 上限 < 0 → タイプが良い / 0 を含む → 同程度 → 変数そのままを採用
  2. 採用したモデルの AUC の95%信頼区間の下限が 0.5 を超えるか
     - 超える → 画面に予測の値を幅つきで出す / 超えない → 「動画の項目だけでは予測の精度は低い」と明記し、参考として出す
  リズムの AUC は52週時点の AUC（打ち切りを考慮）を使う。C-index も併記する。
"""
import pandas as pd

from ev_common import OUT, plt, setup_font

NPJ_TYPE, NPJ_CONT = "R1 タイプ（組み合わせB、3タイプ）", "R2 変数そのまま（直線）"
LOH_TYPE, LOH_CONT = "F1 タイプ（組み合わせA、2タイプ）", "F2 変数そのまま（4つの角度）"


def decide(met, diffs, type_name, cont_name, auc_key):
    d = diffs[(diffs.comparison == f"{cont_name} − {type_name}") & (diffs.metric == auc_key)].iloc[0]
    if d.lo > 0:
        which, reason = cont_name, "変数そのままのほうが AUC が高い（差の信頼区間の下限 > 0）"
    elif d.hi < 0:
        which, reason = type_name, "タイプのほうが AUC が高い（差の信頼区間の上限 < 0）"
    else:
        which, reason = cont_name, "AUC の差の信頼区間が 0 を含む（同程度）ので、決め方どおり変数そのままを採用"
    a = met[(met.model == which) & (met.metric == auc_key)].iloc[0]
    useful = a.lo > 0.5
    return {"diff": d, "chosen": which, "reason": reason, "auc": a, "useful": useful}


def fmt(r, digits=3):
    return f"{r.est:.{digits}f}（{r.lo:.{digits}f}〜{r.hi:.{digits}f}）"


def plot_comparison(npj, loh):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    for ax, (met, key, title) in zip(axes, [(npj, "auc_52w", "リズム（npj 2026、140名）：52週時点の AUC"),
                                           (loh, "auc", "フォーム（Loh 2025、81名）：重み付き AUC")]):
        m = met[met.metric == key].reset_index(drop=True)
        for i, r in m.iterrows():
            color = "#888" if "予測なし" in r.model else "#C47A12" if "タイプ" in r.model else "#1F5F8B"
            ax.plot([r.lo, r.hi], [i, i], color=color, lw=2)
            ax.plot(r.est, i, "o", color=color)
        ax.axvline(0.5, color="#B4443A", ls="--", lw=1)
        ax.set_yticks(range(len(m)))
        ax.set_yticklabels(m.model, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlim(0.2, 0.9)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("AUC と95%信頼区間（破線 0.5 = 当てずっぽう）")
        ax.grid(axis="x", color="#ddd")
    fig.tight_layout()
    fig.savefig(OUT / "model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(cal, pred_col, obs_col, path, title):
    models = [m for m in cal.model.unique()]
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for m in models:
        g = cal[cal.model == m]
        ax.plot(g[pred_col], g[obs_col], "o-", label=m, alpha=0.8)
    lim = [0, 1]
    ax.plot(lim, lim, color="#888", ls="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("予測したケガの割合（3等分したグループの平均）")
    ax.set_ylabel("実際のケガの割合")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_font()
    npj, npj_d = pd.read_csv(OUT / "npj_metrics.csv"), pd.read_csv(OUT / "npj_metric_diffs.csv")
    loh, loh_d = pd.read_csv(OUT / "loh_metrics.csv"), pd.read_csv(OUT / "loh_metric_diffs.csv")
    npj_s, loh_s = pd.read_csv(OUT / "npj_spread.csv"), pd.read_csv(OUT / "loh_spread.csv")
    npj_c, loh_c = pd.read_csv(OUT / "npj_calibration.csv"), pd.read_csv(OUT / "loh_calibration.csv")
    rep = pd.read_csv(OUT / "paper_replication.csv")
    plot_comparison(npj, loh)
    plot_calibration(npj_c, "pred_mean", "observed_km", OUT / "npj_calibration.png", "リズム（npj 2026）：52週以内のケガの割合のキャリブレーション")
    plot_calibration(loh_c, "pred_mean", "observed", OUT / "loh_calibration.png", "フォーム（Loh 2025）：12か月のケガの割合のキャリブレーション")

    dn = decide(npj, npj_d, NPJ_TYPE, NPJ_CONT, "auc_52w")
    dl = decide(loh, loh_d, LOH_TYPE, LOH_CONT, "auc")

    L = ["# ステップ5 結果：予測モデルの比較と、論文の AUC の再現", "", "## 結論", ""]
    for label, dd, spread in (("リズム（npj 2026）", dn, npj_s), ("フォーム（Loh 2025）", dl, loh_s)):
        s = spread[spread.model == dd["chosen"]].iloc[0]
        L += [f"### {label}",
              f"- **予測の値に使うモデル：{dd['chosen']}**。{dd['reason']}（AUC の差 {dd['diff']['diff']:+.3f}［{dd['diff']['lo']:+.3f}〜{dd['diff']['hi']:+.3f}］）",
              f"- 採用したモデルの AUC：{fmt(dd['auc'])}。" + ("**当てずっぽう（0.5）より良い**ので、画面に予測の値を幅つきで出す" if dd["useful"] else "**信頼区間が 0.5 を含み、当てずっぽうより良いとは言えない**。画面では「動画の項目だけでは予測の精度は低い」と明記し、予測の値は参考として幅つきで出す"),
              f"- 予測の広がり（10%点〜90%点）：{s.p10:.0%}〜{s.p90:.0%}。人による違いはこの範囲に収まる", ""]

    r3 = npj_d[(npj_d.comparison == "R3 ＋ケガ歴 − R2 変数そのまま（直線）") & (npj_d.metric == "auc_52w")].iloc[0]
    r4 = npj_d[(npj_d.comparison == "R4 ＋ケガ歴＋練習量＋性別 − R2 変数そのまま（直線）") & (npj_d.metric == "auc_52w")].iloc[0]
    f3 = loh_d[(loh_d.comparison == "F3 ＋性別・年齢・週の走行距離 − F2 変数そのまま（4つの角度）") & (loh_d.metric == "auc")].iloc[0]
    L += ["### ケガ歴・練習量を足した場合",
          f"- リズム：ケガ歴を足すと AUC は {r3['diff']:+.3f}（{r3.lo:+.3f}〜{r3.hi:+.3f}）、さらに練習量と性別を足すと {r4['diff']:+.3f}（{r4.lo:+.3f}〜{r4.hi:+.3f}）で、**良くならず、むしろ下がった**",
          f"- フォーム：性別・年齢・週の走行距離を足すと AUC は {f3['diff']:+.3f}（{f3.lo:+.3f}〜{f3.hi:+.3f}）で、同じく下がった",
          "- この参加者では、ケガ歴や参加時の練習量にも予測の力がほとんどなく、項目を増やすと雑音が増えて悪くなったと考えられる（npj 2026 は参加者の76%が過去1年にケガをしていて、ケガ歴で差が付きにくい）。「ケガ歴や練習量を足せば予測できる」とは、このデータからは言えない", ""]

    rS = rep[rep.features == "根拠の強い39項目（class 1）"].set_index("split")
    L += ["### 論文の AUC の再現（npj 2026、週ごとの行）",
          f"- 論文と同じ「行をランダムに10分割」では AUC {rS.loc['行をランダムに10分割（論文と同じ）', 'mean']:.3f}、**人ごとに10分割すると {rS.loc['人ごとに10分割', 'mean']:.3f}**（根拠の強い39項目、ランダムフォレスト）",
          "- 同じ人の別の週が学習用とテスト用の両方に入ると、その人の特徴（遺伝子型や体格など、週で変わらない値）を覚えるだけで当たりやすくなる。論文の AUC は、この影響で高めに出ていた可能性が高い", ""]

    L += ["## リズム（npj 2026、140名）", "", "人ごとの5分割 × 20回。罰則付き Cox 回帰の罰則の強さは学習データの中で決めた。", "",
          "![](model_comparison.png)", "", "| モデル | C-index | 52週時点の AUC | 52週時点の Brier スコア |", "|---|---|---|---|"]
    for m in npj.model.unique():
        g = npj[npj.model == m].set_index("metric")
        L.append(f"| {m} | {fmt(g.loc['c_index'])} | {fmt(g.loc['auc_52w'])} | {fmt(g.loc['brier_52w'])} |")
    L += ["", "| 比較 | 指標 | 差（95%CI） |", "|---|---|---|"]
    for r in npj_d.itertuples():
        L.append(f"| {r.comparison} | {r.metric} | {r.diff:+.3f}（{r.lo:+.3f}〜{r.hi:+.3f}） |")
    L += ["", "| モデル | 予測の10%点 | 中央値 | 90%点 |", "|---|---|---|---|"]
    for r in npj_s.itertuples():
        L.append(f"| {r.model} | {r.p10:.0%} | {r.p50:.0%} | {r.p90:.0%} |")
    L += ["", "![](npj_calibration.png)", ""]

    L += ["## フォーム（Loh 2025、81名・137脚）", "", "人ごとの5分割（ケガの比率をそろえる）× 50回。脚の重み = 1 ÷ その人の脚の数。", "",
          "| モデル | AUC | Brier スコア | キャリブレーションの傾き | 切片 |", "|---|---|---|---|---|"]
    for m in loh.model.unique():
        g = loh[loh.model == m].set_index("metric")
        # 予測なしは分割の中で全員同じ値なので、傾きに意味がない
        slope = fmt(g.loc["calib_slope"], 2) if pd.notna(g.loc["calib_slope", "est"]) and "予測なし" not in m else "—"
        L.append(f"| {m} | {fmt(g.loc['auc'])} | {fmt(g.loc['brier'])} | {slope} | {fmt(g.loc['calib_intercept'], 2)} |")
    L += ["", "| 比較 | 指標 | 差（95%CI） |", "|---|---|---|"]
    for r in loh_d.itertuples():
        L.append(f"| {r.comparison} | {r.metric} | {r.diff:+.3f}（{r.lo:+.3f}〜{r.hi:+.3f}） |")
    L += ["", "| モデル | 予測の10%点 | 中央値 | 90%点 |", "|---|---|---|---|"]
    for r in loh_s.itertuples():
        L.append(f"| {r.model} | {r.p10:.0%} | {r.p50:.0%} | {r.p90:.0%} |")
    L += ["", "- キャリブレーションの傾きは1が理想。1より小さいと、予測が極端すぎる（過学習）", "", "![](loh_calibration.png)", ""]

    L += ["## 論文の AUC の再現（npj 2026、週ごと6,181行、ランダムフォレスト、5回繰り返し）", "",
          "| 入力の項目 | 評価の分け方 | 各分割の AUC の平均 ± SD | テスト用の予測をまとめた AUC |", "|---|---|---|---|"]
    for r in rep.itertuples():
        L.append(f"| {r.features} | {r.split} | {r.mean:.3f} ± {r.std:.3f} | {r.pooled_oof_auc:.3f} |")
    L += ["", "- 論文の値：ランダムフォレスト 0.781（class 1）、0.784（all features）。論文は項目の選択を全データで行っているが、ここでは選択せず全部使った", ""]

    L += ["## 限界", "",
          "- 人数が少なく（リズム：ケガ105名、フォーム：ケガ26名）、AUC の信頼区間は広い",
          "- タイプのモデルは、交差検証の学習データごとにタイプを作り直している（方法とタイプの数はステップ3で採用したものに固定）",
          "- フォームの膝の屈曲だけのモデルは、ステップ3-2の結果を見てから選んだ項目なので、楽観的に出る",
          "- C-index と AUC は、交差検証の同じ分割の中の組だけで計算した（全分割をまとめると、分割ごとの予測の水準のずれで不当に低く出るため）",
          "- 外部のデータでの検証はしていない", ""]
    (OUT / "prediction_report.md").write_text("\n".join(L), encoding="utf-8")
    print("\n".join(L[:22]))


if __name__ == "__main__":
    main()
