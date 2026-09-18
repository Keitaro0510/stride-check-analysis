"""ステップ3-3：探索したタイプの候補を、境界つきの分布図で並べ、結果のまとめを作る。

入力: outputs/explore_candidates.json, explore_rhythm_candidates.csv, explore_form_candidates.csv, rhythm_features.csv, explore_form_data.csv
出力: outputs/explore_rhythm_gallery.png, explore_form_gallery.png（全候補）
      outputs/explore_rhythm_shortlist.png, explore_form_shortlist.png（絞り込んだ候補、大きめ）
      outputs/explore_report.md
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import OUT, plt, setup_font

ex = importlib.import_module("31_explore_types")
PALETTE = ["#1F5F8B", "#C47A12", "#2E7D5B", "#8A4F9E", "#B4443A", "#3B8EA5", "#7A6A2F", "#D17BA5", "#5A6570"]
AXIS_LABEL = {"cadence_10": "ピッチ（歩/分）", "duty_10": "Duty factor", "rearfoot_10": "接地の仕方",
              "CPD": "骨盤の傾き（°）", "KF": "膝の屈曲（°）", "ALIGN": "脚のアライメント（SD、大きいほど膝が外）"}
SECOND_AXIS = {"cadence_10": "duty_10", "duty_10": "cadence_10", "CPD": "ALIGN", "ALIGN": "CPD", "KF": "CPD"}


def display_axes(d):
    feats = d["features"]
    if len(feats) == 2:
        return feats[0], feats[1]
    return feats[0], SECOND_AXIS[feats[0]]


def draw(ax, df, w, d, big=False):
    xf, yf = display_axes(d)
    labels = np.array(d["labels"])
    x = df[xf].to_numpy(float)
    y = df[yf].to_numpy(float)
    if yf == "rearfoot_10":
        rng = np.random.default_rng(0)
        y = y + rng.uniform(-0.3, 0.3, len(y))
    # k-means の領域
    if d["method"] in ("kmeans", "gmm"):
        m, s = np.array(d["params"]["scaler_mean"]), np.array(d["params"]["scaler_sd"])
        C = np.array(d["params"]["centers_z"])
        gx = np.linspace(x.min() - 0.05 * np.ptp(x), x.max() + 0.05 * np.ptp(x), 300)
        gy = np.linspace(y.min() - 0.05 * np.ptp(y), y.max() + 0.05 * np.ptp(y), 300)
        GX, GY = np.meshgrid(gx, gy)
        Z = np.column_stack([(GX.ravel() - m[0]) / s[0], (GY.ravel() - m[1]) / s[1]])
        lab = np.argmin(((Z[:, None, :] - C[None, :, :]) ** 2).sum(-1), axis=1).reshape(GX.shape)
        ax.contourf(GX, GY, lab, levels=np.arange(-0.5, len(C) + 0.5), colors=PALETTE[:len(C)], alpha=0.12)
        ax.contour(GX, GY, lab, levels=np.arange(0.5, len(C) - 0.5 + 1e-9), colors="#555", linewidths=0.8)
    # 区切りの線
    if d["method"] == "grid":
        for f, nb in d["params"]["spec"]:
            for t in d["params"]["thresholds"][f]:
                if f == "rearfoot_10":
                    ax.axhline(0.5, color="#555", lw=1, ls="--")
                elif f == xf:
                    ax.axvline(t, color="#555", lw=1, ls="--")
                elif f == yf:
                    ax.axhline(t, color="#555", lw=1, ls="--")
    size = np.clip(w, 0.5, 1) * (26 if big else 10)
    for t in np.unique(labels):
        mk = labels == t
        ax.scatter(x[mk], y[mk], s=size[mk], color=PALETTE[int(t) % len(PALETTE)], alpha=0.8, edgecolors="none",
                   label=f"{d['type_names'][str(int(t))] if isinstance(next(iter(d['type_names'])), str) else d['type_names'][int(t)]}")
    ax.set_xlabel(AXIS_LABEL[xf], fontsize=9 if big else 7)
    ax.set_ylabel(AXIS_LABEL[yf], fontsize=9 if big else 7)
    if yf == "rearfoot_10":
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["踵以外", "踵"])
    ax.tick_params(labelsize=8 if big else 6)


def gallery(dataset, df, w, defs, tab, path):
    n = len(tab)
    cols = 5
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3.2, rows * 2.9))
    for ax, (_, r) in zip(axes.ravel(), tab.iterrows()):
        d = defs[r.id]
        draw(ax, df, w, d)
        mark = f"★候補{int(r.shortlist_rank)} " if r.shortlisted else ("" if r.passes_filter else "（基準外）")
        ax.set_title(f"{mark}{r.id}\n安定性 最小{r.min_jaccard:.2f}/平均{r.mean_jaccard:.2f}・最小{r.min_share:.0%}", fontsize=7)
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def shortlist_fig(dataset, df, w, defs, tab, path, title):
    sl = tab[tab.shortlisted].sort_values("shortlist_rank")
    fig, axes = plt.subplots(1, len(sl), figsize=(5.2 * len(sl), 5.6))
    axes = np.atleast_1d(axes)
    for ax, (_, r) in zip(axes, sl.iterrows()):
        draw(ax, df, w, defs[r.id], big=True)
        ax.set_title(f"候補{int(r.shortlist_rank)}：{r.id}\n{r.n_types}タイプ・安定性 平均{r.mean_jaccard:.2f}・境界{r.boundary}", fontsize=9)
        ax.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    setup_font()
    defs = json.loads((OUT / "explore_candidates.json").read_text(encoding="utf-8"))
    r = pd.read_csv(OUT / "rhythm_features.csv")
    r = r[r.imputed_running == 0].reset_index(drop=True)
    f = pd.read_csv(OUT / "explore_form_data.csv")
    tab_r = pd.read_csv(OUT / "explore_rhythm_candidates.csv")
    tab_f = pd.read_csv(OUT / "explore_form_candidates.csv")
    # JSON のキーは文字列になるので、名前の辞書を int キーに戻す
    for block in defs.values():
        for d in block.values():
            d["type_names"] = {int(k): v for k, v in d["type_names"].items()}
    gallery("rhythm", r, np.ones(len(r)), defs["rhythm"], tab_r, OUT / "explore_rhythm_gallery.png")
    gallery("form", f, f.weight.to_numpy(), defs["form"], tab_f, OUT / "explore_form_gallery.png")
    shortlist_fig("rhythm", r, np.ones(len(r)), defs["rhythm"], tab_r, OUT / "explore_rhythm_shortlist.png", "リズムのタイプの候補（npj 2026、140名）")
    shortlist_fig("form", f, f.weight.to_numpy(), defs["form"], tab_f, OUT / "explore_form_shortlist.png", "フォームのタイプの候補（Loh 2025、137脚、点の大きさ＝重み）")

    L = ["# ステップ3-3 結果：見やすいタイプの分け方の探索", "",
         "予測に使わない前提で、「分布図の2軸 = タイプを作る項目」になる分け方を比べた。ケガとの関係は評価に使っていない。", "",
         "- 区切り（grid）：軸ごとに参加者の順位で2区分（中央値）か3区分（3等分）。境界は軸に平行な直線",
         "- k-means：2軸だけでクラスタリング（k=2〜8）。境界は斜めの直線",
         "- 混合ガウスモデル（リズムのみ）：どれも安定性が低く、基準を満たさなかった",
         "- 脚のアライメント：股関節の内転の値と膝の外反の標準化の平均（相関0.64）。大きいほど「股関節の内転が小さく膝が外に開く」と解釈（未確定）",
         "- 絞り込みの基準：安定性の最小 ≥ 0.6、最小タイプ ≥ 10%、3〜9タイプ → 2軸の区切りの上位2つ、k-means の上位1つ、1軸の区切りの上位1つ（安定性の平均 → 偏りの小ささの順）", ""]
    for name, tab, short, gal in (("リズム（npj 2026、140名）", tab_r, "explore_rhythm_shortlist.png", "explore_rhythm_gallery.png"),
                                   ("フォーム（Loh 2025、81名・137脚）", tab_f, "explore_form_shortlist.png", "explore_form_gallery.png")):
        L += [f"## {name}", "", "### 絞り込んだ候補", "", f"![]({short})", "",
              "| 候補 | 分け方 | タイプの数 | 安定性（最小／平均） | 最小タイプ | 境界 | 膝の外反を使う | タイプ（割合） |", "|---|---|---|---|---|---|---|---|"]
        for _, x in tab[tab.shortlisted].sort_values("shortlist_rank").iterrows():
            L.append(f"| {int(x.shortlist_rank)} | {x.id} | {x.n_types} | {x.min_jaccard:.2f}／{x.mean_jaccard:.2f} | {x.min_share:.0%} | {x.boundary} | {'はい' if x.uses_knee_valgus else 'いいえ'} | {x.type_names} |")
        L += ["", "### 全候補", "", f"![]({gal})", "",
              "| 分け方 | タイプの数 | 安定性（最小／平均） | 最小タイプ | 偏り（1=均等） | シルエット係数 | 基準を満たす |", "|---|---|---|---|---|---|---|"]
        for _, x in tab.iterrows():
            L.append(f"| {x.id} | {x.n_types} | {x.min_jaccard:.2f}／{x.mean_jaccard:.2f} | {x.min_share:.0%} | {x.balance:.2f} | {x.silhouette:.2f} | {'✅' if x.passes_filter else ''} |")
        L += [""]
    L += ["## 読み方", "",
          "- 区切り（grid）は、順位で区切るので人数がほぼ均等になり、境界も分布図にまっすぐ引ける。ただし、データにかたまりがあるわけではないので、境界の近くの人は「どちらにも近い」",
          "- k-means は、分布の形に合わせて境界が斜めになる。k を増やすと安定性がすぐ下がった（リズムは k=5 以上、フォームは k=6 以上で基準を下回る）",
          "- 接地の仕方（踵かそれ以外か）は2値なので、リズムの区切りに入れると安定してはっきり分かれる", ""]
    (OUT / "explore_report.md").write_text("\n".join(L), encoding="utf-8")
    print("done")


if __name__ == "__main__":
    main()
