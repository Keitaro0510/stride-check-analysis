"""ステップ3-3：予測に使わない前提で、見やすく説明しやすいタイプの分け方を探索する。

入力: outputs/rhythm_features.csv（140名）, outputs/form_features.csv（81名・137脚、重み付き）
出力: outputs/explore_rhythm_candidates.csv, explore_form_candidates.csv, explore_candidates.json

候補の作り方（どれも「分布図に使う2軸 = タイプを作る項目」になるようにする）:
  grid    : 2軸（または1軸）を参加者の順位で区切る（2区分 = 中央値、3区分 = 3等分）。境界は軸に平行な直線
  kmeans  : 2軸だけで k-means（k=2〜8）。境界は斜めの直線（領域として描ける）
  gmm     : 2軸だけで混合ガウスモデル（full、k=2〜6）。リズムのみ（フォームは重みを使えないため）
軸:
  リズム : ピッチ（cadence_10）、Duty factor（duty_10）、接地の仕方（rearfoot_10: 1=踵接地、0=それ以外）
  フォーム: 骨盤の傾き（CPD）、膝の屈曲（KF）、脚のアライメント（ALIGN = 股関節の内転の値と膝の外反の、重み付き標準化の平均。
           相関0.64。値が大きいほど「股関節の内転が小さく膝が外に開く」、小さいほど「内転が大きく膝が内に入る」と解釈、未確定）
評価（ケガとの関係は使わない）:
  安定性     : 人単位のブートストラップ200回で、区切りやクラスタを作り直したときの Jaccard 係数（最小と平均）
  人数       : 最も小さいタイプの割合（重み付き）
  偏り       : タイプの大きさの均等さ（正規化エントロピー、1 = 均等）
  シルエット : 2軸を標準化した空間で
  境界       : 軸に平行（grid）か斜め（kmeans/gmm）か
  動画       : 精度が未確認の項目（膝の外反を含む ALIGN）を使うか
絞り込み（見た目の比較に出す候補）:
  1. 安定性の最小 ≥ 0.6、最小タイプ ≥ 10%、タイプの数 3〜9
  2. その中から、2軸の区切り（grid）の上位2つ、k-means の上位1つ、1軸の区切りの上位1つ（参考）。順位は安定性の平均 → 偏りの小ささ
  （最初は「grid を優先して安定性の高い順に3つ」にしたが、フォームで1軸の区切りばかりが残り、分布図で見比べられなかったので変更）
"""
import json

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

from common import OUT, SEED

N_BOOT = 200
LABELS = {
    "cadence_10": ("ピッチ", ["ローピッチ", "ミドルピッチ", "ハイピッチ"], ["ローピッチ", "ハイピッチ"]),
    "duty_10": ("Duty factor", ["接地が短め", "接地ふつう", "接地が長め"], ["接地が短め", "接地が長め"]),
    "rearfoot_10": ("接地の仕方", None, ["踵以外で接地", "踵で接地"]),
    "CPD": ("骨盤の傾き", ["骨盤が安定", "骨盤ふつう", "骨盤が落ちる"], ["骨盤が安定", "骨盤が落ちる"]),
    "KF": ("膝の屈曲", ["膝の曲がりが浅い", "膝の曲がりふつう", "膝の曲がりが深い"], ["膝の曲がりが浅い", "膝の曲がりが深い"]),
    "ALIGN": ("脚のアライメント", ["膝が内に入る", "脚まっすぐ", "膝が外に開く"], ["膝が内に入る", "膝が外に開く"]),
}


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = (np.cumsum(w[o]) - 0.5 * w[o]) / w.sum()
    return float(np.interp(q, cw, x[o]))


# ------------------------------------------------------------------ 分け方
def fit_grid(df, w, spec):
    """spec = [(feature, n_bins), ...]。区切りの閾値とラベルを返す。"""
    thr = {}
    for f, nb in spec:
        x = df[f].to_numpy(float)
        if f == "rearfoot_10":
            thr[f] = [0.5]
        else:
            thr[f] = [wquantile(x, w, q) for q in np.linspace(0, 1, nb + 1)[1:-1]]
    return thr


def apply_grid(df, spec, thr):
    code = np.zeros(len(df), int)
    mult = 1
    for f, nb in reversed(spec):
        b = np.searchsorted(thr[f], df[f].to_numpy(float), side="right")
        code += b * mult
        mult *= nb
    return code


def grid_name(spec, code):
    parts = []
    for f, nb in reversed(spec):
        b = code % nb
        code //= nb
        names = LABELS[f][1] if nb == 3 else LABELS[f][2]
        parts.append(names[b])
    return "×".join(reversed(parts))


def fit_cluster(method, k, Z, w, seed, n_init):
    if method == "kmeans":
        return KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z, sample_weight=w)
    return GaussianMixture(n_components=k, covariance_type="full", n_init=max(1, n_init // 5), random_state=seed, reg_covar=1e-3).fit(Z)


def zscale(df, w, feats):
    X = df[feats].to_numpy(float)
    m = np.average(X, axis=0, weights=w)
    s = np.sqrt(np.average((X - m) ** 2, axis=0, weights=w))
    return m, np.where(s == 0, 1, s)


def cluster_name(center_z, feats):
    parts = []
    for f, z in zip(feats, center_z):
        three = LABELS[f][1]
        if f == "rearfoot_10":
            parts.append("踵で接地が多い" if z > 0.5 else "踵以外が多い" if z < -0.5 else None)
        else:
            parts.append(three[2] if z > 0.5 else three[0] if z < -0.5 else None)
    parts = [p for p in parts if p]
    return "×".join(parts) if parts else "平均的"


class Candidate:
    def __init__(self, cid, method, feats, k=None, spec=None):
        self.cid, self.method, self.feats, self.k, self.spec = cid, method, feats, k, spec

    def fit(self, df, w, seed=SEED, n_init=50):
        """(ラベル, 保存用の中身) を返す。ラベルは 0..K-1。"""
        if self.method == "grid":
            thr = fit_grid(df, w, self.spec)
            return apply_grid(df, self.spec, thr), {"thresholds": thr, "spec": self.spec}
        m, s = zscale(df, w, self.feats)
        Z = (df[self.feats].to_numpy(float) - m) / s
        model = fit_cluster(self.method, self.k, Z, w, seed, n_init)
        lab = model.predict(Z)
        centers = model.cluster_centers_ if self.method == "kmeans" else model.means_
        return lab, {"scaler_mean": m.tolist(), "scaler_sd": s.tolist(), "centers_z": centers.tolist()}

    def n_types(self):
        return int(np.prod([nb for _, nb in self.spec])) if self.method == "grid" else self.k


def rhythm_candidates():
    c = []
    grids = [
        [("cadence_10", 3), ("rearfoot_10", 2)], [("cadence_10", 2), ("rearfoot_10", 2)],
        [("duty_10", 3), ("rearfoot_10", 2)], [("cadence_10", 3), ("duty_10", 3)],
        [("cadence_10", 2), ("duty_10", 2)], [("cadence_10", 3), ("duty_10", 2)],
        [("cadence_10", 2), ("duty_10", 3)], [("cadence_10", 3)],
    ]
    for g in grids:
        c.append(Candidate("grid:" + "×".join(f"{f}{n}" for f, n in g), "grid", [f for f, _ in g], spec=g))
    for k in range(2, 9):
        c.append(Candidate(f"kmeans:cadence×duty k={k}", "kmeans", ["cadence_10", "duty_10"], k=k))
    for k in range(2, 7):
        c.append(Candidate(f"gmm:cadence×duty k={k}", "gmm", ["cadence_10", "duty_10"], k=k))
    return c


def form_candidates():
    c = []
    grids = [
        [("CPD", 3), ("KF", 3)], [("CPD", 2), ("KF", 2)], [("CPD", 3), ("KF", 2)], [("CPD", 2), ("KF", 3)],
        [("CPD", 3), ("ALIGN", 3)], [("CPD", 2), ("ALIGN", 2)], [("ALIGN", 3), ("KF", 2)], [("ALIGN", 3), ("KF", 3)],
        [("ALIGN", 3)], [("CPD", 3)],
    ]
    for g in grids:
        c.append(Candidate("grid:" + "×".join(f"{f}{n}" for f, n in g), "grid", [f for f, _ in g], spec=g))
    for pair in (["CPD", "KF"], ["CPD", "ALIGN"], ["ALIGN", "KF"]):
        for k in range(2, 9):
            c.append(Candidate(f"kmeans:{pair[0]}×{pair[1]} k={k}", "kmeans", pair, k=k))
    return c


# ------------------------------------------------------------------ 評価
def jaccard_match(orig, boot, units):
    out = []
    lo, lb = orig[units], boot
    for t in np.unique(lo):
        A = lo == t
        best = 0.0
        for t2 in np.unique(lb):
            B = lb == t2
            u = (A | B).sum()
            if u:
                best = max(best, (A & B).sum() / u)
        out.append(best)
    return np.array(out)


def evaluate(cand, df, w, pids):
    labels, params = cand.fit(df, w)
    K = cand.n_types()
    sizes = pd.Series(w).groupby(labels).sum().reindex(range(K), fill_value=0).to_numpy()
    share = sizes / sizes.sum()
    used = share[share > 0]
    entropy = float(-(used * np.log(used)).sum() / np.log(K)) if K > 1 else 1.0
    m, s = zscale(df, w, cand.feats)
    Z = (df[cand.feats].to_numpy(float) - m) / s
    sil = float(np.average(silhouette_samples(Z, labels), weights=w)) if len(np.unique(labels)) > 1 else np.nan

    rng = np.random.default_rng(SEED)
    persons = np.unique(pids)
    rows_of = {p: np.flatnonzero(pids == p) for p in persons}
    jac = []
    for b in range(N_BOOT):
        pick = rng.choice(persons, len(persons), replace=True)
        rows = np.concatenate([rows_of[p] for p in pick])
        bd, bw = df.iloc[rows].reset_index(drop=True), w[rows]
        try:
            lb, _ = cand.fit(bd, bw, seed=SEED + b, n_init=10)
        except Exception:
            continue
        uniq, first = np.unique(rows, return_index=True)
        jac.append(jaccard_match(labels, lb[first], uniq))
    J = np.nanmean(np.array([j if len(j) == len(np.unique(labels)) else np.full(len(np.unique(labels)), np.nan) for j in jac]), axis=0)
    names = {}
    for t in range(K):
        if cand.method == "grid":
            names[t] = grid_name(cand.spec, t)
        else:
            names[t] = cluster_name(np.array(params["centers_z"])[t], cand.feats)
    uses_ka = "ALIGN" in cand.feats
    return {
        "id": cand.cid, "method": cand.method, "features": "×".join(cand.feats), "n_types": K,
        "n_types_used": int((sizes > 0).sum()), "min_share": float(used.min()), "balance": entropy,
        "silhouette": sil, "min_jaccard": float(np.nanmin(J)), "mean_jaccard": float(np.nanmean(J)),
        "boundary": "軸に平行" if cand.method == "grid" else "斜め", "uses_knee_valgus": uses_ka,
        "type_names": " / ".join(f"{names[t]}（{share[t]:.0%}）" for t in range(K) if share[t] > 0),
    }, labels, params, names


def shortlist(tab):
    ok = tab[(tab.min_jaccard >= 0.6) & (tab.min_share >= 0.10) & tab.n_types_used.between(3, 9) & (tab.n_types_used == tab.n_types)].copy()
    ok = ok.sort_values(["mean_jaccard", "balance"], ascending=[False, False])
    two_axis = ok.features.str.contains("×")
    picks = list(ok[(ok.method == "grid") & two_axis].index[:2]) + list(ok[ok.method == "kmeans"].index[:1]) + list(ok[(ok.method == "grid") & ~two_axis].index[:1])
    return picks, ok


def run(dataset, df, w, pids, cands):
    results = Parallel(n_jobs=12)(delayed(evaluate)(c, df, w, pids) for c in cands)
    tab = pd.DataFrame([r[0] for r in results])
    picks, ok = shortlist(tab)
    tab["passes_filter"] = tab.index.isin(ok.index)
    tab["shortlisted"] = tab.index.isin(picks)
    tab["shortlist_rank"] = tab.index.map({p: i + 1 for i, p in enumerate(picks)})
    tab.to_csv(OUT / f"explore_{dataset}_candidates.csv", index=False)
    defs = {}
    for i, c in enumerate(cands):
        r, labels, params, names = results[i]
        defs[c.cid] = {"method": c.method, "features": c.feats, "k": c.n_types(), "params": params,
                       "type_names": {int(t): n for t, n in names.items()}, "labels": labels.tolist(),
                       "shortlist_rank": int(tab.loc[i, "shortlist_rank"]) if tab.loc[i, "shortlisted"] else None}
    return tab, defs


def main():
    r = pd.read_csv(OUT / "rhythm_features.csv")
    r = r[r.imputed_running == 0].reset_index(drop=True)
    tab_r, defs_r = run("rhythm", r, np.ones(len(r)), r.person_id.to_numpy(), rhythm_candidates())

    f = pd.read_csv(OUT / "form_features.csv")
    w = f.weight.to_numpy()
    zh = (f.HADD - np.average(f.HADD, weights=w)) / np.sqrt(np.average((f.HADD - np.average(f.HADD, weights=w)) ** 2, weights=w))
    zk = (f.KA - np.average(f.KA, weights=w)) / np.sqrt(np.average((f.KA - np.average(f.KA, weights=w)) ** 2, weights=w))
    f["ALIGN"] = (zh + zk) / 2
    f.to_csv(OUT / "explore_form_data.csv", index=False)
    tab_f, defs_f = run("form", f, w, f.pid.to_numpy(), form_candidates())

    (OUT / "explore_candidates.json").write_text(json.dumps({"rhythm": defs_r, "form": defs_f}, ensure_ascii=False, indent=1), encoding="utf-8")
    pd.set_option("display.width", 250)
    cols = ["id", "n_types", "min_share", "balance", "silhouette", "min_jaccard", "mean_jaccard", "passes_filter", "shortlist_rank"]
    for name, tab in (("リズム", tab_r), ("フォーム", tab_f)):
        print(f"\n== {name} ==")
        print(tab[cols].round(2).to_string(index=False))
        for _, row in tab[tab.shortlisted].sort_values("shortlist_rank").iterrows():
            print(f"  候補{int(row.shortlist_rank)} {row.id}: {row.type_names}")


if __name__ == "__main__":
    main()
