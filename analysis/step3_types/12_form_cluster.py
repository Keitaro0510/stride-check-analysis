"""ステップ3-2：フォームのタイプの分け方を比べ、ケガの結果を見る前に決めた基準で採用する分け方を選ぶ。

入力: outputs/form_features.csv
出力: outputs/form_cluster_selection.csv, outputs/form_assignments.csv, outputs/form_models.json

分け方（脚を単位、重み = 1 ÷ その人の脚の数）:
  kmeans    : 重み付き k-means（k=2〜4）
  gmm_diag  : 混合ガウスモデル（対角、k=2〜3）。重みを使えないので、各人から片脚をランダムに選んだデータで当てはめ、全脚に当てはめる
  rule_q4   : 骨盤の傾きと股関節の内転の「値」がそれぞれ上位25%か → 4タイプ（値の向きの解釈は type_names のコメント参照）
  rule_q3   : どちらでもない／骨盤の傾きが上位25%／股関節の内転の値だけが上位25% → 3タイプ
  rule_med  : 骨盤の傾き×股関節の内転の中央値で4タイプ
採用の基準（ケガの結果は見ない）:
  1. 人単位のブートストラップ200回で、全タイプの安定性（Jaccard 係数）0.6以上
  2. 全タイプで重み付き15人分以上
  3. 1・2を満たす k-means・混合ガウスモデルのうち、重み付きシルエット係数が最も高いもの（差0.02以内ならタイプの数が少ないほう）
  4. なければ、ルール（rule_q4 → rule_q3 → rule_med の順に、2を満たす最初のもの）
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn.mixture import GaussianMixture

from common import FORM_LABELS, FORM_MIN_TYPE_SIZE, FORM_N_BOOT, FORM_VARIANTS, MIN_JACCARD, OUT, SEED

RULES = ["rule_q4", "rule_q3", "rule_med"]


def wquantile(x, w, q):
    o = np.argsort(x)
    cw = np.cumsum(w[o]) / w.sum()
    return float(np.interp(q, cw, x[o]))


class WScaler:
    def __init__(self, X, w):
        self.mean = np.average(X, axis=0, weights=w)
        self.sd = np.sqrt(np.average((X - self.mean) ** 2, axis=0, weights=w) * w.sum() / (w.sum() - 1))

    def __call__(self, X):
        return (X - self.mean) / self.sd


def one_limb_per_person(pids, rng):
    """各人からランダムに1脚を選んだ行番号。"""
    idx = []
    for p in np.unique(pids):
        rows = np.flatnonzero(pids == p)
        idx.append(rng.choice(rows))
    return np.array(idx)


def fit_method(method, k, X, w, pids, cols, n_init=50, seed=SEED):
    """(学習データのラベル, 新しい脚のラベルを返す関数, 保存用の中身) を返す。"""
    sc = WScaler(X, w)
    Z = sc(X)
    base = {"scaler_mean": sc.mean.tolist(), "scaler_sd": sc.sd.tolist()}
    if method == "kmeans":
        m = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(Z, sample_weight=w)
        return m.labels_, (lambda Xn: m.predict(sc(Xn))), {**base, "centers_z": m.cluster_centers_.tolist()}
    if method == "gmm_diag":
        rng = np.random.default_rng(seed)
        idx = one_limb_per_person(pids, rng)
        m = GaussianMixture(n_components=k, covariance_type="diag", n_init=max(1, n_init // 5), random_state=seed, reg_covar=1e-3).fit(Z[idx])
        return m.predict(Z), (lambda Xn: m.predict(sc(Xn))), {**base, "means_z": m.means_.tolist(), "covariances": m.covariances_.tolist(), "weights": m.weights_.tolist()}
    ic, ih = cols.index("CPD"), cols.index("HADD")
    if method in ("rule_q4", "rule_q3"):
        tc, th = wquantile(X[:, ic], w, 0.75), wquantile(X[:, ih], w, 0.75)
    elif method == "rule_med":
        tc, th = wquantile(X[:, ic], w, 0.5), wquantile(X[:, ih], w, 0.5)
    else:
        raise ValueError(method)
    thr = {"CPD": tc, "HADD": th}
    if method == "rule_q3":
        f = lambda Xn: np.where(Xn[:, ic] > tc, 1, np.where(Xn[:, ih] > th, 2, 0))  # noqa: E731
        rule = "0=どちらでもない, 1=CPDが上位25%, 2=HADDの値だけが上位25%"
    else:
        f = lambda Xn: (2 * (Xn[:, ic] > tc) + (Xn[:, ih] > th)).astype(int)  # noqa: E731
        rule = "type = 2*(CPD>thr) + (HADD>thr)"
    return f(X), f, {**base, "thresholds": thr, "rule": rule}


def candidates():
    return [("kmeans", k) for k in (2, 3, 4)] + [("gmm_diag", k) for k in (2, 3)] + [("rule_q4", 4), ("rule_q3", 3), ("rule_med", 4)]


def stability(method, k, X, w, pids, cols, labels, rng):
    """人単位で復元抽出して分け直し、元のタイプと最も重なるタイプとの Jaccard 係数（脚単位）を平均する。"""
    persons = np.unique(pids)
    types = np.unique(labels)
    scores = np.full((FORM_N_BOOT, len(types)), np.nan)
    for b in range(FORM_N_BOOT):
        pick = rng.choice(persons, len(persons), replace=True)
        rows = np.concatenate([np.flatnonzero(pids == p) for p in pick])
        # 同じ人が複数回選ばれたときは別人として扱う
        new_pids = np.concatenate([np.full((pids == p).sum(), i) for i, p in enumerate(pick)])
        try:
            lab_b, _, _ = fit_method(method, k, X[rows], w[rows], new_pids, cols, n_init=10, seed=SEED + b)
        except Exception:
            continue
        uniq, first = np.unique(rows, return_index=True)
        lb, lo = lab_b[first], labels[uniq]
        for j, t in enumerate(types):
            A = lo == t
            best = 0.0
            for t2 in np.unique(lb):
                B = lb == t2
                u = (A | B).sum()
                if u:
                    best = max(best, (A & B).sum() / u)
            scores[b, j] = best
    return np.nanmean(scores, axis=0)


def type_names(df, labels, w):
    """タイプの重み付き平均（標準化）から名前を付ける。骨盤の傾きの小さい順に F1, F2, ... と番号を振る。"""
    cols = ["CPD", "HADD", "KF", "KA"]
    mu = np.average(df[cols], axis=0, weights=w)
    sd = np.sqrt(np.average((df[cols] - mu) ** 2, axis=0, weights=w))
    # 角度の向き（未確定。ステップ2で動画から定義を合わせるときに確かめる）:
    #   3つのデータすべてで CPD と HADD の値が負の相関（-0.2〜-0.43）。反対側の骨盤が落ちるほど立脚側の股関節の内転は
    #   大きくなるはずなので、「HADD の値が大きいほど内転は小さい（約90°−内転）」と解釈する。
    #   HADD と KA の値は正の相関なので、「KA が正のとき膝は外に開く（内反）」と解釈する。
    words = {"CPD": ("骨盤が落ちる", "骨盤が安定"), "HADD": ("股関節の内転が小さい", "股関節の内転が大きい"),
             "KF": ("膝の曲がりが深い", "膝の曲がりが浅い"), "KA": ("膝が外に開く", "膝が内に入る")}
    info = []
    for t in np.unique(labels):
        m = labels == t
        z = (np.average(df[cols][m], axis=0, weights=w[m]) - mu) / sd
        parts = [words[c][0] if zc > 0.5 else words[c][1] if zc < -0.5 else None for c, zc in zip(cols, z)]
        parts = [p for p in parts if p]
        info.append((z[0], int(t), "・".join(parts) if parts else "平均的なフォーム", z.tolist()))
    info.sort()
    return {t: {"id": f"F{i + 1}", "name": name, "z_means": dict(zip(cols, zs))} for i, (_, t, name, zs) in enumerate(info)}


def weighted_silhouette(Z, labels, w):
    if len(np.unique(labels)) < 2:
        return np.nan
    return float(np.average(silhouette_samples(Z, labels), weights=w))


def main():
    d = pd.read_csv(OUT / "form_features.csv")
    w = d.weight.to_numpy()
    pids = d.pid.to_numpy()
    rng = np.random.default_rng(SEED)
    rows, chosen = [], {}
    for variant, cols in FORM_VARIANTS.items():
        X = d[cols].to_numpy(float)
        Z = WScaler(X, w)(X)
        for method, k in candidates():
            labels, _, params = fit_method(method, k, X, w, pids, cols)
            sizes = pd.Series(w).groupby(labels).sum().reindex(range(k), fill_value=0).to_numpy()
            used = int((sizes > 0).sum())
            jac = stability(method, k, X, w, pids, cols, labels, rng)
            row = {"variant": variant, "method": method, "k": k, "n_types_used": used,
                   "silhouette_w": weighted_silhouette(Z, labels, w), "min_size_w": float(sizes[sizes > 0].min()),
                   "min_jaccard": float(np.nanmin(jac)), "mean_jaccard": float(np.nanmean(jac))}
            row["meets_size"] = row["min_size_w"] >= FORM_MIN_TYPE_SIZE and used == k
            row["meets_stability"] = row["min_jaccard"] >= MIN_JACCARD
            rows.append(row)
            print(f"{variant} {method:9s} k={k} シルエット {row['silhouette_w']:.3f} 最小（人分） {row['min_size_w']:5.1f} 安定性(最小) {row['min_jaccard']:.2f}")
    sel = pd.DataFrame(rows)
    sel["selected"] = False
    for variant in FORM_VARIANTS:
        v = sel[sel.variant == variant]
        ok = v[v.meets_size & v.meets_stability & ~v.method.isin(RULES)]
        if len(ok):
            best = ok.silhouette_w.max()
            pick = ok[ok.silhouette_w >= best - 0.02].sort_values(["k", "silhouette_w"], ascending=[True, False]).iloc[0]
            reason = "基準を満たす k-means・混合ガウスモデルのうち、重み付きシルエット係数が最も高い（0.02 以内ならタイプの数が少ないほう）"
        else:
            rules = v[v.method.isin(RULES) & v.meets_size]
            rules = rules.assign(order=rules.method.map({r: i for i, r in enumerate(RULES)})).sort_values("order")
            pick = rules.iloc[0] if len(rules) else v[v.method == "rule_med"].iloc[0]
            reason = f"基準（安定性0.6以上・全タイプ{FORM_MIN_TYPE_SIZE}人分以上）を満たす k-means・混合ガウスモデルがないため、人数の基準を満たす最初のルール（rule_q4 → rule_q3 → rule_med）を採用"
        sel.loc[pick.name, "selected"] = True
        chosen[variant] = {"method": pick.method, "k": int(pick.k), "reason": reason}
    sel.to_csv(OUT / "form_cluster_selection.csv", index=False)

    assign = d[["pid", "side", "injured", "weight"]].copy()
    models = {}
    for variant, c in chosen.items():
        cols = FORM_VARIANTS[variant]
        X = d[cols].to_numpy(float)
        labels, _, params = fit_method(c["method"], c["k"], X, w, pids, cols)
        names = type_names(d, labels, w)
        assign[f"type_{variant}"] = [names[int(t)]["id"] for t in labels]
        models[variant] = {"features": cols, **c, "params": params,
                           "types": {v["id"]: {"label": t, "name": v["name"], "z_means": v["z_means"]} for t, v in names.items()}}
        print(f"\n[{variant}] 採用: {c['method']} k={c['k']} … {c['reason']}")
        for t, v in sorted(names.items(), key=lambda x: x[1]["id"]):
            m = labels == t
            print(f"  {v['id']} {v['name']}: {m.sum()} 脚、{w[m].sum():.1f} 人分  " + ", ".join(f"{FORM_LABELS[k2]} {z:+.2f}SD" for k2, z in v["z_means"].items()))
    assign.to_csv(OUT / "form_assignments.csv", index=False)
    (OUT / "form_models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
