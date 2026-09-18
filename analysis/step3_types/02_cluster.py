"""ステップ3-1：リズムのタイプの分け方を比べ、ケガの結果を見る前に決めた基準で採用する分け方を選ぶ。

入力: outputs/rhythm_features.csv
出力: outputs/rhythm_cluster_selection.csv  すべての分け方の評価指標
      outputs/rhythm_assignments.csv        各組み合わせで採用した分け方での、各人のタイプ
      outputs/rhythm_models.json            採用した分け方の中身（標準化の平均と SD、中心、閾値など）

分析の対象: 走り方の値を埋めたと思われる2名を除く140名
分け方: k-means（k=2〜5）、混合ガウスモデル（共分散 full / diag、k=2〜5）、ルール（A: ピッチ×Duty factor の中央値で4タイプ、C: ピッチの3等分）
採用の基準（ケガの結果は見ない）:
  1. 全タイプで安定性（ブートストラップ200回の Jaccard 係数の平均）が 0.6 以上
  2. 全タイプで 20 名以上
  3. 1・2 を満たす k-means・混合ガウスモデルのうち、シルエット係数が最も高いもの（差が 0.02 以内ならタイプの数が少ないほう）
  4. 満たすものがなければ、ルールによる分け方を採用する
"""
import json

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture

from common import MIN_JACCARD, MIN_TYPE_SIZE, N_BOOT, OUT, SEED, VARIANTS


# ------------------------------------------------------------------ 分け方
class Scaler:
    def __init__(self, X: np.ndarray):
        self.mean = X.mean(axis=0)
        self.sd = X.std(axis=0, ddof=1)

    def __call__(self, X: np.ndarray) -> np.ndarray:
        return (X - self.mean) / self.sd


def fit_method(method: str, k: int, Xraw: np.ndarray, cols: list, n_init: int):
    """分け方を当てはめ、(学習データのラベル, 新しい点のラベルを返す関数, 保存用の中身) を返す。"""
    sc = Scaler(Xraw)
    Z = sc(Xraw)
    if method == "kmeans":
        m = KMeans(n_clusters=k, n_init=n_init, random_state=SEED).fit(Z)
        return m.labels_, (lambda X: m.predict(sc(X))), {"centers_z": m.cluster_centers_.tolist(), **_sc(sc)}
    if method.startswith("gmm_"):
        cov = method.split("_")[1]
        m = GaussianMixture(n_components=k, covariance_type=cov, n_init=max(1, n_init // 5), random_state=SEED, reg_covar=1e-4).fit(Z)
        return m.predict(Z), (lambda X: m.predict(sc(X))), {
            "means_z": m.means_.tolist(), "covariances": m.covariances_.tolist(), "weights": m.weights_.tolist(),
            "covariance_type": cov, "bic": float(m.bic(Z)), **_sc(sc)}
    if method == "rule_2x2":
        ic, idu = cols.index("cadence_10"), cols.index("duty_10")
        tc, td = np.median(Xraw[:, ic]), np.median(Xraw[:, idu])
        f = lambda X: (2 * (X[:, ic] > tc) + (X[:, idu] > td)).astype(int)  # noqa: E731
        return f(Xraw), f, {"thresholds": {"cadence_10": float(tc), "duty_10": float(td)},
                            "rule": "type = 2*(cadence>thr) + (duty>thr)", **_sc(sc)}
    if method == "rule_tertile":
        ic = cols.index("cadence_10")
        q1, q2 = np.quantile(Xraw[:, ic], [1 / 3, 2 / 3])
        f = lambda X: ((X[:, ic] > q1).astype(int) + (X[:, ic] > q2).astype(int))  # noqa: E731
        return f(Xraw), f, {"thresholds": {"cadence_10": [float(q1), float(q2)]}, "rule": "tertiles of cadence", **_sc(sc)}
    raise ValueError(method)


def _sc(sc: Scaler) -> dict:
    return {"scaler_mean": sc.mean.tolist(), "scaler_sd": sc.sd.tolist()}


def candidates(variant: str):
    if variant == "C":
        return [("rule_tertile", 3)]
    out = [(m, k) for m in ("kmeans", "gmm_full", "gmm_diag") for k in range(2, 6)]
    out.append(("rule_2x2", 4))
    return out


# ------------------------------------------------------------------ 評価
def stability(method, k, Xraw, cols, labels, rng) -> np.ndarray:
    """clusterboot と同じ考え方: 人を復元抽出して分け直し、元のタイプと最も重なるタイプとの Jaccard 係数を平均する。"""
    n = len(Xraw)
    types = np.unique(labels)
    scores = np.zeros((N_BOOT, len(types)))
    for b in range(N_BOOT):
        idx = rng.choice(n, n, replace=True)
        try:
            lab_b, _, _ = fit_method(method, k, Xraw[idx], cols, n_init=10)
        except Exception:
            scores[b] = np.nan
            continue
        uniq, first_pos = np.unique(idx, return_index=True)
        lb = lab_b[first_pos]
        lo = labels[uniq]
        for j, t in enumerate(types):
            A = lo == t
            best = 0.0
            for t2 in np.unique(lb):
                B = lb == t2
                union = (A | B).sum()
                if union:
                    best = max(best, (A & B).sum() / union)
            scores[b, j] = best
    return np.nanmean(scores, axis=0)


def type_names(df: pd.DataFrame, labels: np.ndarray, variant: str) -> dict:
    """タイプの中身（平均）から、説明しやすい名前を付ける。ピッチの低い順に R1, R2, ... と番号を振る。"""
    z = (df[["cadence_10", "duty_10"]] - df[["cadence_10", "duty_10"]].mean()) / df[["cadence_10", "duty_10"]].std()
    info = []
    for t in np.unique(labels):
        m = labels == t
        cz, dz = z.cadence_10[m].mean(), z.duty_10[m].mean()
        pitch = "ハイピッチ" if cz > 0.4 else "ローピッチ" if cz < -0.4 else "ピッチ中くらい"
        duty = "接地が長め" if dz > 0.4 else "接地が短め" if dz < -0.4 else "接地の割合ふつう"
        name = f"{pitch}・{duty}"
        if variant == "B":
            rf = df.rearfoot_10[m].mean()
            name += "・踵接地が多い" if rf > 0.7 else "・踵接地が少ない" if rf < 0.3 else ""
            if df.alt_strike[m].mean() > 0.5:
                name += "・左右で接地が違う"
        info.append((cz, t, name))
    info.sort()
    return {int(t): {"id": f"R{i + 1}", "name": name} for i, (_, t, name) in enumerate(info)}


def main():
    feats = pd.read_csv(OUT / "rhythm_features.csv")
    main_df = feats[feats.imputed_running == 0].reset_index(drop=True)
    rng = np.random.default_rng(SEED)

    rows, chosen = [], {}
    for variant, cols in VARIANTS.items():
        Xraw = main_df[cols].to_numpy(float)
        Z = (Xraw - Xraw.mean(0)) / Xraw.std(0, ddof=1)
        for method, k in candidates(variant):
            labels, _, params = fit_method(method, k, Xraw, cols, n_init=50)
            sizes = np.bincount(labels, minlength=k)
            n_types = int((sizes > 0).sum())
            sil = silhouette_score(Z, labels) if n_types > 1 else np.nan
            jac = stability(method, k, Xraw, cols, labels, rng)
            row = {"variant": variant, "method": method, "k": k, "n_types_used": n_types,
                   "silhouette": sil, "bic": params.get("bic", np.nan),
                   "min_size": int(sizes[sizes > 0].min()), "min_jaccard": float(np.nanmin(jac)), "mean_jaccard": float(np.nanmean(jac))}
            row["meets_size"] = row["min_size"] >= MIN_TYPE_SIZE and n_types == k
            row["meets_stability"] = row["min_jaccard"] >= MIN_JACCARD
            rows.append(row)
            print(f"{variant} {method:13s} k={k} シルエット {sil:.3f} 最小人数 {row['min_size']:3d} 安定性(最小) {row['min_jaccard']:.2f}")
    sel = pd.DataFrame(rows)

    # 採用する分け方を決める（ケガの結果は使わない）
    sel["selected"] = False
    for variant, cols in VARIANTS.items():
        v = sel[(sel.variant == variant)]
        ok = v[v.meets_size & v.meets_stability & ~v.method.str.startswith("rule")]
        if variant == "C":
            pick = v.iloc[0]
            reason = "設計どおり、ピッチの3等分（1項目なのでクラスタリングはしない）"
        elif len(ok):
            best = ok.silhouette.max()
            near = ok[ok.silhouette >= best - 0.02].sort_values(["k", "silhouette"], ascending=[True, False])
            pick = near.iloc[0]
            reason = "基準を満たす k-means・混合ガウスモデルのうち、シルエット係数が最も高い（0.02 以内ならタイプの数が少ないほう）"
        else:
            pick = v[v.method.str.startswith("rule")].iloc[0]
            reason = "基準（安定性 0.6 以上・全タイプ 20 名以上）を満たす k-means・混合ガウスモデルがないため、ルールによる分け方を採用"
        sel.loc[pick.name, "selected"] = True
        chosen[variant] = {"method": pick.method, "k": int(pick.k), "reason": reason}

    sel.to_csv(OUT / "rhythm_cluster_selection.csv", index=False)

    # 採用した分け方で、全員（埋めた2名を含む）にタイプを付ける
    assign = feats[["person_id", "imputed_running"]].copy()
    models = {}
    for variant, c in chosen.items():
        cols = VARIANTS[variant]
        Xraw = main_df[cols].to_numpy(float)
        labels, predict, params = fit_method(c["method"], c["k"], Xraw, cols, n_init=50)
        names = type_names(main_df, labels, variant)
        all_labels = predict(feats[cols].to_numpy(float))
        assign[f"type_{variant}"] = [names[int(t)]["id"] if int(t) in names else f"X{int(t)}" for t in all_labels]
        models[variant] = {"features": cols, **c, "types": {names[t]["id"]: {"label": t, "name": names[t]["name"]} for t in names}, "params": params}
        print(f"\n[{variant}] 採用: {c['method']} k={c['k']} … {c['reason']}")
        for t, nm in sorted(names.items(), key=lambda x: x[1]["id"]):
            print(f"  {nm['id']} {nm['name']}: {(labels == t).sum()} 名")
    assign.to_csv(OUT / "rhythm_assignments.csv", index=False)
    (OUT / "rhythm_models.json").write_text(json.dumps(models, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
