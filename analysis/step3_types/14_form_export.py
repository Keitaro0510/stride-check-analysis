"""ステップ3-2（3-3の準備を含む）：動画で測ったフォームを、脚ごとにタイプに当てはめるための JSON を書き出す。

入力: outputs/form_features.csv, form_models.json, form_outcomes.csv, form_tests.csv, form_scale_check.csv
出力: outputs/types_form.json

判定のしかた（プロトタイプ側で同じ計算をする。左脚・右脚それぞれ）:
  1. 時速10km以外で撮った場合は、速度の補正で時速10km相当に直す（Loh 2025 の速度と角度の重み付き回帰の傾き）
  2. 標準化して一番近いタイプの中心に当てはめる（k-means の場合。ルールなら閾値）
  3. 重み付きの分布からのマハラノビス距離が97.5%点より遠ければ「近いタイプなし」
  動画の角度は、Loh の2D動画解析と同じ定義・向きに合わせてから当てはめる（ステップ2）
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import FORM_LABELS, FORM_SELECTED_VARIANT, FORM_VARIANTS, OUT

fc = importlib.import_module("12_form_cluster")


def main():
    d = pd.read_csv(OUT / "form_features.csv")
    models = json.loads((OUT / "form_models.json").read_text(encoding="utf-8"))
    out = pd.read_csv(OUT / "form_outcomes.csv")
    tests = pd.read_csv(OUT / "form_tests.csv")
    scale = pd.read_csv(OUT / "form_scale_check.csv")
    main_out = out[out.analysis == "主な分析"]
    w = d.weight.to_numpy()

    X = np.column_stack([np.ones(len(d)), d.treadmill_kmh - 10])
    W = np.diag(w)
    slopes = {a: float(np.linalg.solve(X.T @ W @ X, X.T @ W @ d[a].to_numpy())[1]) for a in FORM_VARIANTS["A"]}

    ref = {}
    for g in ["Loh 2025 ケガなし（参加時）", "Loh & Kong 2026 健常群"]:
        s = scale[scale.group == g].set_index("angle")
        ref[g] = {a: {"mean": float(s.loc[a, "mean"]), "sd": float(s.loc[a, "sd"])} for a in FORM_VARIANTS["A"]}

    export = {
        "source": "Loh et al. 2025 Int J Sports Med（レクリエーションランナー81名、12か月の前向き追跡）。フォームは参加時に2D動画（立脚中期、14歩の平均）",
        "unit": "脚ごと（重み = 1 ÷ その人の脚の数）",
        "outcome": "12か月でケガをした人の割合（重み付き、人単位のブートストラップで95%信頼区間）",
        "selected_variant": FORM_SELECTED_VARIANT,
        "angle_labels": FORM_LABELS,
        "angle_direction_assumption": {
            "HADD": "値が大きいほど股関節の内転は小さい（約90°−内転）と解釈。骨盤の傾きとの負の相関から推定、未確定",
            "KA": "正のとき膝が外に開く（内反）と解釈。股関節の内転の値との正の相関から推定、未確定",
            "CPD": "正のとき反対側の骨盤が下がる",
            "KF": "立脚中期の膝の屈曲角（大きいほど深く曲がる）",
        },
        "speed_adjust_per_kmh": slopes,
        "speed_range_kmh": [8, 13.5],
        "reference_distributions": ref,
        "reference_note": "Loh & Kong 2026 健常群の膝の外反（KA）は符号の向きが他と逆の可能性が高いので、KA の基準には Loh 2025 を使う",
        "variants": {},
    }
    for v, m in models.items():
        cols = FORM_VARIANTS[v]
        Xv = d[cols].to_numpy(float)
        mean, sd = np.array(m["params"]["scaler_mean"]), np.array(m["params"]["scaler_sd"])
        Z = (Xv - mean) / sd
        cov = np.cov(Z, rowvar=False, aweights=w)
        inv = np.linalg.pinv(cov)
        dist = np.sqrt(np.einsum("ij,jk,ik->i", Z, inv, Z))
        o = np.argsort(dist)
        thr = float(np.interp(0.975, np.cumsum(w[o]) / w.sum(), dist[o]))
        labels = d[f"type_{v}"] if f"type_{v}" in d else pd.read_csv(OUT / "form_assignments.csv")[f"type_{v}"]
        types = []
        for tid, info in sorted(m["types"].items()):
            mask = (labels == tid).to_numpy()
            r = main_out[(main_out.variant == v) & (main_out.type == tid) & (main_out.metric == "injury_12m")].iloc[0]
            or_rows = main_out[(main_out.variant == v) & (main_out.type == tid) & main_out.metric.str.startswith("OR")]
            types.append({
                "id": tid, "name": info["name"], "label": info["label"],
                "n_person_equiv": float(r.n_person_equiv), "n_limbs": int(r.n_limbs),
                "mean_raw": {c: float(np.average(d[c][mask], weights=w[mask])) for c in cols},
                "z_means": info["z_means"],
                "injury_12m": {"est": float(r.est), "lo": float(r.lo), "hi": float(r.hi)},
                "odds_ratio": {x.metric: {"est": float(x.est), "lo": float(x.lo), "hi": float(x.hi)} for x in or_rows.itertuples()},
            })
        export["variants"][v] = {
            "features": cols, "method": m["method"], "k": m["k"], "params": m["params"], "ood_mahalanobis_975": thr,
            "types": types,
            "overall": {"injury_12m": float(np.average(d.injured, weights=w)), "n": int(d.pid.nunique())},
            "p_perm": float(tests[tests.variant == v].p.iloc[0]),
        }
    SV = FORM_SELECTED_VARIANT
    export["caution"] = (f"タイプ間でケガの割合に統計的な差はなかった（組み合わせ{SV}の並べ替え検定 p={export['variants'][SV]['p_perm']:.2f}）。"
                         "表示は『似たフォームの人のケガの多さ』であり、本人の確率ではない")
    (OUT / "types_form.json").write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: export[k] for k in ("selected_variant", "speed_adjust_per_kmh", "caution")}, ensure_ascii=False, indent=1))
    for v in export["variants"]:
        print(v, export["variants"][v]["ood_mahalanobis_975"], [(t["id"], t["name"], round(t["injury_12m"]["est"], 3)) for t in export["variants"][v]["types"]])


if __name__ == "__main__":
    main()
