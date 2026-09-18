"""ステップ3-1（3-3の準備を含む）：新しいランナーのリズムを判定するための JSON を書き出し、Fukuchi のランナーで確認する。

入力: outputs/rhythm_features.csv, rhythm_models.json, rhythm_outcomes.csv, step1_data/outputs/fukuchi_steps.csv
出力: outputs/types_rhythm.json, outputs/rhythm_fukuchi_check.csv, outputs/rhythm_export_check.txt

判定のしかた（プロトタイプ側で同じ計算をする）:
  1. 時速10km以外で撮った場合は、速度の補正で時速10km相当に直す（npj 2026 の同じ人の 10→12 km/h の変化から）
  2. 採用した分け方でタイプを決める（k-means なら標準化して一番近い中心、ルールなら閾値）
  3. 参加者全体の分布からのマハラノビス距離が、参加者の97.5%点より遠ければ「近いタイプなし」
組み合わせBの左右差は、npj 2026 では元の単位に戻せないので「集団の中での順位（パーセンタイル）」で持っている。
  動画で測った左右差は、Fukuchi のランナーの左右差の分布の中での順位に変換してから当てはめる（同じような集団だという仮定）。
"""
import importlib
import json

import numpy as np
import pandas as pd

from common import HORIZON_WEEKS, OUT, SELECTED_VARIANT, STEP1, VARIANTS

cluster = importlib.import_module("02_cluster")


def speed_slopes(d: pd.DataFrame) -> dict:
    """同じ人の時速10kmと12kmの差から、1 km/h あたりの変化を出す（中央値）。"""
    return {"cadence_10": float(np.median((d.cadence_12 - d.cadence_10) / 2)),
            "duty_10": float(np.median((d.duty_12 - d.duty_10) / 2))}


def mahalanobis_threshold(Z: np.ndarray):
    cov = np.cov(Z, rowvar=False).reshape(Z.shape[1], Z.shape[1])
    inv = np.linalg.pinv(cov)
    dist = np.sqrt(np.einsum("ij,jk,ik->i", Z, inv, Z))
    return float(np.quantile(dist, 0.975)), inv


def fukuchi_at_10kmh(slopes: dict) -> pd.DataFrame:
    """Fukuchi のランナーの時速10km相当の値。組み合わせBの項目も作る:
    踵接地 = 3.5 m/s で左右とも Rearfoot、左右で接地が違う = 左右の一方だけ Rearfoot、
    Duty factor の左右差 = 接地時間の1歩おきの差の、Fukuchi 内での順位。ピッチの左右差は床反力だけでは分からないので50（中央）とする。"""
    st = pd.read_csv(STEP1 / "fukuchi_steps.csv")
    subj = pd.read_csv(STEP1 / "fukuchi_subjects.csv").set_index("subject")
    v10 = 10 / 3.6
    rows = []
    for sid, g in st.groupby("subject"):
        g = g.set_index("speed_ms")
        if 2.5 in g.index and 3.5 in g.index:
            w = (v10 - 2.5) / 1.0
            cad = g.loc[2.5, "cadence_spm"] + w * (g.loc[3.5, "cadence_spm"] - g.loc[2.5, "cadence_spm"])
            duty = g.loc[2.5, "duty_factor"] + w * (g.loc[3.5, "duty_factor"] - g.loc[2.5, "duty_factor"])
            how = "2.5 と 3.5 m/s から内挿"
        elif 3.5 in g.index:
            dk = 3.5 * 3.6 - 10
            cad = g.loc[3.5, "cadence_spm"] - slopes["cadence_10"] * dk
            duty = g.loc[3.5, "duty_factor"] - slopes["duty_10"] * dk
            how = "3.5 m/s から npj の速度補正"
        else:
            continue
        r_rf, l_rf = subj.loc[sid, "RFSI35"] == "Rearfoot", subj.loc[sid, "LFSI35"] == "Rearfoot"
        rows.append({"subject": sid, "cadence_10": cad, "duty_10": duty, "how": how,
                     "contact_asym": g["contact_asym_pct"].mean(),
                     "rearfoot_10": int(r_rf and l_rf), "alt_strike": int(r_rf != l_rf)})
    out = pd.DataFrame(rows)
    out["duty_asym_pct_10"] = out.contact_asym.rank(pct=True) * 100
    out["cadence_asym_pct_10"] = 50.0
    return out


def main():
    feats = pd.read_csv(OUT / "rhythm_features.csv")
    d = feats[feats.imputed_running == 0].reset_index(drop=True)
    models = json.loads((OUT / "rhythm_models.json").read_text(encoding="utf-8"))
    outcomes = pd.read_csv(OUT / "rhythm_outcomes.csv")
    main_out = outcomes[outcomes.analysis == "主な分析"]
    slopes = speed_slopes(d)
    tests = pd.read_csv(OUT / "rhythm_tests.csv")

    export = {
        "source": "Wu et al. 2026 npj Digital Medicine（持久系ランナー、12か月の前向き追跡）。走り方は参加時に1回、トレッドミル時速10kmで測定",
        "n": int(len(d)),
        "outcome": f"{HORIZON_WEEKS}週以内に初めてケガをした割合（Kaplan-Meier、95%信頼区間）と、100週あたりのケガの週（負の二項回帰）",
        "selected_variant": SELECTED_VARIANT,
        "speed_adjust_per_kmh": slopes,
        "speed_range_kmh": [9, 13],
        "variants": {},
        "caution": "タイプ間でケガの多さに統計的な差はなかった（組み合わせ{}の log-rank p={:.2f}）。表示は『似た走り方の人のケガの多さ』であり、本人の確率ではない".format(
            SELECTED_VARIANT, float(tests[(tests.variant == SELECTED_VARIANT) & tests.test.str.startswith("log-rank")].p.iloc[0])),
        "asymmetry_note": "組み合わせBの左右差はパーセンタイル（0〜100）。動画で測った左右差は、Fukuchi のランナーの分布の中での順位に変換して使う",
    }
    lines = []
    for variant, m in models.items():
        cols = VARIANTS[variant]
        X = d[cols].to_numpy(float)
        Z = (X - np.array(m["params"]["scaler_mean"])) / np.array(m["params"]["scaler_sd"])
        thr, _ = mahalanobis_threshold(Z)
        _, predict, _ = cluster.fit_method(m["method"], m["k"], X, cols, n_init=50)
        labels = predict(X)
        types = []
        for tid, info in sorted(m["types"].items()):
            members = d[labels == info["label"]]
            inc = main_out[(main_out.variant == variant) & (main_out.type == tid) & (main_out.metric == "injury_52w")].iloc[0]
            rate = main_out[(main_out.variant == variant) & (main_out.type == tid) & (main_out.metric == "injury_weeks_per_100")].iloc[0]
            types.append({
                "id": tid, "name": info["name"], "label": info["label"], "n": int(len(members)),
                "mean_raw": {c: float(members[c].mean()) for c in cols},
                "injury_52w": {"est": float(inc.est), "lo": float(inc.lo), "hi": float(inc.hi), "events": int(inc.events)},
                "injury_weeks_per_100": {"est": float(rate.est), "lo": float(rate.lo), "hi": float(rate.hi)},
            })
        overall_inc = cluster_overall(d)
        export["variants"][variant] = {
            "features": cols, "method": m["method"], "k": m["k"], "params": m["params"],
            "ood_mahalanobis_975": thr, "types": types, "overall": overall_inc,
        }
        lines.append(f"[{variant}] {m['method']} k={m['k']} 近いタイプなしの閾値（マハラノビス距離）{thr:.2f}")

    (OUT / "types_rhythm.json").write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    # Fukuchi のランナーを判定して、偏りと「近いタイプなし」を確認（画面に使う組み合わせ）
    SV = SELECTED_VARIANT
    cols = VARIANTS[SV]
    mS = export["variants"][SV]
    fk = fukuchi_at_10kmh(slopes)
    XS = d[cols].to_numpy(float)
    _, predictS, _ = cluster.fit_method(mS["method"], mS["k"], XS, cols, n_init=50)
    label_to_id = {t["label"]: t["id"] for t in mS["types"]}
    mean, sd = np.array(mS["params"]["scaler_mean"]), np.array(mS["params"]["scaler_sd"])
    _, inv = mahalanobis_threshold((XS - mean) / sd)

    def classify(Xf):
        z = (Xf - mean) / sd
        dist = np.sqrt(np.einsum("ij,jk,ik->i", z, inv, z))
        return [label_to_id[int(x)] for x in predictS(Xf)], dist

    Xf = fk[cols].to_numpy(float)
    fk[f"type_{SV}"], fk["mahalanobis"] = classify(Xf)
    fk["ood"] = fk.mahalanobis > mS["ood_mahalanobis_975"]

    # 基準（Fukuchi の平均）の不確かさ: ピッチと Duty factor の平均を ±1 標準誤差ずらしたとき、判定が変わる人の割合
    st = pd.read_csv(STEP1 / "fukuchi_steps.csv")
    near = st[st.speed_ms.isin([2.5, 3.5])]
    se = {"cadence_10": near.cadence_spm.std() / np.sqrt(near.subject.nunique()),
          "duty_10": near.duty_factor.std() / np.sqrt(near.subject.nunique())}
    ic, idu = cols.index("cadence_10"), cols.index("duty_10")
    changes = []
    for dc in (-1, 1):
        for dd in (-1, 1):
            shifted = Xf.copy()
            shifted[:, ic] += dc * se["cadence_10"]
            shifted[:, idu] += dd * se["duty_10"]
            t2, _ = classify(shifted)
            changes.append(float(np.mean(np.array(t2) != fk[f"type_{SV}"].to_numpy())))
    fk.to_csv(OUT / "rhythm_fukuchi_check.csv", index=False)

    lines += [
        "",
        f"速度の補正（1 km/h あたり）: ピッチ {slopes['cadence_10']:+.2f} 歩/分, Duty factor {slopes['duty_10']:+.4f}",
        f"[組み合わせ{SV}] Fukuchi のランナー {len(fk)} 名（時速10km相当）: " + ", ".join(f"{k} {v}名" for k, v in fk[f"type_{SV}"].value_counts().sort_index().items()),
        f"  npj 2026 の140名: " + ", ".join(f"{t['id']} {t['n']}名" for t in mS["types"]),
        f"  近いタイプなし（マハラノビス距離 > {mS['ood_mahalanobis_975']:.2f}）: {int(fk.ood.sum())} 名",
        f"  Fukuchi の踵接地 {int(fk.rearfoot_10.sum())} 名、左右で接地が違う {int(fk.alt_strike.sum())} 名（ピッチの左右差は分からないので中央の50として判定）",
        f"  Fukuchi の平均 ピッチ {fk.cadence_10.mean():.1f}, Duty factor {fk.duty_10.mean():.3f} / npj ピッチ {mean[ic]:.1f}, Duty factor {mean[idu]:.3f}",
        f"基準の平均を ±1標準誤差（ピッチ {se['cadence_10']:.2f} 歩/分、Duty factor {se['duty_10']:.4f}）ずらしたとき、タイプが変わる Fukuchi のランナーの割合: 最大 {max(changes):.0%}",
    ]
    text = "\n".join(lines)
    (OUT / "rhythm_export_check.txt").write_text(text, encoding="utf-8")
    print(text)


def cluster_overall(d: pd.DataFrame) -> dict:
    from lifelines import KaplanMeierFitter

    kmf = KaplanMeierFitter().fit(d.time, d.event)
    s = float(kmf.survival_function_at_times(HORIZON_WEEKS).iloc[0])
    ci = kmf.confidence_interval_survival_function_
    row = ci[ci.index <= HORIZON_WEEKS].iloc[-1]
    return {"injury_52w": {"est": 1 - s, "lo": 1 - float(row.iloc[1]), "hi": 1 - float(row.iloc[0])}, "n": int(len(d)), "events": int(d.event.sum())}


if __name__ == "__main__":
    main()
