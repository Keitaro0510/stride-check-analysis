"""ステップ4-4：平均的なランナーの基準（時速10km相当）と、ピッチを変えたときのシミュレーション。

入力: outputs/load_dataset.csv, step3_types/outputs/types_form.json（Loh の角度の分布との比較用）
出力: outputs/load_reference.csv, load_feature_reference.csv, load_cadence_sim.csv, load_final_models.json

平均的なランナー:
  2.5 m/s と 3.5 m/s の両方がある人×脚で、各指標と入力の値を時速10km（2.78 m/s）に直線で内挿し、平均と SD を出す
最終モデル:
  全データで L3（と Lang）のリッジ回帰を学習（罰則の強さは人ごとの5分割で決める）
ピッチのシミュレーション（因果ではなく、人どうしの違いから作った推定）:
  1. 2.5・3.5 m/s のデータで、ピッチにつられて変わる入力（接地時間、Duty factor、オーバーストライド、接地時の膝・すね・足の角度、
     立脚中期の膝の屈曲、骨盤の上下動、骨盤の傾き、股関節の内転、膝の外反）を、ピッチ・速度・身長・体重で回帰し、ピッチの傾きを求める
  2. 平均的なランナー（時速10km）のピッチを +5%・+10% にし、1 の傾きで他の入力も動かして、L3 で負担を予測し直す
  3. 人単位のブートストラップ200回（1 と L3 を学習し直す）で、変化の95%信頼区間を出す
  画面に出す条件（結果を見る前に決めたもの）: 先行研究の向き（ピッチを上げると膝の伸展モーメント・股関節の外転モーメントが下がる）と一致し、
  かつ +10% の変化の95%信頼区間が0を含まない部位だけ。足首・床反力・負荷率は先行研究の向きが一定しないので、画面には出さない
"""
import json
import warnings

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_common import OUT, SEED, TARGET_SPEED_MS

warnings.filterwarnings("ignore")
TARGETS = ["knee_ext_moment", "ankle_pf_moment", "hip_abd_moment", "vgrf_peak_bw", "loading_rate_bw_s"]
BASE = ["speed_ms", "mass_kg", "height_cm"]
TIMING = ["cadence_spm", "duty", "contact_s"]
KINE = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride", "pelvis_vertical_osc"]
L3 = BASE + TIMING + KINE
LANG = BASE + ["CPD", "HADD", "KA", "KF"]
DEPENDENT = ["duty", "contact_s", "overstride", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "KF", "pelvis_vertical_osc", "CPD", "HADD", "KA"]
EXPECTED = {"knee_ext_moment": -1, "hip_abd_moment": -1}
ALPHAS = [0.1, 1, 3, 10, 30, 100, 300, 1000]


def fit_ridge(d, target, feats):
    ok = d[target].notna()
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    gs = GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=GroupKFold(5), scoring="neg_mean_squared_error")
    gs.fit(d.loc[ok, feats], d.loc[ok, target], groups=d.loc[ok, "subject"])
    return gs.best_estimator_


def export_ridge(model, feats):
    imp, sc, rg = model.named_steps["simpleimputer"], model.named_steps["standardscaler"], model.named_steps["ridge"]
    return {"features": feats, "impute_median": imp.statistics_.tolist(), "scaler_mean": sc.mean_.tolist(), "scaler_scale": sc.scale_.tolist(),
            "coef": rg.coef_.tolist(), "intercept": float(rg.intercept_), "alpha": float(rg.alpha),
            "formula": "x の欠けは impute_median で埋める; z = (x − scaler_mean) / scaler_scale; y = intercept + Σ coef × z"}


def interpolate_10kmh(d, cols):
    rows = []
    for (s, side), g in d.groupby(["subject", "side"]):
        g = g.set_index("speed_ms")
        if 2.5 in g.index and 3.5 in g.index:
            w = (TARGET_SPEED_MS - 2.5) / 1.0
            rows.append({"subject": s, "side": side, **{c: g.loc[2.5, c] + w * (g.loc[3.5, c] - g.loc[2.5, c]) for c in cols}})
    return pd.DataFrame(rows)


def cadence_slopes(d):
    sub = d[d.speed_ms.isin([2.5, 3.5])]
    slopes = {}
    for dep in DEPENDENT:
        ok = sub[dep].notna()
        X = sub.loc[ok, ["cadence_spm", "speed_ms", "height_cm", "mass_kg"]]
        slopes[dep] = float(LinearRegression().fit(X, sub.loc[ok, dep]).coef_[0])
    return slopes


def simulate(d, ref_vec, pct_list=(5, 10)):
    models = {t: fit_ridge(d, t, L3) for t in TARGETS}
    slopes = cadence_slopes(d)
    base = pd.DataFrame([ref_vec])[L3]
    out = {}
    for pct in pct_list:
        v = base.copy()
        dc = ref_vec["cadence_spm"] * pct / 100
        v["cadence_spm"] += dc
        for dep, sl in slopes.items():
            v[dep] += sl * dc
        for t, m in models.items():
            b, n = float(m.predict(base)[0]), float(m.predict(v)[0])
            out[(t, pct)] = (n - b) / b * 100
    return out, slopes


def main():
    d = pd.read_csv(OUT / "load_dataset.csv")
    cols = TARGETS + L3
    ref = interpolate_10kmh(d, [c for c in cols if c != "speed_ms"])
    ref["speed_ms"] = TARGET_SPEED_MS
    load_ref = pd.DataFrame([{"target": t, "mean": ref[t].mean(), "sd": ref[t].std(), "p10": ref[t].quantile(0.1), "p90": ref[t].quantile(0.9),
                              "n_legs": int(ref[t].notna().sum()), "n_subjects": int(ref.loc[ref[t].notna(), "subject"].nunique())} for t in TARGETS])
    load_ref.to_csv(OUT / "load_reference.csv", index=False)

    ftypes = json.loads((OUT.parent.parent / "step3_types" / "outputs" / "types_form.json").read_text(encoding="utf-8"))
    loh = ftypes["reference_distributions"]["Loh 2025 ケガなし（参加時）"]
    feat_ref = pd.DataFrame([{"feature": f, "mean": ref[f].mean(), "sd": ref[f].std(),
                              "loh2025_mean": loh[f]["mean"] if f in loh else np.nan, "loh2025_sd": loh[f]["sd"] if f in loh else np.nan}
                             for f in L3 if f != "speed_ms"])
    feat_ref.to_csv(OUT / "load_feature_reference.csv", index=False)

    ref_vec = ref[L3].mean().to_dict()
    ref_vec["speed_ms"] = TARGET_SPEED_MS
    point, slopes = simulate(d, ref_vec)
    rng = np.random.default_rng(SEED)
    subjects = d.subject.unique()
    boots = []
    for b in range(200):
        pick = rng.choice(subjects, len(subjects), replace=True)
        bd = pd.concat([d[d.subject == s].assign(subject=f"{s}_{i}") for i, s in enumerate(pick)], ignore_index=True)
        try:
            boots.append(simulate(bd, ref_vec)[0])
        except Exception:
            pass
    rows = []
    for (t, pct), v in point.items():
        vals = [bb[(t, pct)] for bb in boots]
        lo, hi = np.percentile(vals, [2.5, 97.5])
        exp = EXPECTED.get(t)
        show = pct == 10 and exp is not None and np.sign(v) == exp and (hi < 0 or lo > 0)
        rows.append({"target": t, "cadence_change_pct": pct, "load_change_pct": v, "lo": lo, "hi": hi,
                     "expected_direction": {-1: "下がる", 1: "上がる"}.get(exp, "一定しない"), "show_on_screen": bool(show)})
    sim = pd.DataFrame(rows)
    # +5% は +10% と同じ判断にそろえる
    show10 = sim[sim.cadence_change_pct == 10].set_index("target").show_on_screen
    sim["show_on_screen"] = sim.target.map(show10)
    sim.to_csv(OUT / "load_cadence_sim.csv", index=False)

    final = {"L3": {t: export_ridge(fit_ridge(d, t, L3), L3) for t in TARGETS},
             "Lang": {t: export_ridge(fit_ridge(d, t, LANG), LANG) for t in TARGETS},
             "cadence_slopes_per_spm": slopes, "reference_input_10kmh": ref_vec}
    (OUT / "load_final_models.json").write_text(json.dumps(final, ensure_ascii=False, indent=1), encoding="utf-8")

    pd.set_option("display.width", 200)
    print(load_ref.round(3).to_string(index=False))
    print(feat_ref.round(3).to_string(index=False))
    print(pd.Series(slopes).round(4).to_string())
    print(sim.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
