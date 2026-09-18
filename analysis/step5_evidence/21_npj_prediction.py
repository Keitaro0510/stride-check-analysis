"""ステップ5：リズム（npj 2026）で、タイプのケガの割合と、変数をそのまま使ったモデルの予測精度を比べる。

入力: step3_types/outputs/rhythm_features.csv（140名、走り方の値を埋めた2名を除く）, step1_data/outputs/npj_person.csv
出力: outputs/npj_cv_predictions.csv, npj_metrics.csv, npj_metric_diffs.csv, npj_calibration.csv, npj_spread.csv

予測する結果: 52週以内に初めてケガをするか（打ち切りを考慮）
モデル:
  R0  予測なし（学習データ全体の52週以内のケガの割合）
  R1  タイプ: 学習データで組み合わせBの k-means（3タイプ）を作り直し、各タイプの52週以内のケガの割合
  R2  変数そのまま: 組み合わせBの6項目、罰則付き Cox 回帰（直線）
  R2b 変数そのまま（曲線）: R2 にピッチと Duty factor の2乗の項を足す
  R3  R2 ＋ケガ歴（過去1年のケガの日数、疲労骨折の既往）
  R4  R3 ＋練習量（参加時の平均の練習時間、直近の走行距離）＋性別
評価: 人ごとの5分割 × 20回。罰則の強さは学習データの中の3分割で決める（C-index が最大）。
指標: C-index（Harrell）、52週時点の AUC（Uno、打ち切りの逆確率で重み付け）、52週時点の Brier スコア（同）
  C-index と AUC は、同じ分割（fold）の中の人どうしの組だけで計算する。分割ごとに学習データが違うので、
  全分割の予測をまとめて比べると、分割ごとの予測の水準のずれが「逆向きの予測」として数えられ、不当に低く出るため
  （予測なしのモデルがちょうど0.5になる）。Brier スコアは水準のずれも含めて評価すべきなので、まとめて計算する。
実行: python3 21_npj_prediction.py            … 予測から評価まで
      python3 21_npj_prediction.py --eval-only … 保存した予測から評価だけやり直す
信頼区間: テスト用の予測を人単位で500回ブートストラップ（繰り返し20回の平均）。モデル間の差も同じブートストラップで出す。
左右差は、ステップ3のパーセンタイル（全員で順位を付けた値）ではなく、npj の正規化された値をそのまま使う
（順位付けにテスト用の人が混ざらないようにするため。順位は単調な変換なので、タイプの中身は変わらない）。
"""
import sys
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold

from ev_common import HORIZON, N_BOOT, N_FOLDS, N_REPEATS_NPJ, OUT, SEED, STEP1, STEP3, import_step3, percentile_ci

warnings.filterwarnings("ignore")

VIDEO = ["cadence_10", "duty_10", "cadence_asym_10", "duty_asym_10", "rearfoot_10", "alt_strike"]
HISTORY = ["injury_days_prev_year", "past_stress_injury"]
LOAD = ["run_hours_norm", "past_month_distance_norm", "sex"]
MODELS = {
    "R0 予測なし": None,
    "R1 タイプ（組み合わせB、3タイプ）": "type",
    "R2 変数そのまま（直線）": VIDEO,
    "R2b 変数そのまま（曲線）": VIDEO + ["cadence_10_sq", "duty_10_sq"],
    "R3 ＋ケガ歴": VIDEO + HISTORY,
    "R4 ＋ケガ歴＋練習量＋性別": VIDEO + HISTORY + LOAD,
}
PENALIZERS = [0.01, 0.1, 0.5, 1.0, 5.0]


def load_data():
    f = pd.read_csv(STEP3 / "rhythm_features.csv")
    norm = pd.read_csv(STEP1 / "npj_person.csv")[["person_id", "Cadence_asymmetry_10", "Duty_factor_asymmetry_10"]]
    d = f[f.imputed_running == 0].merge(norm, on="person_id")
    d["cadence_asym_10"] = d.Cadence_asymmetry_10
    d["duty_asym_10"] = d.Duty_factor_asymmetry_10
    return d.reset_index(drop=True)


# ------------------------------------------------------------------ モデル
def km_risk(time, event, t=HORIZON):
    kmf = KaplanMeierFitter().fit(time, event)
    return 1.0 - float(kmf.survival_function_at_times(t).iloc[0])


def fit_cox(train, cols, penalizer):
    Xtr = train[cols]
    m, s = Xtr.mean(), Xtr.std().replace(0, 1)
    df = ((Xtr - m) / s).assign(time=train.time.to_numpy(), event=train.event.to_numpy())
    cph = CoxPHFitter(penalizer=penalizer).fit(df, "time", "event")
    return cph, m, s


def predict_cox(model, test, cols):
    cph, m, s = model
    Xte = (test[cols] - m) / s
    sf = cph.predict_survival_function(Xte, times=[HORIZON])
    return 1.0 - sf.to_numpy()[0]


def tune_penalizer(train, cols, rng_seed):
    kf = KFold(3, shuffle=True, random_state=rng_seed)
    scores = {p: [] for p in PENALIZERS}
    for tr, va in kf.split(train):
        a, b = train.iloc[tr], train.iloc[va]
        for p in PENALIZERS:
            try:
                risk = predict_cox(fit_cox(a, cols, p), b, cols)
                scores[p].append(concordance_index(b.time, -risk, b.event))
            except Exception:
                scores[p].append(np.nan)
    return max(PENALIZERS, key=lambda p: np.nanmean(scores[p]))


def predict_fold(name, spec, train, test, seed):
    if spec is None:
        return np.full(len(test), km_risk(train.time, train.event)), np.nan
    if spec == "type":
        cluster = import_step3("02_cluster")
        cols = ["cadence_10", "duty_10", "cadence_asym_10", "duty_asym_10", "rearfoot_10", "alt_strike"]
        labels, predict, _ = cluster.fit_method("kmeans", 3, train[cols].to_numpy(float), cols, n_init=20)
        risks = {}
        for t in np.unique(labels):
            g = train[labels == t]
            risks[t] = km_risk(g.time, g.event)
        return np.array([risks[int(t)] for t in predict(test[cols].to_numpy(float))]), np.nan
    p = tune_penalizer(train, spec, seed)
    return predict_cox(fit_cox(train, spec, p), test, spec), p


def run_repeat(d, r):
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + r)
    rows = []
    for fold, (tr, te) in enumerate(kf.split(d)):
        train, test = d.iloc[tr], d.iloc[te]
        # 2乗の項は学習データの平均・SDで作る
        for c in ("cadence_10", "duty_10"):
            m, s = train[c].mean(), train[c].std()
            train = train.assign(**{c + "_sq": ((train[c] - m) / s) ** 2})
            test = test.assign(**{c + "_sq": ((test[c] - m) / s) ** 2})
        for name, spec in MODELS.items():
            pred, pen = predict_fold(name, spec, train, test, SEED + r * 100 + fold)
            rows.append(pd.DataFrame({"repeat": r, "fold": fold, "model": name, "person_id": test.person_id.to_numpy(),
                                      "time": test.time.to_numpy(), "event": test.event.to_numpy(), "pred_risk_52w": pred, "penalizer": pen}))
    return pd.concat(rows)


# ------------------------------------------------------------------ 指標
def censoring_km(time, event):
    return KaplanMeierFitter().fit(time, 1 - event)


def G_at(gkm, t):
    return np.clip(gkm.survival_function_at_times(t).to_numpy(), 1e-3, None)


def uno_auc(time, event, risk, gkm, t=HORIZON):
    cases = (time <= t) & (event == 1)
    controls = (time > t) | ((time == t) & (event == 0))
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan
    w = 1.0 / G_at(gkm, time[cases] - 0.5)
    rc, rn = risk[cases][:, None], risk[controls][None, :]
    comp = (rc > rn).mean(axis=1) + 0.5 * (rc == rn).mean(axis=1)
    return float(np.sum(w * comp) / np.sum(w))


def ipcw_brier(time, event, risk, gkm, t=HORIZON):
    cases = (time <= t) & (event == 1)
    controls = (time > t) | ((time == t) & (event == 0))
    s = np.zeros(len(time))
    s[cases] = (1 - risk[cases]) ** 2 / G_at(gkm, time[cases] - 0.5)
    s[controls] = risk[controls] ** 2 / G_at(gkm, np.array([t]))[0]
    return float(s.mean())


def fold_c_index(time, event, risk, fold):
    """Harrell の C-index。比べる組は同じ分割の中だけ。"""
    same = fold[:, None] == fold[None, :]
    comparable = (time[:, None] < time[None, :]) & (event[:, None] == 1) & same
    conc = (risk[:, None] > risk[None, :]) + 0.5 * (risk[:, None] == risk[None, :])
    n = comparable.sum()
    return float((conc * comparable).sum() / n) if n else np.nan


def fold_uno_auc(time, event, risk, fold, gkm, t=HORIZON):
    """52週時点の AUC（Uno）。ケガをした人と52週を超えてケガのない人の組は、同じ分割の中だけ。"""
    cases = (time <= t) & (event == 1)
    controls = (time > t) | ((time == t) & (event == 0))
    if cases.sum() == 0 or controls.sum() == 0:
        return np.nan
    w = 1.0 / G_at(gkm, time[cases] - 0.5)
    same = fold[cases][:, None] == fold[controls][None, :]
    rc, rn = risk[cases][:, None], risk[controls][None, :]
    comp = ((rc > rn) + 0.5 * (rc == rn)) * same
    n_ctrl = same.sum(axis=1)
    ok = n_ctrl > 0
    return float(np.sum(w[ok] * comp[ok].sum(axis=1)) / np.sum(w[ok] * n_ctrl[ok]))


def metrics(time, event, risk, fold):
    gkm = censoring_km(time, event)
    return {"c_index": fold_c_index(time, event, risk, fold), "auc_52w": fold_uno_auc(time, event, risk, fold, gkm),
            "brier_52w": ipcw_brier(time, event, risk, gkm)}


def main():
    if "--eval-only" in sys.argv:
        preds = pd.read_csv(OUT / "npj_cv_predictions.csv")
    else:
        d = load_data()
        print(f"人数 {len(d)}、ケガ {int(d.event.sum())}")
        parts = Parallel(n_jobs=10)(delayed(run_repeat)(d, r) for r in range(N_REPEATS_NPJ))
        preds = pd.concat(parts, ignore_index=True)
        preds.to_csv(OUT / "npj_cv_predictions.csv", index=False)

    models = list(MODELS)
    # 人×繰り返しの予測の表（行=人、列=繰り返し）
    wide = {m: preds[preds.model == m].pivot(index="person_id", columns="repeat", values="pred_risk_52w") for m in models}
    folds = preds[preds.model == models[0]].pivot(index="person_id", columns="repeat", values="fold").loc[wide[models[0]].index]
    base = preds[(preds.model == models[0]) & (preds.repeat == 0)].set_index("person_id").loc[wide[models[0]].index]
    time, event = base.time.to_numpy(float), base.event.to_numpy(int)

    def avg_metrics(idx):
        out = {}
        for m in models:
            vals = [metrics(time[idx], event[idx], wide[m][r].to_numpy()[idx], folds[r].to_numpy()[idx]) for r in range(N_REPEATS_NPJ)]
            out[m] = pd.DataFrame(vals).mean()
        return out

    point = avg_metrics(np.arange(len(time)))
    rng = np.random.default_rng(SEED)
    boot_idx = [rng.integers(0, len(time), len(time)) for _ in range(N_BOOT)]
    boots = Parallel(n_jobs=10)(delayed(avg_metrics)(idx) for idx in boot_idx)

    rows = []
    for m in models:
        for k in ("c_index", "auc_52w", "brier_52w"):
            lo, hi = percentile_ci([b[m][k] for b in boots])
            rows.append({"model": m, "metric": k, "est": point[m][k], "lo": lo, "hi": hi})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "npj_metrics.csv", index=False)

    comps = [("R2 変数そのまま（直線）", "R1 タイプ（組み合わせB、3タイプ）"), ("R2b 変数そのまま（曲線）", "R1 タイプ（組み合わせB、3タイプ）"),
             ("R2 変数そのまま（直線）", "R0 予測なし"), ("R1 タイプ（組み合わせB、3タイプ）", "R0 予測なし"),
             ("R3 ＋ケガ歴", "R2 変数そのまま（直線）"), ("R4 ＋ケガ歴＋練習量＋性別", "R2 変数そのまま（直線）"), ("R4 ＋ケガ歴＋練習量＋性別", "R0 予測なし")]
    drows = []
    for a, b in comps:
        for k in ("c_index", "auc_52w", "brier_52w"):
            lo, hi = percentile_ci([bb[a][k] - bb[b][k] for bb in boots])
            drows.append({"comparison": f"{a} − {b}", "metric": k, "diff": point[a][k] - point[b][k], "lo": lo, "hi": hi})
    pd.DataFrame(drows).to_csv(OUT / "npj_metric_diffs.csv", index=False)

    # キャリブレーション（予測を3等分、全繰り返しをまとめる）と予測の広がり
    cal, spread = [], []
    for m in models:
        p = preds[preds.model == m].copy()
        spread.append({"model": m, "p10": p.pred_risk_52w.quantile(0.1), "p50": p.pred_risk_52w.median(), "p90": p.pred_risk_52w.quantile(0.9)})
        if p.pred_risk_52w.nunique() < 3:
            groups = [("全員", p)]
        else:
            p["bin"] = p.groupby("repeat").pred_risk_52w.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["低", "中", "高"]))
            groups = list(p.groupby("bin"))
        for label, g in groups:
            cal.append({"model": m, "bin": label, "pred_mean": g.pred_risk_52w.mean(), "observed_km": km_risk(g.time, g.event), "n_rows": len(g)})
    pd.DataFrame(cal).to_csv(OUT / "npj_calibration.csv", index=False)
    pd.DataFrame(spread).to_csv(OUT / "npj_spread.csv", index=False)

    pd.set_option("display.width", 200)
    print(met.round(3).to_string(index=False))  # noqa
    print(pd.DataFrame(drows).round(3).to_string(index=False))
    print(pd.DataFrame(spread).round(3).to_string(index=False))
    print(pd.DataFrame(cal).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
