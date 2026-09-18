"""ステップ5：フォーム（Loh 2025）で、タイプのケガの割合と、変数をそのまま使ったモデルの予測精度を比べる。

入力: step3_types/outputs/form_features.csv（81名・137脚、重み = 1 ÷ その人の脚の数）
出力: outputs/loh_cv_predictions.csv, loh_metrics.csv, loh_metric_diffs.csv, loh_calibration.csv, loh_spread.csv

予測する結果: 12か月でケガをしたか（0/1）
モデル:
  F0    予測なし（学習データ全体の重み付きのケガの割合）
  F1    タイプ: 学習データで重み付き k-means（2タイプ）を作り直し、各タイプの重み付きのケガの割合
  F2    変数そのまま: 骨盤の傾き・股関節の内転・膝の屈曲・膝の外反、重み付きの罰則付きロジスティック回帰（L2）
  F3    F2 ＋性別・年齢・週の走行距離
  F2-KF 参考：膝の屈曲だけ（ステップ3-2の結果を見てから選んだ項目なので、楽観的に出る）
評価: 人ごとの5分割（ケガの有無の比率をそろえる）× 50回。罰則の強さは学習データの中の3分割で決める（重み付き対数損失が最小）。
指標: 重み付き AUC、重み付き Brier スコア、キャリブレーションの傾きと切片（calibration-in-the-large）
  AUC は、同じ分割（fold）の中の脚どうしの組だけで計算する（分割ごとの予測の水準のずれで不当に低く出ないように。予測なしがちょうど0.5になる）。
実行: python3 22_loh_prediction.py            … 予測から評価まで
      python3 22_loh_prediction.py --eval-only … 保存した予測から評価だけやり直す
信頼区間: テスト用の予測を人単位で500回ブートストラップ（繰り返し50回の平均）。モデル間の差も同じブートストラップで出す。
"""
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold

from ev_common import N_BOOT, N_FOLDS, N_REPEATS_LOH, OUT, SEED, STEP3, import_step3, percentile_ci

warnings.filterwarnings("ignore")

ANGLES = ["CPD", "HADD", "KF", "KA"]
MODELS = {
    "F0 予測なし": None,
    "F1 タイプ（組み合わせA、2タイプ）": "type",
    "F2 変数そのまま（4つの角度）": ANGLES,
    "F3 ＋性別・年齢・週の走行距離": ANGLES + ["male", "age", "km_per_week"],
    "参考 F2-KF 膝の屈曲だけ（事後に選んだ項目）": ["KF"],
}
CS = [0.01, 0.1, 1.0, 10.0, 100.0]


def wmean(x, w):
    return float(np.average(x, weights=w))


def person_folds(d, n_splits, seed):
    persons = d.groupby("pid").injured.first()
    skf = StratifiedKFold(n_splits, shuffle=True, random_state=seed)
    for tr, te in skf.split(persons.index, persons.to_numpy()):
        yield d.pid.isin(persons.index[tr]).to_numpy(), d.pid.isin(persons.index[te]).to_numpy()


def fit_logit(train, cols, C):
    w = train.weight.to_numpy()
    m = np.average(train[cols], axis=0, weights=w)
    s = np.sqrt(np.average((train[cols] - m) ** 2, axis=0, weights=w))
    s = np.where(s == 0, 1, s)
    clf = LogisticRegression(C=C, max_iter=1000).fit((train[cols] - m) / s, train.injured, sample_weight=w)
    return clf, m, s


def predict_logit(model, test, cols):
    clf, m, s = model
    return clf.predict_proba((test[cols] - m) / s)[:, 1]


def tune_C(train, cols, seed):
    losses = {C: [] for C in CS}
    for tr, va in person_folds(train, 3, seed):
        a, b = train[tr], train[va]
        if a.injured.nunique() < 2:
            continue
        for C in CS:
            p = predict_logit(fit_logit(a, cols, C), b, cols)
            losses[C].append(log_loss(b.injured, p, sample_weight=b.weight, labels=[0, 1]))
    return min(CS, key=lambda C: np.mean(losses[C]))


def predict_fold(spec, train, test, seed):
    if spec is None:
        return np.full(len(test), wmean(train.injured, train.weight)), np.nan
    if spec == "type":
        fc = import_step3("12_form_cluster")
        X = train[ANGLES].to_numpy(float)
        labels, predict, _ = fc.fit_method("kmeans", 2, X, train.weight.to_numpy(), train.pid.to_numpy(), ANGLES, n_init=20)
        risk = {int(t): wmean(train.injured[labels == t], train.weight[labels == t]) for t in np.unique(labels)}
        return np.array([risk[int(t)] for t in predict(test[ANGLES].to_numpy(float))]), np.nan
    C = tune_C(train, spec, seed)
    return predict_logit(fit_logit(train, spec, C), test, spec), C


def run_repeat(d, r):
    rows = []
    for fold, (tr, te) in enumerate(person_folds(d, N_FOLDS, SEED + r)):
        train, test = d[tr].reset_index(drop=True), d[te].reset_index(drop=True)
        for name, spec in MODELS.items():
            pred, C = predict_fold(spec, train, test, SEED + r * 100 + fold)
            rows.append(pd.DataFrame({"repeat": r, "fold": fold, "model": name, "pid": test.pid, "side": test.side,
                                      "weight": test.weight, "injured": test.injured, "pred": pred, "C": C}))
    return pd.concat(rows)


def calib(y, p, w):
    """キャリブレーションの傾き（logit(予測) の係数）と切片（logit(予測) をオフセットにしたときの切片）。"""
    p = np.clip(p, 1e-4, 1 - 1e-4)
    lp = np.log(p / (1 - p))
    if np.std(lp) < 1e-8:
        slope = np.nan
    else:
        slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial(), freq_weights=w).fit().params[1]
    inter = sm.GLM(y, np.ones((len(y), 1)), family=sm.families.Binomial(), offset=lp, freq_weights=w).fit().params[0]
    return slope, inter


def fold_auc(y, p, w, fold):
    """重み付き AUC。ケガあり×ケガなしの組は同じ分割の中だけ、組の重み = 両方の脚の重みの積。"""
    pos, neg = y == 1, y == 0
    same = fold[pos][:, None] == fold[neg][None, :]
    pw = w[pos][:, None] * w[neg][None, :] * same
    if pw.sum() == 0:
        return np.nan
    comp = (p[pos][:, None] > p[neg][None, :]) + 0.5 * (p[pos][:, None] == p[neg][None, :])
    return float((comp * pw).sum() / pw.sum())


def metrics(y, p, w, fold):
    auc = fold_auc(y, p, w, fold) if len(np.unique(y)) == 2 else np.nan
    brier = float(np.average((p - y) ** 2, weights=w))
    slope, inter = calib(y, p, w)
    return {"auc": auc, "brier": brier, "calib_slope": slope, "calib_intercept": inter}


def main():
    if "--eval-only" in sys.argv:
        preds = pd.read_csv(OUT / "loh_cv_predictions.csv")
    else:
        d = pd.read_csv(STEP3 / "form_features.csv")
        print(f"脚 {len(d)}、人 {d.pid.nunique()}、ケガ {d[d.injured == 1].pid.nunique()} 名")
        parts = Parallel(n_jobs=10)(delayed(run_repeat)(d, r) for r in range(N_REPEATS_LOH))
        preds = pd.concat(parts, ignore_index=True)
        preds.to_csv(OUT / "loh_cv_predictions.csv", index=False)

    models = list(MODELS)
    key = ["pid", "side"]
    wide = {m: preds[preds.model == m].pivot_table(index=key, columns="repeat", values="pred") for m in models}
    folds = preds[preds.model == models[0]].pivot_table(index=key, columns="repeat", values="fold").loc[wide[models[0]].index]
    base = preds[(preds.model == models[0]) & (preds.repeat == 0)].set_index(key).loc[wide[models[0]].index]
    y, w = base.injured.to_numpy(int), base.weight.to_numpy(float)
    pid_arr = np.array([k[0] for k in wide[models[0]].index])
    rows_of = {p: np.flatnonzero(pid_arr == p) for p in np.unique(pid_arr)}

    def avg_metrics(idx):
        out = {}
        for m in models:
            vals = [metrics(y[idx], wide[m][r].to_numpy()[idx], w[idx], folds[r].to_numpy()[idx]) for r in range(N_REPEATS_LOH)]
            out[m] = pd.DataFrame(vals).mean()
        return out

    point = avg_metrics(np.arange(len(y)))
    rng = np.random.default_rng(SEED)
    persons = np.array(list(rows_of))
    boot_idx = [np.concatenate([rows_of[p] for p in rng.choice(persons, len(persons), replace=True)]) for _ in range(N_BOOT)]
    boots = Parallel(n_jobs=10)(delayed(avg_metrics)(idx) for idx in boot_idx)

    rows = []
    for m in models:
        for k in ("auc", "brier", "calib_slope", "calib_intercept"):
            lo, hi = percentile_ci([b[m][k] for b in boots])
            rows.append({"model": m, "metric": k, "est": point[m][k], "lo": lo, "hi": hi})
    met = pd.DataFrame(rows)
    met.to_csv(OUT / "loh_metrics.csv", index=False)

    comps = [("F2 変数そのまま（4つの角度）", "F1 タイプ（組み合わせA、2タイプ）"), ("F2 変数そのまま（4つの角度）", "F0 予測なし"),
             ("F1 タイプ（組み合わせA、2タイプ）", "F0 予測なし"), ("F3 ＋性別・年齢・週の走行距離", "F2 変数そのまま（4つの角度）"),
             ("参考 F2-KF 膝の屈曲だけ（事後に選んだ項目）", "F0 予測なし")]
    drows = []
    for a, b in comps:
        for k in ("auc", "brier"):
            lo, hi = percentile_ci([bb[a][k] - bb[b][k] for bb in boots])
            drows.append({"comparison": f"{a} − {b}", "metric": k, "diff": point[a][k] - point[b][k], "lo": lo, "hi": hi})
    pd.DataFrame(drows).to_csv(OUT / "loh_metric_diffs.csv", index=False)

    cal, spread = [], []
    for m in models:
        p = preds[preds.model == m].copy()
        spread.append({"model": m, "p10": p.pred.quantile(0.1), "p50": p.pred.median(), "p90": p.pred.quantile(0.9)})
        if p.pred.nunique() < 3:
            groups = [(f"予測 {v:.2f}", g) for v, g in p.groupby(p.pred.round(3))] if p.pred.nunique() <= 2 else [("全員", p)]
        else:
            p["bin"] = p.groupby("repeat").pred.transform(lambda s: pd.qcut(s.rank(method="first"), 3, labels=["低", "中", "高"]))
            groups = list(p.groupby("bin"))
        for label, g in groups:
            cal.append({"model": m, "bin": label, "pred_mean": wmean(g.pred, g.weight), "observed": wmean(g.injured, g.weight), "n_rows": len(g)})
    pd.DataFrame(cal).to_csv(OUT / "loh_calibration.csv", index=False)
    pd.DataFrame(spread).to_csv(OUT / "loh_spread.csv", index=False)

    pd.set_option("display.width", 200)
    print(met.round(3).to_string(index=False))
    print(pd.DataFrame(drows).round(3).to_string(index=False))
    print(pd.DataFrame(spread).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
