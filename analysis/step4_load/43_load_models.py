"""ステップ4-3：動画で測れる値から負担の指標を推定するモデルを、1人ずつ除く交差検証で比べる。

入力: outputs/marker_features.csv, load_targets.csv, step1_data/outputs/fukuchi_subjects.csv
出力: outputs/load_dataset.csv, load_cv_predictions.csv, load_model_metrics.csv, load_noise.csv, load_model_comparison.png

データ: 人×速度×脚（マーカーと負担の指標の品質確認で除外したものを除く）
モデルの入力:
  L0     基準: 速度、体重、身長
  L1     L0 ＋ タイミング: ピッチ、Duty factor、接地時間
  L2     L0 ＋ 角度など: 骨盤の傾き、股関節の内転、膝の外反、膝の屈曲、接地時の膝の屈曲・すねの傾き・足の角度、オーバーストライド、骨盤の上下動
  L3     L1 ＋ L2
  Lang   L0 ＋ Loh と同じ4つの角度（ステップ5-1 用）
方法: リッジ回帰（主。欠けは学習データの中央値、罰則の強さは学習データの中の人ごとの5分割で決める）、勾配ブースティング（比較。深さ2）
評価: 1人ずつ除く交差検証（同じ人の速度・左右は学習とテストに分けない）
指標: R²、RMSE、RMSE ÷ 平均、スピアマンの順位相関、
      「L0 を超えて説明できた割合」R²_extra = 1 − SSE(モデル) ÷ SSE(L0 のリッジ回帰)（速度と体格で決まる分を除いた、走り方による個人差をどれだけ当てられるか）
      R²_extra の95%信頼区間は、人単位のブートストラップ1000回
動画並みの誤差: L3 のリッジ回帰で、テスト用の入力にだけ誤差（標準偏差）を足して、R²_extra の落ち方を見る（10回の平均）。
  誤差の大きさ（1倍）: 角度 3°、オーバーストライド 0.03、骨盤の上下動 10 mm、ピッチ 3 歩/分、Duty factor 0.02、接地時間 0.015 秒
  （仮の値。ステップ2で実際の誤差が分かったら置き換える）
"""
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, GroupKFold, LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from load_common import OUT, SEED, STEP1, plt, setup_font

warnings.filterwarnings("ignore")

TARGETS = {"knee_ext_moment": "膝のお皿の裏（膝の伸展モーメント）", "ankle_pf_moment": "アキレス腱（足首の底屈モーメント）",
           "hip_abd_moment": "股関節の外側（股関節の外転モーメント）", "vgrf_peak_bw": "すね（鉛直床反力のピーク）",
           "loading_rate_bw_s": "すね（負荷率）"}
BASE = ["speed_ms", "mass_kg", "height_cm"]
TIMING = ["cadence_spm", "duty", "contact_s"]
KINE = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride", "pelvis_vertical_osc"]
FEATURE_SETS = {"L0": BASE, "L1": BASE + TIMING, "L2": BASE + KINE, "L3": BASE + TIMING + KINE, "Lang": BASE + ["CPD", "HADD", "KA", "KF"]}
NOISE_SD = {"CPD": 3, "HADD": 3, "KA": 3, "KF": 3, "knee_flex_contact": 3, "shank_angle_contact": 3, "foot_angle_contact": 3,
            "overstride": 0.03, "pelvis_vertical_osc": 10, "cadence_spm": 3, "duty": 0.02, "contact_s": 0.015}
ALPHAS = [0.1, 1, 3, 10, 30, 100, 300, 1000]


def load_dataset():
    mf = pd.read_csv(OUT / "marker_features.csv")
    lt = pd.read_csv(OUT / "load_targets.csv")
    subj = pd.read_csv(STEP1 / "fukuchi_subjects.csv")[["subject", "Mass", "Height"]].rename(columns={"Mass": "mass_kg", "Height": "height_cm"})
    d = mf[~mf.qc_exclude].merge(lt[~lt.qc_exclude], on=["subject", "speed_ms", "side"]).merge(subj, on="subject")
    return d.reset_index(drop=True)


def ridge_model(groups_train):
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), Ridge())
    n_splits = min(5, len(np.unique(groups_train)))
    return GridSearchCV(pipe, {"ridge__alpha": ALPHAS}, cv=GroupKFold(n_splits), scoring="neg_mean_squared_error")


def gbm_model():
    return HistGradientBoostingRegressor(max_depth=2, learning_rate=0.05, max_iter=150, min_samples_leaf=8, random_state=SEED)


def loso_predict(d, target, feats, method, noise_level=0.0, rng=None):
    y = d[target].to_numpy(float)
    X = d[feats].to_numpy(float)
    g = d.subject.to_numpy()
    pred = np.full(len(d), np.nan)
    ok = np.isfinite(y)
    for tr, te in LeaveOneGroupOut().split(X, y, g):
        tr = tr[ok[tr]]
        if method == "ridge":
            m = ridge_model(g[tr]).fit(X[tr], y[tr], groups=g[tr])
        else:
            m = gbm_model().fit(X[tr], y[tr])
        Xte = X[te].copy()
        if noise_level > 0:
            for j, f in enumerate(feats):
                if f in NOISE_SD:
                    Xte[:, j] += rng.normal(0, NOISE_SD[f] * noise_level, len(te))
        pred[te] = m.predict(Xte)
    return pred


def metrics(y, p, p0, idx=None):
    if idx is not None:
        y, p, p0 = y[idx], p[idx], p0[idx]
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(p0)
    y, p, p0 = y[ok], p[ok], p0[ok]
    sse, sse0 = np.sum((y - p) ** 2), np.sum((y - p0) ** 2)
    return {"r2": 1 - sse / np.sum((y - y.mean()) ** 2), "rmse": float(np.sqrt(np.mean((y - p) ** 2))),
            "rel_rmse": float(np.sqrt(np.mean((y - p) ** 2)) / np.mean(y)), "spearman": float(spearmanr(y, p).correlation),
            "r2_extra": 1 - sse / sse0}


def main():
    setup_font()
    d = load_dataset()
    d.to_csv(OUT / "load_dataset.csv", index=False)
    print(f"データ: {len(d)} 件（{d.subject.nunique()} 名）")

    jobs = [(t, fs, m) for t in TARGETS for fs in FEATURE_SETS for m in ("ridge", "gbm")]
    preds = Parallel(n_jobs=12)(delayed(loso_predict)(d, t, FEATURE_SETS[fs], m) for t, fs, m in jobs)
    P = d[["subject", "speed_ms", "side"]].copy()
    for (t, fs, m), p in zip(jobs, preds):
        P[f"{t}|{fs}|{m}"] = p
    P.to_csv(OUT / "load_cv_predictions.csv", index=False)

    rng = np.random.default_rng(SEED)
    subjects = d.subject.unique()
    rows_of = {s: np.flatnonzero(d.subject.to_numpy() == s) for s in subjects}
    boot = [np.concatenate([rows_of[s] for s in rng.choice(subjects, len(subjects), replace=True)]) for _ in range(1000)]
    out = []
    for t, fs, m in jobs:
        y = d[t].to_numpy(float)
        p, p0 = P[f"{t}|{fs}|{m}"].to_numpy(), P[f"{t}|L0|ridge"].to_numpy()
        r = metrics(y, p, p0)
        bx = [metrics(y, p, p0, b)["r2_extra"] for b in boot]
        r.update({"target": t, "feature_set": fs, "method": m, "r2_extra_lo": float(np.nanpercentile(bx, 2.5)),
                  "r2_extra_hi": float(np.nanpercentile(bx, 97.5)), "mean": float(np.nanmean(y)), "n": int(np.isfinite(y).sum())})
        out.append(r)
    met = pd.DataFrame(out)
    met.to_csv(OUT / "load_model_metrics.csv", index=False)

    # 動画並みの誤差（L3 リッジ）
    noise_rows = []
    for t in TARGETS:
        y = d[t].to_numpy(float)
        p0 = P[f"{t}|L0|ridge"].to_numpy()
        for level in (0.0, 1.0, 2.0):
            reps = 1 if level == 0 else 10
            vals = Parallel(n_jobs=10)(delayed(loso_predict)(d, t, FEATURE_SETS["L3"], "ridge", level, np.random.default_rng(SEED + i)) for i in range(reps))
            r2x = [metrics(y, v, p0)["r2_extra"] for v in vals]
            noise_rows.append({"target": t, "noise_level": level, "r2_extra_mean": float(np.mean(r2x)), "r2_extra_sd": float(np.std(r2x))})
    pd.DataFrame(noise_rows).to_csv(OUT / "load_noise.csv", index=False)

    # 図
    fig, axes = plt.subplots(1, len(TARGETS), figsize=(4 * len(TARGETS), 4), sharey=True)
    for ax, t in zip(axes, TARGETS):
        sub = met[(met.target == t) & (met.feature_set != "L0")]
        for i, (fs, g) in enumerate(sub.groupby("feature_set", sort=False)):
            for j, r in enumerate(g.itertuples()):
                x = i + (j - 0.5) * 0.3
                c = "#1F5F8B" if r.method == "ridge" else "#C47A12"
                ax.plot([x, x], [r.r2_extra_lo, r.r2_extra_hi], color=c, lw=2)
                ax.plot(x, r.r2_extra, "o", color=c, label=("リッジ回帰" if r.method == "ridge" else "勾配ブースティング") if i == 0 else None)
        ax.axhline(0, color="#888", ls="--", lw=1)
        ax.set_xticks(range(sub.feature_set.nunique()))
        ax.set_xticklabels(sub.feature_set.unique())
        ax.set_title(TARGETS[t], fontsize=9)
        ax.set_ylim(-0.6, 0.8)
    axes[0].set_ylabel("R²_extra（速度・体格を超えて説明できた割合）")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "load_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    pd.set_option("display.width", 250)
    print(met[["target", "feature_set", "method", "r2", "rel_rmse", "spearman", "r2_extra", "r2_extra_lo", "r2_extra_hi"]].round(3).to_string(index=False))
    print(pd.DataFrame(noise_rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
