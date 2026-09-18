"""ステップ5：npj 2026 の論文の AUC（ランダムフォレスト 0.781〜0.784）が、評価の分け方でどれだけ変わるかを確かめる。

入力: step1_data/outputs/npj_weekly.parquet（週ごと6,181行、person_id 付き）
出力: outputs/paper_replication.csv, outputs/paper_replication_folds.csv

設定（論文にそろえる）:
  - 1行 = 1人×1週、ラベル = RRI（その週のケガ）
  - モデル = scikit-learn の既定値のランダムフォレスト
  - 論文の評価 = 行をランダムに10分割（ケガの比率をそろえる）、各分割の AUC の平均
比べる評価:
  - 行をランダムに10分割（論文と同じ）
  - 人ごとに10分割（同じ人の週が学習用とテスト用に分かれない。ケガの比率はできるだけそろえる）
入力の項目:
  - 根拠の強い項目（論文の class 1、MOESM2 の39項目）
  - そこから、追跡中のケガの記録（tracking_period_injury）を除いた38項目
  - 全項目（論文の all features、257項目）。論文の項目選択（Relief/LASSO を全データで実施）は行わず、全部使う
各設定を分け方を変えて5回繰り返す。
"""
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from ev_common import OUT, SEED, STEP1

warnings.filterwarnings("ignore")

CLASS1 = ["sex", "Age", "lower_limb_days_total", "average_run_hours", "average_interval_training_frequency", "EDEQ_total",
          "hip_abduction_peak_torque", "total_ad_ab_ratio", "knee_extension_peak_torque", "knee_flexion_peak_torque",
          "navicular_drop", "navicular_drop_asymmetry", "Q_angle", "Q_angle_asymmetry", "VALR_12", "Impact_peak_12",
          "Duty_factor_12", "fat_intake_avg", "BMI", "BMD_spine", "past_month_distance", "past_month_ratio", "SC_past_season",
          "non_running_past_season", "tracking_period_injury", "rs11225395", "rs1144393", "rs650108", "rs591058", "rs2252070",
          "rs4986938", "rs1800012", "rs4789932", "rs9340799", "rs970547", "rs1800795", "rs13946", "rs12722", "class1_SNP_risk_score"]
N_REPEATS = 5


def main():
    w = pd.read_parquet(STEP1 / "npj_weekly.parquet")
    all_feats = [c for c in w.columns if c not in ("person_id", "week_index", "RRI")]
    feature_sets = {
        "根拠の強い39項目（class 1）": CLASS1,
        "39項目から追跡中のケガの記録を除く": [c for c in CLASS1 if c != "tracking_period_injury"],
        "全257項目（項目選択なし）": all_feats,
    }
    y = w.RRI.to_numpy()
    groups = w.person_id.to_numpy()
    fold_rows = []
    for fs_name, cols in feature_sets.items():
        X = w[cols].to_numpy(float)
        for split in ("行をランダムに10分割（論文と同じ）", "人ごとに10分割"):
            for r in range(N_REPEATS):
                if split.startswith("行"):
                    cv = StratifiedKFold(10, shuffle=True, random_state=SEED + r).split(X, y)
                else:
                    cv = StratifiedGroupKFold(10, shuffle=True, random_state=SEED + r).split(X, y, groups)
                oof = np.zeros(len(y))
                for k, (tr, te) in enumerate(cv):
                    rf = RandomForestClassifier(random_state=SEED + r, n_jobs=-1).fit(X[tr], y[tr])
                    p = rf.predict_proba(X[te])[:, 1]
                    oof[te] = p
                    fold_rows.append({"features": fs_name, "split": split, "repeat": r, "fold": k,
                                      "auc": roc_auc_score(y[te], p), "n_test": len(te)})
                fold_rows.append({"features": fs_name, "split": split, "repeat": r, "fold": "pooled", "auc": roc_auc_score(y, oof), "n_test": len(y)})
                print(fs_name, split, r, f"{np.mean([x['auc'] for x in fold_rows if x['features'] == fs_name and x['split'] == split and x['repeat'] == r and x['fold'] != 'pooled']):.3f}")
    folds = pd.DataFrame(fold_rows)
    folds.to_csv(OUT / "paper_replication_folds.csv", index=False)
    per_fold = folds[folds.fold != "pooled"]
    summary = per_fold.groupby(["features", "split"]).auc.agg(["mean", "std"]).reset_index()
    pooled = folds[folds.fold == "pooled"].groupby(["features", "split"]).auc.mean().rename("pooled_oof_auc").reset_index()
    summary = summary.merge(pooled, on=["features", "split"])
    summary.to_csv(OUT / "paper_replication.csv", index=False)
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
