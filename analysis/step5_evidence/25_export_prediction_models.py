"""ステップ5-3：プロトタイプ用に、変数をそのまま使う予測モデルを全データで学習し、JSON に書き出す。

入力: step3_types/outputs/rhythm_features.csv, form_features.csv, step1_data/outputs/npj_person.csv,
      step3_types/outputs/types_rhythm.json, types_form.json（速度の補正）,
      outputs/npj_cv_predictions.csv, loh_cv_predictions.csv, npj_metrics.csv, loh_metrics.csv, npj_spread.csv, loh_spread.csv（精度）
出力: outputs/prediction_models.json, outputs/prediction_models_check.txt

モデル（ステップ5-2で採用したもの）:
  リズム  : 罰則付き Cox 回帰（R2、組み合わせBの6項目）。52週以内にケガをする予測
  フォーム: 重み付きの罰則付きロジスティック回帰（F2、4つの角度）。12か月でケガをする予測、脚ごと
罰則の強さ: 5-2 の交差検証で最も多く選ばれた値（リズム penalizer、フォーム C）
計算式（プロトタイプ側で同じ計算をする）:
  リズム  : z = (x − scaler_mean) / scaler_sd、lp = Σ coef × (z − center)、risk = 1 − S0 ^ exp(lp)
  フォーム: z = (x − scaler_mean) / scaler_sd、risk = 1 / (1 + exp(−(intercept + Σ coef × z)))
  項目ごとの影響: リズムは coef × (z − center)、フォームは coef × z（参加者の平均的な値と比べて、予測をどちら向きにどれだけ動かすか）
不確かさ: 人単位のブートストラップで200回学習し直した係数も書き出す。プロトタイプで同じ計算を200回して、2.5%〜97.5%点を予測の幅にする
入力の注意:
  リズムの左右差（cadence_asym_10, duty_asym_10）は npj 2026 の正規化された値で、元の単位に戻せない。
  動画で測った左右差は、Fukuchi の分布での順位（パーセンタイル）に変換し、`asymmetry_quantiles` で npj の値に直してから入れる。
"""
import json
import sys
import warnings

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.linear_model import LogisticRegression

from ev_common import HORIZON, OUT, SEED, STEP1, STEP3

warnings.filterwarnings("ignore")
sys.argv = sys.argv[:1]  # 21/22 を読み込むときに --eval-only などが渡らないように
N_BOOT = 200


def load_module(name):
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parent / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name.replace("-", "_"), path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mode_value(series):
    return float(series.value_counts().idxmax())


# ------------------------------------------------------------------ リズム
def fit_rhythm(d, cols, penalizer):
    mu, sd = d[cols].mean(), d[cols].std().replace(0, 1)
    df = ((d[cols] - mu) / sd).assign(time=d.time.to_numpy(), event=d.event.to_numpy())
    cph = CoxPHFitter(penalizer=penalizer).fit(df, "time", "event")
    s0 = float(cph.baseline_survival_.loc[:HORIZON].iloc[-1, 0])
    return {"scaler_mean": mu.to_numpy(), "scaler_sd": sd.to_numpy(), "coef": cph.params_[cols].to_numpy(),
            "center": cph._norm_mean[cols].to_numpy(), "S0_52w": s0}


def rhythm_risk(p, X):
    z = (X - p["scaler_mean"]) / p["scaler_sd"]
    lp = ((z - p["center"]) * p["coef"]).sum(axis=1)
    return 1 - p["S0_52w"] ** np.exp(lp)


# ------------------------------------------------------------------ フォーム
def fit_form(d, cols, C):
    w = d.weight.to_numpy()
    mu = np.average(d[cols], axis=0, weights=w)
    sd = np.sqrt(np.average((d[cols] - mu) ** 2, axis=0, weights=w))
    clf = LogisticRegression(C=C, max_iter=1000).fit((d[cols] - mu) / sd, d.injured, sample_weight=w)
    return {"scaler_mean": mu, "scaler_sd": sd, "coef": clf.coef_[0], "intercept": float(clf.intercept_[0])}


def form_risk(p, X):
    z = (X - p["scaler_mean"]) / p["scaler_sd"]
    return 1 / (1 + np.exp(-(p["intercept"] + (z * p["coef"]).sum(axis=1))))


def tolist(p):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()}


def metric(met, model, key):
    r = met[(met.model == model) & (met.metric == key)].iloc[0]
    return {"est": float(r.est), "lo": float(r.lo), "hi": float(r.hi)}


def main():
    m21 = load_module("21_npj_prediction")
    m22 = load_module("22_loh_prediction")
    rng = np.random.default_rng(SEED)
    lines = []

    # ---- リズム
    d = m21.load_data()
    cols_r = m21.VIDEO
    cv_r = pd.read_csv(OUT / "npj_cv_predictions.csv")
    pen = mode_value(cv_r[cv_r.model == "R2 変数そのまま（直線）"].drop_duplicates(["repeat", "fold"]).penalizer)
    pr = fit_rhythm(d, cols_r, pen)
    # 確認: 書き出した式と lifelines の予測が一致するか
    lib = m21.predict_cox(m21.fit_cox(d, cols_r, pen), d, cols_r)
    diff_r = float(np.abs(rhythm_risk(pr, d[cols_r].to_numpy(float)) - lib).max())
    boots_r = []
    for _ in range(N_BOOT):
        idx = rng.integers(0, len(d), len(d))
        try:
            boots_r.append(tolist(fit_rhythm(d.iloc[idx].reset_index(drop=True), cols_r, pen)))
        except Exception:
            pass
    ins_r = rhythm_risk(pr, d[cols_r].to_numpy(float))
    norm = pd.read_csv(STEP1 / "npj_person.csv")
    norm = norm[norm.person_id.isin(d.person_id)]
    qs = np.linspace(0, 100, 101)
    asym_q = {"percentile": qs.tolist(),
              "cadence_asym_10": np.percentile(norm.Cadence_asymmetry_10, qs).tolist(),
              "duty_asym_10": np.percentile(norm.Duty_factor_asymmetry_10, qs).tolist()}
    rtypes = json.loads((STEP3 / "types_rhythm.json").read_text(encoding="utf-8"))
    met_r, spr_r = pd.read_csv(OUT / "npj_metrics.csv"), pd.read_csv(OUT / "npj_spread.csv")
    name_r = "R2 変数そのまま（直線）"

    # ---- フォーム
    f = pd.read_csv(STEP3 / "form_features.csv")
    cols_f = m22.ANGLES
    cv_f = pd.read_csv(OUT / "loh_cv_predictions.csv")
    C = mode_value(cv_f[cv_f.model == "F2 変数そのまま（4つの角度）"].drop_duplicates(["repeat", "fold"]).C)
    pf = fit_form(f, cols_f, C)
    lib_f = m22.predict_logit(m22.fit_logit(f, cols_f, C), f, cols_f)
    diff_f = float(np.abs(form_risk(pf, f[cols_f].to_numpy(float)) - lib_f).max())
    persons = f.pid.unique()
    rows_of = {p: np.flatnonzero(f.pid.to_numpy() == p) for p in persons}
    boots_f = []
    for _ in range(N_BOOT):
        rows = np.concatenate([rows_of[p] for p in rng.choice(persons, len(persons), replace=True)])
        bd = f.iloc[rows].reset_index(drop=True)
        if bd.injured.nunique() == 2:
            boots_f.append(tolist(fit_form(bd, cols_f, C)))
    ins_f = form_risk(pf, f[cols_f].to_numpy(float))
    ftypes = json.loads((STEP3 / "types_form.json").read_text(encoding="utf-8"))
    met_f, spr_f = pd.read_csv(OUT / "loh_metrics.csv"), pd.read_csv(OUT / "loh_spread.csv")
    name_f = "F2 変数そのまま（4つの角度）"

    export = {
        "created": "2026-09-17（ステップ5-3）",
        "note": "予測はタイプを使わず、測った値をそのまま使う（ステップ5-2の比較で採用）。どちらも精度は限られるので、幅と精度を必ず一緒に表示する",
        "rhythm": {
            "outcome": "52週以内に初めてケガをする予測（持久系ランナーの研究参加者のデータから）",
            "source": "npj 2026（Wu ほか）140名、12か月の前向き追跡。走り方は参加時にトレッドミル時速10kmで測定",
            "model": "罰則付き Cox 回帰（L2）",
            "features": cols_r,
            "feature_labels": {"cadence_10": "ピッチ（歩/分）", "duty_10": "Duty factor（接地時間÷1歩の時間）",
                               "cadence_asym_10": "ピッチの左右差（npj の正規化値）", "duty_asym_10": "Duty factor の左右差（npj の正規化値）",
                               "rearfoot_10": "踵で接地（1/0）", "alt_strike": "左右で接地の仕方が違う（1/0）"},
            "penalizer": pen,
            "formula": "z = (x − scaler_mean) / scaler_sd; lp = Σ coef × (z − center); risk = 1 − S0_52w ^ exp(lp)",
            "params": tolist(pr),
            "contribution": "coef × (z − center)。正なら予測を上げ、負なら下げる",
            "bootstrap_params": boots_r,
            "asymmetry_quantiles": asym_q,
            "speed_adjust_per_kmh": rtypes["speed_adjust_per_kmh"],
            "performance_cv": {"auc_52w": metric(met_r, name_r, "auc_52w"), "c_index": metric(met_r, name_r, "c_index"),
                               "brier_52w": metric(met_r, name_r, "brier_52w"),
                               "prediction_spread_p10_p90": [float(spr_r[spr_r.model == name_r].p10.iloc[0]), float(spr_r[spr_r.model == name_r].p90.iloc[0])],
                               "judgement": "AUC の95%信頼区間が0.5を含む。動画で測れるリズムの値だけでは、ケガをする人としない人をほとんど見分けられない"},
            "in_sample_prediction_range": [float(ins_r.min()), float(ins_r.max())],
            "overall_injury_52w": float(m21.km_risk(d.time, d.event)),
        },
        "form": {
            "outcome": "12か月でケガをする予測（脚ごと。レクリエーションランナーの研究参加者のデータから）",
            "source": "Loh 2025 81名（137脚）、12か月の前向き追跡。フォームは参加時にトレッドミルの2D動画で測定",
            "model": "重み付きの罰則付きロジスティック回帰（L2）。重み = 1 ÷ その人の脚の数",
            "features": cols_f,
            "feature_labels": {"CPD": "骨盤の傾き（°）", "HADD": "股関節の内転の値（°、大きいほど内転が小さいと解釈）",
                               "KF": "膝の屈曲（°）", "KA": "膝の外反の値（°、正のとき膝が外に開くと解釈）"},
            "C": C,
            "formula": "z = (x − scaler_mean) / scaler_sd; risk = 1 / (1 + exp(−(intercept + Σ coef × z)))",
            "params": tolist(pf),
            "contribution": "coef × z。正なら予測を上げ、負なら下げる",
            "bootstrap_params": boots_f,
            "speed_adjust_per_kmh": ftypes["speed_adjust_per_kmh"],
            "angle_direction_assumption": ftypes["angle_direction_assumption"],
            "performance_cv": {"auc": metric(met_f, name_f, "auc"), "brier": metric(met_f, name_f, "brier"),
                               "calibration_slope": metric(met_f, name_f, "calib_slope"),
                               "prediction_spread_p10_p90": [float(spr_f[spr_f.model == name_f].p10.iloc[0]), float(spr_f[spr_f.model == name_f].p90.iloc[0])],
                               "judgement": "AUC の95%信頼区間の下限が0.5を超える。当てずっぽうよりは良いが、弱い予測。キャリブレーションの傾きが1未満で、予測はやや極端に出る"},
            "in_sample_prediction_range": [float(ins_f.min()), float(ins_f.max())],
            "overall_injury_12m": float(np.average(f.injured, weights=f.weight)),
        },
    }
    (OUT / "prediction_models.json").write_text(json.dumps(export, ensure_ascii=False, indent=1), encoding="utf-8")

    lines += [
        "== プロトタイプ用の予測モデル ==",
        f"[リズム] penalizer={pen}（5-2 で最も多く選ばれた値）、S0(52週)={pr['S0_52w']:.4f}",
        "  係数（標準化1SDあたり、Cox）: " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(cols_r, pr["coef"])),
        f"  書き出した式と lifelines の予測の最大差: {diff_r:.2e}",
        f"  全データでの予測の範囲: {ins_r.min():.1%}〜{ins_r.max():.1%}（参加者全体の52週以内のケガ {export['rhythm']['overall_injury_52w']:.1%}）",
        f"  ブートストラップの係数: {len(boots_r)} 組",
        f"[フォーム] C={C}（5-2 で最も多く選ばれた値）、切片 {pf['intercept']:+.3f}",
        "  係数（標準化1SDあたり、ロジスティック）: " + ", ".join(f"{c} {v:+.3f}" for c, v in zip(cols_f, pf["coef"])),
        f"  書き出した式と scikit-learn の予測の最大差: {diff_f:.2e}",
        f"  全データでの予測の範囲: {ins_f.min():.1%}〜{ins_f.max():.1%}（参加者全体の12か月のケガ {export['form']['overall_injury_12m']:.1%}）",
        f"  ブートストラップの係数: {len(boots_f)} 組",
    ]
    # 例：参加者の平均的な値と、完成イメージのダミーの値で計算してみる
    ex_f = np.array([[7.8, 76.1, 36.2, -3.9], [5.2, 80.5, 39.8, 2.1]])
    rf = form_risk(pf, ex_f)
    bf = np.array([form_risk({k: np.array(v) if isinstance(v, list) else v for k, v in b.items()}, ex_f) for b in boots_f])
    lines.append("  例（完成イメージの左脚・右脚のダミーの値）: " + ", ".join(
        f"{lab} {r:.0%}（ブートストラップ 2.5〜97.5%: {np.percentile(bf[:, i], 2.5):.0%}〜{np.percentile(bf[:, i], 97.5):.0%}）" for i, (lab, r) in enumerate(zip(["左脚", "右脚"], rf))))
    text = "\n".join(lines)
    (OUT / "prediction_models_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
