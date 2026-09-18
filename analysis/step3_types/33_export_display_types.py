"""ステップ3-3：画面に使うタイプ（9/17 ユーザーが案1を選択）の判定用ファイルを書き出す。

  リズム  : ピッチ（2区分、中央値）× 接地の仕方（踵／踵以外）→ 4タイプ（npj 2026、140名）
  フォーム: 骨盤の傾き（2区分、重み付き中央値）× 膝の屈曲（2区分、重み付き中央値）→ 4タイプ（Loh 2025、137脚、脚ごと）

入力: outputs/explore_candidates.json, rhythm_features.csv, explore_form_data.csv, types_rhythm.json, types_form.json
出力: outputs/types_display.json, outputs/types_display_check.txt

判定のしかた（プロトタイプ側で同じ計算をする）:
  1. 時速10km以外で撮った場合は、速度の補正で時速10km相当に直す（リズム: npj の 10→12 km/h、フォーム: Loh 2025 の回帰の傾き）
  2. 閾値と比べてタイプを決める（閾値より大きければ「高い」側）
  3. 値が「閾値の95%信頼区間（人単位のブートストラップ2000回）」に入っていれば、「どちらのタイプにも近い」と表示する
"""
import json

import numpy as np
import pandas as pd

from common import OUT, SEED

RHYTHM_ID = "grid:cadence_102×rearfoot_102"
FORM_ID = "grid:CPD2×KF2"
N_BOOT = 2000


def wmedian(x, w):
    o = np.argsort(x)
    cw = (np.cumsum(w[o]) - 0.5 * w[o]) / w.sum()
    return float(np.interp(0.5, cw, x[o]))


def boot_median_ci(x, w, pids, rng):
    persons = np.unique(pids)
    rows_of = {p: np.flatnonzero(pids == p) for p in persons}
    vals = []
    for _ in range(N_BOOT):
        rows = np.concatenate([rows_of[p] for p in rng.choice(persons, len(persons), replace=True)])
        vals.append(wmedian(x[rows], w[rows]))
    return [float(v) for v in np.percentile(vals, [2.5, 97.5])]


def summarize(df, w, labels, names, feats):
    out = []
    for t in sorted(names):
        m = labels == t
        out.append({"label": int(t), "name": names[t], "share": float(w[m].sum() / w.sum()), "n_units": int(m.sum()),
                    "n_person_equiv": float(w[m].sum()),
                    "mean": {f: float(np.average(df[f][m], weights=w[m])) for f in feats}})
    return out


def main():
    rng = np.random.default_rng(SEED)
    defs = json.loads((OUT / "explore_candidates.json").read_text(encoding="utf-8"))
    rtypes = json.loads((OUT / "types_rhythm.json").read_text(encoding="utf-8"))
    ftypes = json.loads((OUT / "types_form.json").read_text(encoding="utf-8"))

    r = pd.read_csv(OUT / "rhythm_features.csv")
    r = r[r.imputed_running == 0].reset_index(drop=True)
    wr = np.ones(len(r))
    dr = defs["rhythm"][RHYTHM_ID]
    lr = np.array(dr["labels"])
    thr_cad = dr["params"]["thresholds"]["cadence_10"][0]
    ci_cad = boot_median_ci(r.cadence_10.to_numpy(float), wr, r.person_id.to_numpy(), rng)

    f = pd.read_csv(OUT / "explore_form_data.csv")
    wf = f.weight.to_numpy()
    df_ = defs["form"][FORM_ID]
    lf = np.array(df_["labels"])
    thr_cpd = df_["params"]["thresholds"]["CPD"][0]
    thr_kf = df_["params"]["thresholds"]["KF"][0]
    ci_cpd = boot_median_ci(f.CPD.to_numpy(float), wf, f.pid.to_numpy(), rng)
    ci_kf = boot_median_ci(f.KF.to_numpy(float), wf, f.pid.to_numpy(), rng)

    export = {
        "decided": "2026-09-17 ユーザーが探索の案1を選択（リズム・フォームとも）",
        "note": "タイプは走り方の特徴の説明。ケガの予測には使わない（予測は測った値をそのまま使うモデル）",
        "rhythm": {
            "source": "npj 2026（Wu ほか）140名、トレッドミル時速10kmの床反力（参加時）",
            "unit": "人",
            "rule": "type = 2 × (cadence_10 > cadence 閾値) + rearfoot_10（1 = 踵で接地）",
            "axes": {"x": {"feature": "cadence_10", "label": "ピッチ（歩/分、時速10km）", "threshold": thr_cad, "threshold_ci95": ci_cad,
                           "low": "ローピッチ", "high": "ハイピッチ"},
                     "y": {"feature": "rearfoot_10", "label": "接地の仕方", "threshold": 0.5, "low": "踵以外で接地", "high": "踵で接地"}},
            "speed_adjust_per_kmh": {"cadence_10": rtypes["speed_adjust_per_kmh"]["cadence_10"]},
            "types": summarize(r, wr, lr, {int(k): v for k, v in dr["type_names"].items()}, ["cadence_10", "duty_10", "rearfoot_10"]),
            "distribution": {"cadence_10": {"mean": float(r.cadence_10.mean()), "sd": float(r.cadence_10.std()),
                                            "p10": float(r.cadence_10.quantile(0.1)), "p90": float(r.cadence_10.quantile(0.9))},
                             "rearfoot_share": float(r.rearfoot_10.mean())},
            "near_boundary_rule": "cadence_10 が threshold_ci95 の範囲内なら「ローピッチとハイピッチのどちらにも近い」",
            "stability_mean_jaccard": None,
        },
        "form": {
            "source": "Loh 2025 81名（137脚）、トレッドミルの2D動画（立脚中期、14歩の平均、参加時）",
            "unit": "脚（左右それぞれ判定）。重み = 1 ÷ その人の脚の数",
            "rule": "type = 2 × (CPD > CPD 閾値) + (KF > KF 閾値)",
            "axes": {"x": {"feature": "CPD", "label": "骨盤の傾き（°）", "threshold": thr_cpd, "threshold_ci95": ci_cpd, "low": "骨盤が安定", "high": "骨盤が落ちる"},
                     "y": {"feature": "KF", "label": "膝の屈曲（°）", "threshold": thr_kf, "threshold_ci95": ci_kf, "low": "膝の曲がりが浅い", "high": "膝の曲がりが深い"}},
            "speed_adjust_per_kmh": {"CPD": ftypes["speed_adjust_per_kmh"]["CPD"], "KF": ftypes["speed_adjust_per_kmh"]["KF"]},
            "types": summarize(f, wf, lf, {int(k): v for k, v in df_["type_names"].items()}, ["CPD", "KF"]),
            "distribution": {a: {"mean": float(np.average(f[a], weights=wf)), "sd": float(np.sqrt(np.average((f[a] - np.average(f[a], weights=wf)) ** 2, weights=wf)))}
                             for a in ("CPD", "KF")},
            "near_boundary_rule": "CPD または KF が、それぞれの threshold_ci95 の範囲内なら、その軸について「どちらにも近い」",
            "angle_note": "CPD は正のとき反対側の骨盤が下がる。KF は大きいほど膝が深く曲がる。角度の定義は Loh の2D動画解析に合わせる（ステップ2）",
        },
    }
    cands_r = pd.read_csv(OUT / "explore_rhythm_candidates.csv").set_index("id")
    cands_f = pd.read_csv(OUT / "explore_form_candidates.csv").set_index("id")
    export["rhythm"]["stability_mean_jaccard"] = float(cands_r.loc[RHYTHM_ID, "mean_jaccard"])
    export["form"]["stability_mean_jaccard"] = float(cands_f.loc[FORM_ID, "mean_jaccard"])
    (OUT / "types_display.json").write_text(json.dumps(export, ensure_ascii=False, indent=2), encoding="utf-8")

    near_r = ((r.cadence_10 >= ci_cad[0]) & (r.cadence_10 <= ci_cad[1])).mean()
    near_f = (((f.CPD >= ci_cpd[0]) & (f.CPD <= ci_cpd[1])) | ((f.KF >= ci_kf[0]) & (f.KF <= ci_kf[1])))
    lines = [
        "== 画面に使うタイプ ==",
        f"[リズム] ピッチの閾値 {thr_cad:.1f} 歩/分（95%CI {ci_cad[0]:.1f}〜{ci_cad[1]:.1f}）× 接地の仕方",
        *[f"  {t['label']} {t['name']}: {t['share']:.0%}（{t['n_units']}名）ピッチ平均 {t['mean']['cadence_10']:.1f}, Duty factor 平均 {t['mean']['duty_10']:.3f}" for t in export["rhythm"]["types"]],
        f"  「どちらにも近い」になる参加者: {near_r:.0%}",
        f"[フォーム] 骨盤の傾きの閾値 {thr_cpd:.2f}°（95%CI {ci_cpd[0]:.2f}〜{ci_cpd[1]:.2f}）× 膝の屈曲の閾値 {thr_kf:.2f}°（95%CI {ci_kf[0]:.2f}〜{ci_kf[1]:.2f}）",
        *[f"  {t['label']} {t['name']}: {t['share']:.0%}（{t['n_units']}脚）骨盤の傾き平均 {t['mean']['CPD']:.1f}°, 膝の屈曲平均 {t['mean']['KF']:.1f}°" for t in export["form"]["types"]],
        f"  どちらかの軸で「どちらにも近い」になる脚（重み付き）: {np.average(near_f, weights=wf):.0%}",
    ]
    text = "\n".join(lines)
    (OUT / "types_display_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
