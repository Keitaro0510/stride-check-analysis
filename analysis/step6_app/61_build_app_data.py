"""ステップ6：アプリ用のデータ（app/data/app_data.js）、サンプル3名、ゴールデンテストの期待値を作る。

入力:
  step3_types/outputs/types_display.json, types_form.json, rhythm_features.csv
  step5_evidence/outputs/prediction_models.json（フォームの部分だけ使う）
  step4_load/outputs/load.json, load_final_models.json, load_feature_reference.csv, marker_features.csv
  step1_data/outputs/fukuchi_subjects.csv
出力:
  app/data/app_data.js          window.APP_DATA = {...}（ブラウザは fetch なしで読める）
  app/data/samples/sample_*.json 仕様書の形式の measures
  tests/golden.json             入力と期待する出力（app_reference.py で計算）
  outputs/build_check.txt       確認の記録

サンプル3名（Fukuchi 2017、CC BY 4.0。2.5 と 3.5 m/s の値を時速10kmに内挿。接地の仕方は記録の RFSI35/LFSI35）:
  1. ローピッチ × 踵で接地
  2. ハイピッチ × 踵以外で接地
  3. 少なくとも片脚が「骨盤が落ちる × 膝の曲がりが浅い」（左右でタイプが違う人を優先）
  各条件で、区切りに近くなく、値の範囲外の印がなく、リズムの値がタイプの中心に最も近い人を選ぶ
"""
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

import app_reference as ref

HERE = Path(__file__).resolve().parent
A = HERE.parent
APP_DATA = HERE / "app" / "data"
(APP_DATA / "samples").mkdir(parents=True, exist_ok=True)
(HERE / "tests").mkdir(exist_ok=True)
(HERE / "outputs").mkdir(exist_ok=True)
V10 = 10 / 3.6


def load_json(p):
    return json.loads(Path(p).read_text(encoding="utf-8"))


def build_data():
    td = load_json(A / "step3_types/outputs/types_display.json")
    tf = load_json(A / "step3_types/outputs/types_form.json")
    pm = load_json(A / "step5_evidence/outputs/prediction_models.json")["form"]
    ld = load_json(A / "step4_load/outputs/load.json")
    lf = load_json(A / "step4_load/outputs/load_final_models.json")
    fref = pd.read_csv(A / "step4_load/outputs/load_feature_reference.csv").set_index("feature")
    rh = pd.read_csv(A / "step3_types/outputs/rhythm_features.csv")
    rh = rh[rh.imputed_running == 0]

    loh = tf["reference_distributions"]["Loh 2025 ケガなし（参加時）"]
    rhythm_dist = {"cadence_spm": {"mean": float(rh.cadence_10.mean()), "sd": float(rh.cadence_10.std())},
                   "duty": {"mean": float(rh.duty_10.mean()), "sd": float(rh.duty_10.std())},
                   "contact_s": {"mean": float(rh.contact_10.mean()), "sd": float(rh.contact_10.std())},
                   "flight_s": {"mean": float(rh.flight_10.mean()), "sd": float(rh.flight_10.std())},
                   "pelvis_vertical_osc_mm": {"mean": float(fref.loc["pelvis_vertical_osc", "mean"]), "sd": float(fref.loc["pelvis_vertical_osc", "sd"])}}
    form_dist = {a: loh[a] for a in ("CPD", "HADD", "KF", "KA")}
    side_dist = {f: {"mean": float(fref.loc[f, "mean"]), "sd": float(fref.loc[f, "sd"])}
                 for f in ("knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride")}
    data = {
        "version": "2026-09-17",
        "speed_range_kmh": [8.0, 13.5],
        "types": {"rhythm": td["rhythm"], "form": td["form"]},
        "form_prediction": {k: pm[k] for k in ("features", "params", "bootstrap_params", "speed_adjust_per_kmh", "performance_cv", "overall_injury_12m", "angle_direction_assumption")},
        "loads": {k: {"label": r["label"], "target_label": r["target_label"], "model": r["model"], "reference_mean": r["reference_10kmh"]["mean"],
                      "error_pct": r["error_pct_of_reference"], "stars": r["stars"], "r2_extra": r["cv"]["r2_extra"],
                      "individual_difference_estimable": r["individual_difference_estimable"],
                      "cadence_sim": {str(x["cadence_change_pct"]): x["load_change_pct"] for x in r["cadence_sim"]}, "show_cadence_sim": r["show_cadence_sim"]}
                  for k, r in ld["regions"].items()},
        "load_defaults": {"mass_kg": lf["reference_input_10kmh"]["mass_kg"], "height_cm": lf["reference_input_10kmh"]["height_cm"]},
        "distributions": {"rhythm": rhythm_dist, "form": form_dist, "side": side_dist},
        "plausibility": {**rhythm_dist, **form_dist, **side_dist},
        "sources": {
            "npj2026": "Wu ほか 2026, npj Digital Medicine（持久系ランナー142名、12か月の前向き追跡、CC BY 4.0）",
            "loh2025": "Loh ほか 2025, Int J Sports Med（ランナー81名、12か月の前向き追跡、CC BY-NC）",
            "fukuchi2017": "Fukuchi ほか 2017, PeerJ（ランナー39名、3D動作と床反力、CC BY 4.0）",
        },
    }
    return data


def fukuchi_10kmh():
    mf = pd.read_csv(A / "step4_load/outputs/marker_features.csv")
    mf = mf[~mf.qc_exclude]
    subj = pd.read_csv(A / "step1_data/outputs/fukuchi_subjects.csv").set_index("subject")
    cols = ["CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride",
            "pelvis_vertical_osc", "contact_s", "flight_s", "cadence_spm", "duty", "n_steps"]
    people = []
    for s, g in mf.groupby("subject"):
        legs = {}
        for side, gs in g.groupby("side"):
            gs = gs.set_index("speed_ms")
            if 2.5 in gs.index and 3.5 in gs.index:
                w = (V10 - 2.5) / 1.0
                legs[side] = {c: float(gs.loc[2.5, c] + w * (gs.loc[3.5, c] - gs.loc[2.5, c])) for c in cols}
        if set(legs) != {"L", "R"}:
            continue
        rf_r, rf_l = subj.loc[s, "RFSI35"] == "Rearfoot", subj.loc[s, "LFSI35"] == "Rearfoot"
        rhythm = {k: float(np.mean([legs["L"][k], legs["R"][k]])) for k in ("cadence_spm", "contact_s", "flight_s", "duty", "pelvis_vertical_osc")}
        m = {
            "video": {"rear": None, "side": None, "note": "サンプル：研究データのランナー（Fukuchi ほか 2017、CC BY 4.0）。マーカーを2Dに投影して計算した値を、時速10kmに内挿"},
            "subject": {"speed_kmh": 10.0, "height_cm": float(subj.loc[s, "Height"]), "mass_kg": float(subj.loc[s, "Mass"]),
                        "sex": "M" if subj.loc[s, "Gender"] == "M" else "F"},
            "rhythm": {"cadence_spm": rhythm["cadence_spm"], "contact_s": rhythm["contact_s"], "flight_s": rhythm["flight_s"], "duty": rhythm["duty"],
                       "foot_strike": "rearfoot" if (rf_r and rf_l) else "non_rearfoot", "alt_strike": bool(rf_r != rf_l),
                       "cadence_asym_pct": None, "duty_asym_pct": None, "pelvis_vertical_osc_mm": rhythm["pelvis_vertical_osc"]},
            "legs": {side: {**{k: legs[side][k] for k in ("CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride")},
                            "n_steps": int(round(legs[side]["n_steps"]))} for side in ("L", "R")},
            "quality": {"step_sd": None, "low_confidence_frames_pct": None, "notes": []},
        }
        people.append((int(s), m))
    return people


def choose_samples(people, data):
    rows = []
    rd = data["types"]["rhythm"]
    for s, m in people:
        e = ref.evaluate(m, data)
        rt = e["rhythm_type"]
        center = rd["types"][rt["label"]]["mean"]["cadence_10"]
        rows.append({"subject": s, "m": m, "e": e, "rt": rt["label"], "near": rt["near_boundary"]["cadence"],
                     "legs": {k: v["type"]["label"] for k, v in e["legs"].items()},
                     "leg_near": any(any(v["type"]["near_boundary"].values()) for v in e["legs"].values()),
                     "n_flags": len(e["flags"]), "dist": abs(rt["cadence_10"] - center)})
    df = pd.DataFrame(rows)
    ok = df[~df.near.astype(bool) & (df.n_flags == 0)]
    picks = []
    s1 = ok[ok.rt == 1].sort_values("dist")
    s2 = ok[ok.rt == 2].sort_values("dist")
    picks.append(s1.iloc[0] if len(s1) else None)
    picks.append(s2.iloc[0] if len(s2) else None)
    used = {p.subject for p in picks if p is not None}
    cand3 = ok[~ok.subject.isin(used) & ok.legs.apply(lambda d: 2 in d.values())].copy()
    cand3["differ"] = cand3.legs.apply(lambda d: len(set(d.values())) > 1)
    cand3 = cand3.sort_values(["differ", "leg_near", "dist"], ascending=[False, True, True])
    picks.append(cand3.iloc[0] if len(cand3) else None)
    return picks, df


def golden_cases(samples, data):
    cases = [("sample_" + str(i + 1), copy.deepcopy(m)) for i, m in enumerate(samples)]
    base = copy.deepcopy(samples[0])
    # 区切りの付近、速度の補正、欠け、範囲外、タイプの組み合わせを変えた入力
    c = copy.deepcopy(base); c["rhythm"]["cadence_spm"] = data["types"]["rhythm"]["axes"]["x"]["threshold"]; cases.append(("cadence_at_threshold", c))
    c = copy.deepcopy(base); c["subject"]["speed_kmh"] = 12.0; cases.append(("speed_12kmh", c))
    c = copy.deepcopy(base); c["subject"]["speed_kmh"] = 14.0; cases.append(("speed_out_of_range", c))
    c = copy.deepcopy(base); c["subject"]["speed_kmh"] = None; c["subject"]["height_cm"] = None; c["subject"]["mass_kg"] = None; cases.append(("subject_missing", c))
    c = copy.deepcopy(base); c["legs"]["L"]["KA"] = None; cases.append(("leg_missing_KA", c))
    c = copy.deepcopy(base); c["legs"]["L"]["KF"] = None; c["legs"]["R"]["KF"] = None; cases.append(("kf_not_measured", c))
    c = copy.deepcopy(base); c["rhythm"]["pelvis_vertical_osc_mm"] = None; c["legs"]["R"]["overstride"] = None; cases.append(("load_inputs_missing", c))
    c = copy.deepcopy(base); c["rhythm"]["cadence_spm"] = 230.0; c["legs"]["L"]["KF"] = 70.0; cases.append(("out_of_range_values", c))
    c = copy.deepcopy(base); c["rhythm"]["foot_strike"] = "non_rearfoot"; cases.append(("non_rearfoot", c))
    tx, ty = data["types"]["form"]["axes"]["x"]["threshold"], data["types"]["form"]["axes"]["y"]["threshold"]
    for i, (cpd, kf) in enumerate([(tx - 2, ty - 3), (tx - 2, ty + 3), (tx + 2, ty - 3), (tx + 2, ty + 3), (tx, ty)]):
        c = copy.deepcopy(base); c["legs"]["L"]["CPD"] = cpd; c["legs"]["L"]["KF"] = kf; cases.append((f"form_grid_{i}", c))
    c = copy.deepcopy(base); del c["legs"]["R"]; cases.append(("left_leg_only", c))
    c = copy.deepcopy(base); c["rhythm"]["cadence_spm"] = None; cases.append(("rhythm_missing_cadence", c))
    return [{"name": n, "input": m, "expected": ref.evaluate(m, data)} for n, m in cases]


def main():
    data = build_data()
    people = fukuchi_10kmh()
    picks, table = choose_samples(people, data)
    samples = []
    lines = ["== ステップ6 アプリ用データ ==", f"サンプル候補（時速10kmに内挿できた人）: {len(people)} 名"]
    for i, p in enumerate(picks):
        if p is None:
            lines.append(f"サンプル{i + 1}: 条件に合う人がいない")
            continue
        m = p.m
        m["video"]["sample_id"] = i + 1
        samples.append(m)
        (APP_DATA / "samples" / f"sample_{i + 1}.json").write_text(json.dumps(m, ensure_ascii=False, indent=1), encoding="utf-8")
        e = p.e
        lines.append(f"サンプル{i + 1}: Fukuchi {p.subject}番 リズム「{e['rhythm_type']['name']}」（ピッチ {e['rhythm_type']['cadence_10']:.1f}）、"
                     + "、".join(f"{k}脚「{v['type']['name']}」予測 {v['prediction']['risk']:.0%}（{v['prediction']['lo']:.0%}〜{v['prediction']['hi']:.0%}）" for k, v in e["legs"].items())
                     + "、負担 " + "、".join(f"{data['loads'][k]['label']} {v['pct']:+.0f}%" if v["estimable"] else f"{data['loads'][k]['label']} 推定できない" for k, v in e["loads"].items()))
    data["samples"] = samples
    (APP_DATA / "app_data.js").write_text("window.APP_DATA = " + json.dumps(data, ensure_ascii=False) + ";\n", encoding="utf-8")
    golden = golden_cases(samples, data)
    (HERE / "tests" / "golden.json").write_text(json.dumps(golden, ensure_ascii=False, indent=1), encoding="utf-8")

    # 基準の実装が、元のモデルと同じ予測を出すかの確認
    import importlib.util
    import sys

    sys.path.insert(0, str(A / "step4_load"))
    spec = importlib.util.spec_from_file_location("sim44", A / "step4_load/44_reference_and_sim.py")
    sim44 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sim44)
    d = pd.read_csv(A / "step4_load/outputs/load_dataset.csv")
    diffs = {}
    for key, r in data["loads"].items():
        t = load_json(A / "step4_load/outputs/load.json")["regions"][key]["target"]
        pipe = sim44.fit_ridge(d, t, sim44.L3)
        mine = np.array([ref._ridge(r["model"], row) for row in d[sim44.L3].to_dict("records")])
        diffs[key] = float(np.nanmax(np.abs(mine - pipe.predict(d[sim44.L3]))))
    lines.append("負担: 基準の実装と scikit-learn のパイプラインの予測の最大差: " + ", ".join(f"{k} {v:.1e}" for k, v in diffs.items()))
    lines.append(f"ゴールデンテスト: {len(golden)} 件")
    size = (APP_DATA / "app_data.js").stat().st_size
    lines.append(f"app_data.js: {size / 1024:.0f} KB")
    text = "\n".join(lines)
    (HERE / "outputs" / "build_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
