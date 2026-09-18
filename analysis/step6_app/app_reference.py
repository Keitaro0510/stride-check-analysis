"""ステップ6：画面の計算の「基準の実装」（Python）。JavaScript の app/model.js は、これと同じ結果を出す。

入力（measures）: 仕様書（analysis/step2_video/video_measurement_spec.md）の形式
  subject: speed_kmh, height_cm, mass_kg, sex
  rhythm : cadence_spm, contact_s, flight_s, duty, foot_strike（"rearfoot" / "non_rearfoot"）, alt_strike, pelvis_vertical_osc_mm, ...
  legs   : {"L": {CPD, HADD, KA, KF, knee_flex_contact, shank_angle_contact, foot_angle_contact, overstride}, "R": {...}}
出力（evaluate の返り値）:
  speed      : 速度の補正の情報
  rhythm_type: リズムのタイプ、区切りに近いか
  legs       : 脚ごとのフォームのタイプ、区切りに近いか、ケガの予測（点推定と 2.5〜97.5% の幅）
  loads      : 部位ごとの平均的なランナーとの差（%）、誤差の幅（%）、星、推定できるか、ピッチの変化
  flags      : 値のもっともらしさ（平均 ± 3SD の外）
"""
import math

TARGET_KMH = 10.0


def _num(x):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else float(x)


def speed_info(m, data):
    s = _num(m["subject"].get("speed_kmh"))
    assumed = s is None
    s = TARGET_KMH if assumed else s
    lo, hi = data["speed_range_kmh"]
    return {"speed_kmh": s, "assumed": assumed, "delta_kmh": s - TARGET_KMH, "out_of_range": not (lo <= s <= hi)}


def rhythm_type(m, data, sp):
    t = data["types"]["rhythm"]
    cad = _num(m["rhythm"].get("cadence_spm"))
    fs = m["rhythm"].get("foot_strike")
    if cad is None or fs not in ("rearfoot", "non_rearfoot"):
        return None
    cad10 = cad - t["speed_adjust_per_kmh"]["cadence_10"] * sp["delta_kmh"]
    thr = t["axes"]["x"]["threshold"]
    lo, hi = t["axes"]["x"]["threshold_ci95"]
    label = 2 * int(cad10 > thr) + int(fs == "rearfoot")
    return {"label": label, "name": t["types"][label]["name"], "share": t["types"][label]["share"],
            "cadence_10": cad10, "near_boundary": {"cadence": lo <= cad10 <= hi}}


def form_values_10(leg, data, sp):
    """KF は動画からは測れないので、無いときは研究データの中央値（タイプの KF のしきい値）で補う。"""
    adj = data["form_prediction"]["speed_adjust_per_kmh"]
    out, imputed = {}, []
    for a in ("CPD", "HADD", "KF", "KA"):
        v = _num(leg.get(a))
        out[a] = None if v is None else v - adj[a] * sp["delta_kmh"]
    if out["KF"] is None:
        out["KF"] = data["types"]["form"]["axes"]["y"]["threshold"]
        imputed.append("KF")
    return out, imputed


def form_type(v10, data):
    t = data["types"]["form"]
    if v10["CPD"] is None or v10["KF"] is None:
        return None
    tx, ty = t["axes"]["x"]["threshold"], t["axes"]["y"]["threshold"]
    label = 2 * int(v10["CPD"] > tx) + int(v10["KF"] > ty)
    return {"label": label, "name": t["types"][label]["name"], "share": t["types"][label]["share"],
            "near_boundary": {"CPD": t["axes"]["x"]["threshold_ci95"][0] <= v10["CPD"] <= t["axes"]["x"]["threshold_ci95"][1],
                              "KF": t["axes"]["y"]["threshold_ci95"][0] <= v10["KF"] <= t["axes"]["y"]["threshold_ci95"][1]}}


def _logistic(p, x):
    z = sum(c * (xi - mu) / sd for c, xi, mu, sd in zip(p["coef"], x, p["scaler_mean"], p["scaler_sd"]))
    return 1.0 / (1.0 + math.exp(-(p["intercept"] + z)))


def _percentile(values, q):
    """numpy の既定（線形補間）と同じ。"""
    v = sorted(values)
    pos = (len(v) - 1) * q / 100.0
    lo = math.floor(pos)
    hi = min(lo + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (pos - lo)


def form_prediction(v10, data):
    fp = data["form_prediction"]
    x = [v10[a] for a in fp["features"]]
    if any(v is None for v in x):
        return None
    boots = [_logistic(b, x) for b in fp["bootstrap_params"]]
    return {"risk": _logistic(fp["params"], x), "lo": _percentile(boots, 2.5), "hi": _percentile(boots, 97.5)}


def _ridge(model, inputs):
    total = model["intercept"]
    for f, med, mu, sc, c in zip(model["features"], model["impute_median"], model["scaler_mean"], model["scaler_scale"], model["coef"]):
        v = _num(inputs.get(f))
        v = med if v is None else v
        total += c * (v - mu) / sc
    return total


def load_inputs(m, leg, data):
    subj, rh = m["subject"], m["rhythm"]
    defaults = data["load_defaults"]
    height = _num(subj.get("height_cm"))
    mass = _num(subj.get("mass_kg"))
    speed = _num(subj.get("speed_kmh"))
    return {
        "speed_ms": (TARGET_KMH if speed is None else speed) / 3.6,
        "mass_kg": defaults["mass_kg"] if mass is None else mass,
        "height_cm": defaults["height_cm"] if height is None else height,
        "cadence_spm": rh.get("cadence_spm"), "duty": rh.get("duty"), "contact_s": rh.get("contact_s"),
        "pelvis_vertical_osc": rh.get("pelvis_vertical_osc_mm"),
        **{k: leg.get(k) for k in ("CPD", "HADD", "KA", "KF", "knee_flex_contact", "shank_angle_contact", "foot_angle_contact", "overstride")},
    }


def loads(m, data):
    out = {}
    legs = [s for s in ("L", "R") if s in m["legs"] and m["legs"][s]]
    for key, r in data["loads"].items():
        if not r["individual_difference_estimable"]:
            out[key] = {"estimable": False, "stars": r["stars"]}
            continue
        preds = [_ridge(r["model"], load_inputs(m, m["legs"][s], data)) for s in legs]
        value = sum(preds) / len(preds)
        ref = r["reference_mean"]
        out[key] = {"estimable": True, "value": value, "pct": (value - ref) / ref * 100, "error_pct": r["error_pct"], "stars": r["stars"],
                    "cadence_sim": r["cadence_sim"] if r["show_cadence_sim"] else None}
    return out


def flags(m, data):
    out = []
    dist = data["plausibility"]
    rh = m["rhythm"]
    for k in ("cadence_spm", "duty", "contact_s", "flight_s", "pelvis_vertical_osc_mm"):
        v = _num(rh.get(k))
        if v is not None and k in dist and abs(v - dist[k]["mean"]) > 3 * dist[k]["sd"]:
            out.append({"where": "rhythm", "item": k, "value": v})
    for s, leg in m["legs"].items():
        for k, v in leg.items():
            v = _num(v) if not isinstance(v, str) else None
            if v is not None and k in dist and abs(v - dist[k]["mean"]) > 3 * dist[k]["sd"]:
                out.append({"where": s, "item": k, "value": v})
    return out


def evaluate(m, data):
    sp = speed_info(m, data)
    legs = {}
    for s, leg in m["legs"].items():
        if not leg:
            continue
        v10, imputed = form_values_10(leg, data, sp)
        legs[s] = {"values_10": v10, "imputed": imputed, "type": form_type(v10, data), "prediction": form_prediction(v10, data)}
    return {"speed": sp, "rhythm_type": rhythm_type(m, data, sp), "legs": legs, "loads": loads(m, data), "flags": flags(m, data)}
