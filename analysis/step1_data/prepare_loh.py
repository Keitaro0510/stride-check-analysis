"""Loh 2025（前向き）と Loh & Kong 2026（横断）を、1脚1行と1人1行の表に整える。

Loh 2025
  入力: analysis_data/loh2025_prospective/*.xlsx（Data_left, Data_right）
  前提（著者に確認しないため、ここで決める）:
    - ラベルは Injured（人ごと、12か月以内のケガ）だけを使う。Injury 列は意味が不明なので使わない
    - 左足首の背屈（LAF）は約40°と約75°の値が混在しているので、足首の背屈は使わない
    - 足部の角度（FA, FI, RE, Tmax, RX）は動画で安定して測れない見込みなので、タイプ分けには使わない（表には残す）
    - 入力ミスと思われる値（ピッチ100未満）は欠損にする
  データの特徴:
    - ケガをしなかった55名は左右両脚の角度がある。ケガをした26名は、1名を除き片側の脚だけに角度がある
      （Injury 列が空欄の人は左、1 の人は右）。ケガをした側の脚だけを載せている可能性があるが、未確認
    - そのため、1人1行の表の角度は「角度のある側の平均」とし、左右差はタイプ分けに使わない
  出力: outputs/loh2025_limbs.csv, outputs/loh2025_person.csv

Loh & Kong 2026
  入力: analysis_data/lohkong2026_cross_sectional/*.xlsx
  出力: outputs/lk2026_limbs.csv（受傷した脚と、健常群の左右の脚）
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

ANGLES = ["HADD", "CPD", "KF", "AF", "KA", "FA", "FI", "FE", "Tmax", "FX"]
FORM_ANGLES = ["HADD", "CPD", "KF", "KA"]  # タイプ分けに使う4項目

# ------------------------------------------------------------------ Loh 2025
LOH_SHARED = {
    "Participant No.": "pid", "Injured": "injured", "Height (cm)": "height_cm", "Age (years)": "age",
    "Sex": "male", "Weight (kg)": "mass_kg", "Skeletal Muscle Mass (kg)": "muscle_kg", "BMI (kg/m2)": "bmi",
    "Percent Body Fat (%)": "body_fat_pct", "Treadmill running speed (km/h)": "treadmill_kmh",
    "Training speed (km/h)": "training_kmh", "Mileage (km/week)": "km_per_week", "Run experience (years)": "run_years",
    "Overhead Squat": "fms_squat", "Hurdle Step": "fms_hurdle", "Inline Lunge": "fms_lunge",
    "Shoulder Mobility": "fms_shoulder", "Active straight leg raise": "fms_aslr",
    "Trunk stability pushup": "fms_pushup", "Rotary stability": "fms_rotary", "ALL": "fms_total",
}
# Loh 2025 の足部の角度は Readme で RE（外返し）・RX（外返しの幅）と表記。Loh & Kong 2026 の FE・FX に名前をそろえる
LOH_ANGLE_ALIAS = {"RE": "FE", "RX": "FX"}


def _loh_side(df: pd.DataFrame, side: str) -> pd.DataFrame:
    s = side[0].upper()  # L / R
    out = df[list(LOH_SHARED)].rename(columns=LOH_SHARED)
    out["side"] = s
    for a in ["HADD", "CPD", "KF", "AF", "KA", "FA", "FI", "RE", "Tmax", "RX"]:
        out[LOH_ANGLE_ALIAS.get(a, a)] = df[f"{s}{a}"]
    grf = {
        "Cadence": "cadence_spm",
        "Step_Length (m)": "step_length_m",
        f"Stance_Time_{s} (s)": "stance_s",
        f"Swing_Time_{s} (s)": "swing_s",
        "NormalPeakForceL (BW)" if s == "L" else "Norm-PeakForce_R (BW)": "peak_force_bw",
        "NormLoading_Rate_L (BW/s)" if s == "L" else "Norm-Loading_Rate_R": "loading_rate_bw_s",
        f"Leg_Stiffness_{s} (N/m)": "leg_stiffness",
    }
    for src, dst in grf.items():
        out[dst] = df[src] if src in df.columns else np.nan
    return out


def prepare_loh2025():
    f = next((ROOT / "analysis_data/loh2025_prospective").glob("*.xlsx"))
    left = pd.read_excel(f, sheet_name="Data_left")
    right = pd.read_excel(f, sheet_name="Data_right")
    limbs = pd.concat([_loh_side(left, "left"), _loh_side(right, "right")], ignore_index=True)

    flags = []
    # 入力ミスと思われる値
    bad_cad = limbs["cadence_spm"] < 100
    limbs.loc[bad_cad, ["cadence_spm", "step_length_m"]] = np.nan
    flags.append(f"ピッチ100未満を欠損に: {sorted(limbs.loc[bad_cad, 'pid'].unique().tolist())}")
    # 足首の背屈: 左に約40°の値が混在
    limbs["AF_suspect"] = (limbs["side"] == "L") & (limbs["AF"] < 60)
    flags.append(f"左足首の背屈が60°未満（定義違いの疑い）: {int(limbs['AF_suspect'].sum())} 脚")
    limbs["has_angles"] = limbs[FORM_ANGLES].notna().all(axis=1)
    limbs["has_grf"] = limbs["stance_s"].notna()
    limbs.to_csv(OUT / "loh2025_limbs.csv", index=False)

    # 1人1行（角度は左右の平均と左右差）
    shared = limbs[limbs.side == "L"][list(LOH_SHARED.values())].set_index("pid")
    L = limbs[limbs.side == "L"].set_index("pid")
    R = limbs[limbs.side == "R"].set_index("pid")
    person = shared.copy()
    for a in FORM_ANGLES:
        person[a] = pd.concat([L[a], R[a]], axis=1).mean(axis=1)
    person["angle_sides"] = L[FORM_ANGLES].notna().all(axis=1).astype(int) + R[FORM_ANGLES].notna().all(axis=1).astype(int)
    person["angle_side"] = np.select([person["angle_sides"] == 2, L["HADD"].notna(), R["HADD"].notna()], ["LR", "L", "R"], "")
    person["cadence_spm"] = L["cadence_spm"]
    person["stance_s"] = (L["stance_s"] + R["stance_s"]) / 2
    person["loading_rate_bw_s"] = (L["loading_rate_bw_s"] + R["loading_rate_bw_s"]) / 2
    person["has_angles"] = person[FORM_ANGLES].notna().all(axis=1)
    person = person.reset_index()
    person.to_csv(OUT / "loh2025_person.csv", index=False)

    print("== Loh 2025 ==")
    print(f"人: {len(person)} 名（ケガ {int(person.injured.sum())} 名）")
    ct = pd.crosstab(person.angle_side, person.injured, margins=True)
    print("角度のある側 × ケガ:\n", ct.to_string())
    for x in flags:
        print(" -", x)


# ------------------------------------------------------------------ Loh & Kong 2026
LK_TYPE_REGION = {
    "Low back pain": "腰・股関節", "Gluteal pain": "腰・股関節", "Hip pain": "腰・股関節",
    "Quadriceps strain": "太もも・膝", "Hamstring strain": "太もも・膝", "Iliotibial band syndrome": "太もも・膝",
    "Patellofemoral pain": "太もも・膝", "Patella tendonitis": "太もも・膝", "Chondromalacia": "太もも・膝",
    "Medial knee pain": "太もも・膝",
    "Medial Tibial Stress Syndrome / Shin Splint": "下腿・足首・足", "Compartment pressure syndrome": "下腿・足首・足",
    "Ankle pain": "下腿・足首・足", "Ankle Pain": "下腿・足首・足", "Achilles tendinopathy": "下腿・足首・足",
    "Plantar fasciitis": "下腿・足首・足", "Metatarsalgia": "下腿・足首・足",
}


def prepare_lk2026():
    f = next((ROOT / "analysis_data/lohkong2026_cross_sectional").glob("*.xlsx"))
    ctrl = pd.read_excel(f, sheet_name="Controls (n = 44)").dropna(how="all")
    part = pd.read_excel(f, sheet_name="Injured Participants (n = 155)")
    part = part[part["Participant"].astype(str).str.match(r"^P\d+$")].copy()
    part["pid"] = part["Participant"].str[1:].astype(int)
    part["Injury"] = part["Injury"].replace({"Ankle Pain": "Ankle pain"})

    rows = []
    for _, r in ctrl.iterrows():
        for s in ("L", "R"):
            rows.append({"group": "control", "pid": f"C{int(r['No.'])}", "side": s, "sex": r["Sex"], "age": r["Age (years)"],
                         "injury": None, "region": None, **{a: r[f"{a}_{s}"] for a in ANGLES}})
    for sheet, s, idcol in (("Injured Kinematics (left)", "L", "No."), ("Injured Kinematics (right)", "R", "No")):
        k = pd.read_excel(f, sheet_name=sheet).dropna(how="all")
        k = k[pd.to_numeric(k[idcol], errors="coerce").notna()]
        k.columns = [c.replace("TMAX", "Tmax") for c in k.columns]
        for _, r in k.iterrows():
            pid = int(r[idcol])
            info = part[part.pid == pid]
            injury = info["Injury"].iloc[0] if len(info) else None
            rows.append({"group": "injured", "pid": f"P{pid}", "side": s,
                         "sex": info["Sex"].iloc[0] if len(info) else r["Sex"],
                         "age": info["Age"].iloc[0] if len(info) else np.nan,
                         "injury": injury, "region": LK_TYPE_REGION.get(injury),
                         **{a: r[f"{a}_{s}"] for a in ANGLES}})
    limbs = pd.DataFrame(rows)
    limbs.to_csv(OUT / "lk2026_limbs.csv", index=False)

    print("\n== Loh & Kong 2026 ==")
    print(f"脚: 健常 {int((limbs.group == 'control').sum())}、受傷 {int((limbs.group == 'injured').sum())}")
    print("受傷した脚のケガの種類:\n", limbs[limbs.group == "injured"]["injury"].value_counts(dropna=False).to_string())
    print("部位:\n", limbs[limbs.group == "injured"]["region"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    prepare_loh2025()
    prepare_lk2026()
