"""npj 2026（Wu ほか）の週ごとデータに人のIDを付け、週ごとの表と1人1行の表を作る。

入力: analysis_data/npj2026_wu/europepmc_PMC12987969_supplementaryFiles.zip の MOESM3（258列）
出力: outputs/npj_weekly.parquet, outputs/npj_person.csv
"""
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ZIP = ROOT / "analysis_data/npj2026_wu/europepmc_PMC12987969_supplementaryFiles.zip"
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)

# 参加時に1回だけ測った走り方（床反力から計算）。動画で測れるものに印を付ける
RUNNING_TIMING = [
    "Step_frequency_10", "Step_frequency_12", "Contact_time_10", "Contact_time_12",
    "Flight_time_10", "Flight_time_12", "Duty_factor_10", "Duty_factor_12",
    "Cadence_asymmetry_10", "Cadence_asymmetry_12", "Duty_factor_asymmetry_10", "Duty_factor_asymmetry_12",
    "Alt_strike",
]
RUNNING_LOADING = [
    "VALR_10", "VALR_12", "VILR_10", "VILR_12", "Impact_peak_10", "Impact_peak_12",
    "VALR_asymmetry_10", "VALR_asymmetry_12", "VILR_asymmetry_10", "VILR_asymmetry_12",
    "Impact_peak_asymmetry_10", "Impact_peak_asymmetry_12",
]


def load_weekly() -> pd.DataFrame:
    with zipfile.ZipFile(ZIP) as z:
        name = next(n for n in z.namelist() if "MOESM3" in n)
        with z.open(name) as f:
            return pd.read_excel(f)


def add_person_id(d: pd.DataFrame) -> pd.DataFrame:
    """遺伝子型は人ごとに固定なので、SNP＋性別＋身長の組み合わせで人を区別する。
    行は人ごとに連続して並んでいるので、組み合わせが変わった行を人の切れ目とみなす。"""
    snps = [c for c in d.columns if c.startswith("rs")]
    key = d[snps + ["sex", "height"]].round(6).astype(str).agg("|".join, axis=1)
    block = (key != key.shift()).cumsum() - 1
    d = d.copy()
    d.insert(0, "person_id", block.astype(int))
    d.insert(1, "week_index", d.groupby("person_id").cumcount())
    return d


def person_table(d: pd.DataFrame) -> pd.DataFrame:
    g = d.groupby("person_id")
    first = g.first()
    out = pd.DataFrame(index=first.index)
    out["sex"] = first["sex"]
    out["n_weeks"] = g.size()
    out["n_injury_weeks"] = g["RRI"].sum()
    out["any_injury"] = (out["n_injury_weeks"] > 0).astype(int)
    out["injury_weeks_per_100"] = out["n_injury_weeks"] / out["n_weeks"] * 100
    out["first_injury_week"] = g.apply(lambda s: s["week_index"][s["RRI"] == 1].min() if s["RRI"].any() else np.nan)
    # 参加時の値（走り方は1人1値、2値ある人は最初の値）
    for c in RUNNING_TIMING + RUNNING_LOADING + ["lower_limb_days_total", "past_stress_injury", "Age", "BMI", "Mass"]:
        out[c] = first[c]
    out["n_distinct_timing_values"] = g["Step_frequency_10"].nunique()
    return out.reset_index()


def main():
    d = add_person_id(load_weekly())
    d.to_parquet(OUT / "npj_weekly.parquet", index=False)
    p = person_table(d)
    p.to_csv(OUT / "npj_person.csv", index=False)

    print(f"週ごと: {len(d)} 行, 人: {d.person_id.nunique()} 名")
    print(f"1人あたりの週数: 中央値 {p.n_weeks.median():.0f}（{p.n_weeks.min()}〜{p.n_weeks.max()}）")
    print(f"1回以上ケガ: {p.any_injury.sum()} 名 ({p.any_injury.mean():.0%})")
    print(f"ケガの週: {int(d.RRI.sum())} 週")
    print(f"走り方の値が2つある人: {(p.n_distinct_timing_values > 1).sum()} 名")
    print(f"性別の内訳（0/1）: {p.sex.value_counts().to_dict()}")


if __name__ == "__main__":
    main()
