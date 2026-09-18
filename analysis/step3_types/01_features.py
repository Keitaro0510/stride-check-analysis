"""ステップ3-1：リズムのタイプ分けに使う、1人1行の特徴量と結果の表を作る。

入力: step1_data/outputs/npj_person.csv（正規化のまま）, npj_person_raw.csv（元の単位）, npj_weekly.parquet
出力: outputs/rhythm_features.csv, outputs/rhythm_data_check.txt

データの確認（論文の本文より）:
  走り方は最初の測定会で1回だけ測った。ただし停電のため、2名は約4か月後に測定（誰かは特定できない）、
  1名は同じ性別の他の参加者の中央値で埋めた。埋めた人は、タイミングの値が同じ性別の他の人の中央値と一致するかで探す。
  確認の結果、ID 70 と 71（身長・体重・年齢が違う別人）が、タイミングの全項目で完全に同じ値で、
  女性の中央値とほぼ一致した（遺伝子の値も同じで、これも埋めた値と思われる）。論文では1名だが、
  どちらも実際の測定値ではない可能性が高いので、2名とも imputed_running = 1 とし、タイプ分けから除く。
結果の定義:
  time  = 初めてケガをした週（1始まり）。ケガがなければ追跡した週数（打ち切り）
  event = 追跡中に1回以上ケガをしたか
"""
import numpy as np
import pandas as pd

from common import OUT, STEP1

TIMING_NORM = ["Step_frequency_10", "Contact_time_10", "Flight_time_10", "Duty_factor_10",
               "Step_frequency_12", "Contact_time_12", "Flight_time_12", "Duty_factor_12"]


def find_imputed(norm: pd.DataFrame) -> pd.DataFrame:
    """同じ性別の「他の参加者」の中央値と、タイミングの値がすべて一致する人を探す。"""
    rows = []
    for i, r in norm.iterrows():
        others = norm[(norm.sex == r.sex) & (norm.index != i)]
        match = [np.isclose(r[c], others[c].median(), atol=1e-9) for c in TIMING_NORM]
        rows.append({"person_id": r.person_id, "n_match": int(sum(match))})
    m = pd.DataFrame(rows)
    return m[m.n_match >= len(TIMING_NORM) // 2]


def main():
    norm = pd.read_csv(STEP1 / "npj_person.csv")
    raw = pd.read_csv(STEP1 / "npj_person_raw.csv")
    weekly = pd.read_parquet(STEP1 / "npj_weekly.parquet")
    first = weekly.sort_values(["person_id", "week_index"]).groupby("person_id").first()

    imputed = find_imputed(norm)
    d = pd.DataFrame({"person_id": raw.person_id})
    d["sex"] = raw.sex  # 0 = 男性
    # リズム（元の単位）
    d["cadence_10"] = raw.Step_frequency_10
    d["duty_10"] = raw.Duty_factor_10
    d["contact_10"] = raw.Contact_time_10
    d["flight_10"] = raw.Flight_time_10
    d["cadence_12"] = raw.Step_frequency_12
    d["duty_12"] = raw.Duty_factor_12
    d["contact_12"] = raw.Contact_time_12
    d["flight_12"] = raw.Flight_time_12
    # 元の単位に戻せない項目はパーセンタイル（0〜100）
    d["cadence_asym_pct_10"] = norm.Cadence_asymmetry_10.rank(pct=True) * 100
    d["duty_asym_pct_10"] = norm.Duty_factor_asymmetry_10.rank(pct=True) * 100
    # 補足資料: 踵から接地しない人は負荷率を0とした → 負荷率 > 0 を踵接地の目印にする
    d["rearfoot_10"] = (norm.VALR_10 > 0).astype(int)
    d["cadence_asym_pct_12"] = norm.Cadence_asymmetry_12.rank(pct=True) * 100
    d["duty_asym_pct_12"] = norm.Duty_factor_asymmetry_12.rank(pct=True) * 100
    d["rearfoot_12"] = (norm.VALR_12 > 0).astype(int)
    d["alt_strike"] = norm.Alt_strike.astype(int)
    # 調整に使う項目（正規化のまま。回帰の調整なので単位は不要）
    d["height_norm"] = first.loc[d.person_id, "height"].to_numpy()
    d["injury_days_prev_year"] = raw.lower_limb_days_total
    d["past_stress_injury"] = raw.past_stress_injury.astype(int)
    d["run_hours_norm"] = first.loc[d.person_id, "average_run_hours"].to_numpy()
    d["past_month_distance_norm"] = first.loc[d.person_id, "past_month_distance"].to_numpy()
    # 結果
    d["n_weeks"] = raw.n_weeks
    d["n_injury_weeks"] = raw.n_injury_weeks
    d["event"] = raw.any_injury
    d["time"] = np.where(raw.any_injury == 1, raw.first_injury_week + 1, raw.n_weeks)
    d["injured_week1"] = (raw.first_injury_week == 0).astype(int)
    d["imputed_running"] = d.person_id.isin(imputed.person_id).astype(int)
    d.to_csv(OUT / "rhythm_features.csv", index=False)

    dup = norm[norm.duplicated(subset=TIMING_NORM, keep=False)].person_id.tolist()
    lines = [
        "== ステップ3-1 データの確認 ==",
        f"タイミングの全項目が他の人と完全に一致する人: {dup}",
        f"人数: {len(d)}",
        f"走り方を中央値で埋めたと思われる人: {len(imputed)} 名 {imputed.to_dict('records')}",
        f"ケガ（1回以上）: {d.event.sum()} 名 / 追跡の最初の週にケガ: {d.injured_week1.sum()} 名 / 追跡4週未満: {(d.n_weeks < 4).sum()} 名",
        f"追跡した週数: 中央値 {d.n_weeks.median():.0f}（{d.n_weeks.min()}〜{d.n_weeks.max()}）",
        f"踵接地: {d.rearfoot_10.sum()} 名 / 左右で接地の仕方が違う: {d.alt_strike.sum()} 名",
        "",
        "時速10kmのリズム（平均 ± SD）:",
        d[["cadence_10", "duty_10", "contact_10", "flight_10"]].agg(["mean", "std", "min", "max"]).round(3).to_string(),
        "",
        f"ピッチと Duty factor の相関（10km/h）: {d.cadence_10.corr(d.duty_10):.2f}",
    ]
    text = "\n".join(lines)
    (OUT / "rhythm_data_check.txt").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
