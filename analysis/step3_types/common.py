"""ステップ3で共通に使うパスと設定。"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
STEP1 = HERE.parent / "step1_data" / "outputs"

SEED = 20260917
N_BOOT = 200
MIN_TYPE_SIZE = 20
MIN_JACCARD = 0.6
HORIZON_WEEKS = 52

# 項目の組み合わせ（詳細設計の A / B / C）
VARIANTS = {
    "A": ["cadence_10", "duty_10"],
    "B": ["cadence_10", "duty_10", "cadence_asym_pct_10", "duty_asym_pct_10", "rearfoot_10", "alt_strike"],
    "C": ["cadence_10"],
}
# 画面に使う組み合わせ（9/17 ユーザーの判断で A から B に変更）
SELECTED_VARIANT = "B"
# 時速12kmで分け直す感度分析用の対応
VARIANTS_12 = {
    "A": ["cadence_12", "duty_12"],
    "B": ["cadence_12", "duty_12", "cadence_asym_pct_12", "duty_asym_pct_12", "rearfoot_12", "alt_strike"],
    "C": ["cadence_12"],
}
FEATURE_LABELS = {
    "cadence_10": "ピッチ（歩/分）",
    "duty_10": "Duty factor（接地時間÷1歩の時間）",
    "cadence_asym_pct_10": "ピッチの左右差（パーセンタイル）",
    "duty_asym_pct_10": "Duty factor の左右差（パーセンタイル）",
    "rearfoot_10": "踵から接地",
    "alt_strike": "左右で接地の仕方が違う",
}


def setup_font():
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

# ------------------------------------------------------------------ 3-2 フォームのタイプ（Loh 2025）
FORM_VARIANTS = {
    "A": ["CPD", "HADD", "KF", "KA"],   # 骨盤の傾き、股関節の内転、膝の屈曲、膝の外反
    "B": ["CPD", "HADD", "KF"],         # 膝の外反を除く（2D動画での誤差が大きい可能性）
}
FORM_SELECTED_VARIANT = "A"  # ステップ2の結果で A / B を選ぶ。それまでは A
FORM_MIN_TYPE_SIZE = 15      # 重み付きの人数（9/17 ユーザー了承）
FORM_N_BOOT = 200
FORM_LABELS = {"CPD": "骨盤の傾き", "HADD": "股関節の内転", "KF": "膝の屈曲", "KA": "膝の外反"}
