"""ステップ5で共通に使うパス、設定、評価の関数。
（ステップ3のモジュールが `common` を使うので、名前がぶつからないよう ev_common にしている）
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "outputs"
OUT.mkdir(exist_ok=True)
STEP1 = HERE.parent / "step1_data" / "outputs"
STEP3_DIR = HERE.parent / "step3_types"
STEP3 = STEP3_DIR / "outputs"

SEED = 20260917
N_FOLDS = 5
N_REPEATS_NPJ = 20
N_REPEATS_LOH = 50
N_BOOT = 500
HORIZON = 52


def import_step3(name: str):
    """ステップ3のスクリプト（例: 02_cluster）をモジュールとして読み込む。"""
    import importlib

    if str(STEP3_DIR) not in sys.path:
        sys.path.insert(0, str(STEP3_DIR))
    return importlib.import_module(name)


def setup_font():
    plt.rcParams["font.family"] = ["Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False


def percentile_ci(values, alpha=0.05):
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    if len(v) == 0:
        return np.nan, np.nan
    return float(np.percentile(v, 100 * alpha / 2)), float(np.percentile(v, 100 * (1 - alpha / 2)))
