#!/usr/bin/env bash
# ステップ4（部位ごとの負担の推定）をまとめて実行する。出力は outputs/
# 前提: ステップ1（../step1_data/run.sh）と、ステップ3-2（../step3_types/run.sh form。Loh の角度の分布との比較に使う）を実行済み
# 所要時間: 約10分
set -euo pipefail
cd "$(dirname "$0")"
python3 41_marker_features.py
python3 42_load_targets.py
python3 43_load_models.py
python3 44_reference_and_sim.py
python3 45_export_report.py
