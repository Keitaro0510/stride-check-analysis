#!/usr/bin/env bash
# ステップ3をまとめて実行する。出力は outputs/
# 前提: ステップ1（../step1_data/run.sh）を実行済み
# 使い方: bash run.sh          … 3-1 と 3-2 の両方（約10分）
#         bash run.sh rhythm   … 3-1 リズムだけ（約2分）
#         bash run.sh form     … 3-2 フォームだけ（約8分）
#         bash run.sh explore  … 3-3 見やすいタイプの探索だけ（約1分。3-1・3-2 の成果物が必要）
set -euo pipefail
cd "$(dirname "$0")"
part="${1:-all}"
if [[ "$part" == "all" || "$part" == "rhythm" ]]; then
  python3 01_features.py
  python3 02_cluster.py
  python3 03_outcomes.py
  python3 04_export.py
  python3 05_report.py
fi
if [[ "$part" == "all" || "$part" == "form" ]]; then
  python3 11_form_features.py
  python3 12_form_cluster.py
  python3 13_form_outcomes.py
  python3 14_form_export.py
  python3 15_form_report.py
fi
if [[ "$part" == "all" || "$part" == "explore" ]]; then
  python3 31_explore_types.py
  python3 32_explore_gallery.py
  python3 33_export_display_types.py
fi
