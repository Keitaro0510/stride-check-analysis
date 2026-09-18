#!/usr/bin/env bash
# ステップ5（予測モデルの比較、論文の AUC の再現、プロトタイプ用の予測モデルの書き出し）をまとめて実行する。出力は outputs/
# 前提: ステップ1とステップ3（../step1_data/run.sh, ../step3_types/run.sh）を実行済み
# 使い方: bash run.sh            … すべて（約10分）
#         bash run.sh eval-only  … 保存した予測から評価とレポートだけやり直す（約2分）
set -euo pipefail
cd "$(dirname "$0")"
if [[ "${1:-all}" == "eval-only" ]]; then
  python3 21_npj_prediction.py --eval-only
  python3 22_loh_prediction.py --eval-only
else
  python3 21_npj_prediction.py
  python3 22_loh_prediction.py
  python3 23_npj_paper_replication.py
fi
python3 24_report.py
python3 25_export_prediction_models.py
