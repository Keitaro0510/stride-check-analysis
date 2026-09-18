#!/usr/bin/env bash
# ステップ6（プロトタイプの画面）を作り直す。数秒で終わる
# 前提: ステップ1・3・4・5 を実行済み（../stepN_xxx/outputs/ を読む）。Node.js（テスト用）
# 使い方: bash run.sh
set -euo pipefail
cd "$(dirname "$0")"
python3 61_build_app_data.py      # app/data/app_data.js、サンプル3名、tests/golden.json
node tests/test_model.mjs         # app/model.js が Python の基準の実装と一致するか
python3 62_build_artifact.py      # outputs/stride_check.html（1ファイルにまとめた画面）
