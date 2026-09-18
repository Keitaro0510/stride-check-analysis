#!/usr/bin/env bash
# ステップ1をまとめて実行する。出力は outputs/
set -euo pipefail
cd "$(dirname "$0")"
python3 prepare_npj.py
python3 prepare_loh.py
python3 prepare_fukuchi.py
python3 denormalize_npj.py
