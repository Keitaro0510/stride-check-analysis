#!/usr/bin/env bash
# Pose estimation for every video in Running_mp4 (4 in parallel); skips finished ones.
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$(dirname "$0")"
mkdir -p outputs/mp
ls "$ROOT"/Running_mp4/*/*.mp4 | while read -r v; do
  s=$(basename "$(dirname "$v")"); n=$(basename "$v" _anonymized.mp4)
  out="outputs/mp/${s}_${n}.json"
  [[ -f $out ]] || echo "$v $out"
done | xargs -P 4 -n 2 sh -c '.venv/bin/python run_pose_py.py "$0" "$1" 2>/dev/null | tail -1'
