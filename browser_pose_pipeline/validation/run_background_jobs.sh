#!/usr/bin/env bash
# Long jobs for the Twente validation, meant to run inside `screen` so they
# survive a closed SSH session.  Safe to re-run: the download resumes, pose
# estimation skips finished videos, extraction runs only after the checksum
# matches.  Log: outputs/background_jobs.log
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
DATA="$ROOT/data/twente_locomotion_6sensors_2022"
LOG="$HERE/outputs/background_jobs.log"
mkdir -p "$HERE/outputs/mp"
log() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

pose() {
  log "pose: start"
  ls "$ROOT"/Running_mp4/*/*.mp4 | while read -r v; do
    s=$(basename "$(dirname "$v")"); n=$(basename "$v" _anonymized.mp4)
    out="$HERE/outputs/mp/${s}_${n}.json"
    [[ -f $out ]] || echo "$v $out"
  done | xargs -P 8 -n 2 sh -c "cd '$HERE' && .venv/bin/python run_pose_py.py \"\$0\" \"\$1\" 2>/dev/null | tail -1" >> "$LOG"
  log "pose: done ($(ls "$HERE"/outputs/mp/Subj*.json | wc -l) files)"
}

download() {
  cd "$DATA/archives" || return 1
  if [[ ! -f Processed_data.rar ]]; then
    log "download: start"
    ./pdl.sh "https://zenodo.org/records/6457662/files/Processed_data.rar?download=1" Processed_data.rar 17916719552 12 >> "$LOG" 2>&1
  fi
  sum=$(md5sum Processed_data.rar | cut -d' ' -f1)
  if [[ $sum != c52802227f3aaf93551122d047232361 ]]; then log "download: MD5 MISMATCH ($sum)"; return 1; fi
  log "download: MD5 ok"
  if [[ ! -f "$DATA/.extracted" ]]; then
    log "extract: start"
    ~/.local/opt/7zip/7zz x -y -o"$DATA" Processed_data.rar > "$DATA/extract.log" 2>&1 && touch "$DATA/.extracted"
    log "extract: exit $? (see $DATA/extract.log)"
  fi
}

pose & download & wait
log "all jobs finished"
