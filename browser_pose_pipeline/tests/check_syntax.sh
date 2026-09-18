#!/usr/bin/env bash
set -euo pipefail
node --check "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/extract/extract.js"
