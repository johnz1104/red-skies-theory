#!/usr/bin/env bash
# Copy the code (and entry files) to a VM, or bring a run back.
#   cloud/sync.sh up   ubuntu@<vm-ip>            # code + runs/entries -> ~/gtcw
#   cloud/sync.sh down ubuntu@<vm-ip> runs/e46   # a run directory -> local
# rsync carries uncommitted changes and local entry files, which a clone would not.
set -euo pipefail
cd "$(dirname "$0")/.."
direction=${1:?up or down}; host=${2:?user@host}
case "$direction" in
  up)
    rsync -az --delete \
      --exclude .git --exclude .venv --exclude '__pycache__' --exclude '.mypy_cache' \
      --exclude '.ruff_cache' --exclude '.pytest_cache' --exclude runs \
      ./ "$host:gtcw/"
    rsync -az runs/entries/ "$host:gtcw/runs/entries/" ;;
  down)
    run=${3:?run directory such as runs/e46}
    mkdir -p "$run"
    rsync -az "$host:gtcw/$run/" "$run/" ;;
  *) echo "usage: $0 up|down user@host [run]" >&2; exit 2 ;;
esac
