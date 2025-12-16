#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}" )" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [ -z "$ISAACLAB_PATH" ]; then
    echo "[InterMimic] Please export ISAACLAB_PATH to your IsaacLab installation before running this script." >&2
    exit 1
fi

ISAACLAB_SETUP="$ISAACLAB_PATH/isaaclab.sh"
if [ ! -f "$ISAACLAB_SETUP" ]; then
    echo "[InterMimic] Could not find isaaclab.sh at $ISAACLAB_SETUP" >&2
    exit 1
fi


export INTERMIMIC_PATH="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/isaacgym/src:$REPO_ROOT/isaaclab/src:$PYTHONPATH"

"$ISAACLAB_SETUP" -p "$REPO_ROOT/isaaclab/examples/data_replay.py" "$@"
