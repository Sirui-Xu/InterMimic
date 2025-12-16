#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run_distill \
    --task InterMimic_All \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_all_test.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_all.yaml \
    --test \
    --checkpoint checkpoints/smplx_student/nn/mimic.pth \
    --num_envs 16
