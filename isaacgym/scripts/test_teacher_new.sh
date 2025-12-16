#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_test_new.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --test \
    --checkpoint checkpoints/smplx_teachers_new/sub2.pth \
    --num_envs 16
