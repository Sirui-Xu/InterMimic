#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

python -m intermimic.run \
    --task InterMimicG1 \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_g1_29dof_with_hand.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo_g1_29dof_with_hand.yaml \
    --checkpoint checkpoints/g1/sub8.pth \
    --test \
    --num_envs 16
