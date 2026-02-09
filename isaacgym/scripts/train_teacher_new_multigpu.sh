#!/bin/sh
set -e

SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
REPO_ROOT="$(CDPATH= cd -- "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="$REPO_ROOT/isaacgym/src:$REPO_ROOT:$PYTHONPATH"

# Number of GPUs to use (adjust as needed)
NUM_GPUS=${NUM_GPUS:-$(nvidia-smi -L | wc -l)}

torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPUS \
    -m intermimic.run \
    --task InterMimic \
    --cfg_env isaacgym/src/intermimic/data/cfg/omomo_train_new.yaml \
    --cfg_train isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml \
    --headless \
    --output checkpoints \
    --multi_gpu
