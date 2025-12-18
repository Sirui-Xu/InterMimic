#!/bin/bash
# Test InterMimic policy inference on IsaacLab
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Check for ISAACLAB_PATH
if [ -z "$ISAACLAB_PATH" ]; then
    echo "[InterMimic] Please export ISAACLAB_PATH to your IsaacLab installation before running this script." >&2
    echo "[InterMimic] Example: export ISAACLAB_PATH=/path/to/IsaacLab" >&2
    exit 1
fi

ISAACLAB_SETUP="$ISAACLAB_PATH/isaaclab.sh"
if [ ! -f "$ISAACLAB_SETUP" ]; then
    echo "[InterMimic] Could not find isaaclab.sh at $ISAACLAB_SETUP" >&2
    exit 1
fi

# Set up paths
export INTERMIMIC_PATH="$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT:$REPO_ROOT/isaacgym/src:$REPO_ROOT/isaaclab/src:$REPO_ROOT/isaaclab/examples:$PYTHONPATH"

# Build arguments array
ARGS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint)
            ARGS+=("--checkpoint" "$2")
            shift 2
            ;;
        --config)
            ARGS+=("--config" "$2")
            shift 2
            ;;
        --num_envs)
            ARGS+=("--num_envs" "$2")
            shift 2
            ;;
        --headless)
            ARGS+=("--headless")
            shift
            ;;
        --test_steps)
            ARGS+=("--test_steps" "$2")
            shift 2
            ;;
        --no-continuous)
            ARGS+=("--no-continuous")
            shift
            ;;
        --record-video)
            ARGS+=("--record-video" "$2")
            shift 2
            ;;
        --video-fps)
            ARGS+=("--video-fps" "$2")
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --checkpoint PATH    Path to checkpoint (default: checkpoints/smplx_teachers/sub2.pth)"
            echo "  --config PATH        Path to training YAML config (e.g., isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml)"
            echo "  --num_envs N         Number of environments (default: 16)"
            echo "  --headless           Run without rendering"
            echo "  --test_steps N       Number of test steps (default: 100, for single episode mode)"
            echo "  --no-continuous      Disable continuous mode (run single episode, continuous is ON by default)"
            echo "  --record-video PATH  Record video to PATH (MP4 file, disables continuous mode)"
            echo "  --video-fps N        Frame rate for video recording (default: 30)"
            echo "  --help               Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0 --checkpoint checkpoints/smplx_teachers/sub2.pth --num_envs 16"
            echo "  $0 --checkpoint checkpoints/smplx_teachers/sub2.pth --config isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml"
            echo "  $0 --headless --num_envs 32"
            echo "  $0 --no-continuous --checkpoint checkpoints/smplx_teachers/sub2.pth  # Run single episode"
            echo "  $0 --record-video output.mp4 --checkpoint checkpoints/smplx_teachers/sub2.pth"
            echo ""
            echo "Note:"
            echo "  - Continuous mode is ON by default (runs until Ctrl+C with auto-reset)"
            echo "  - Use --no-continuous or --record-video to run a single episode"
            echo "  - Requires ISAACLAB_PATH to be set"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Run the test using IsaacLab's Python environment
"$ISAACLAB_SETUP" -p "$REPO_ROOT/isaaclab/examples/test_policy_inference.py" "${ARGS[@]}"
