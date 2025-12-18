#!/usr/bin/env python3
"""Test policy inference on IsaacLab using Isaac Gym checkpoint.

This module contains the core testing logic for policy inference.
Can be run directly or imported from scripts.

Usage:
    python -m test_policy_inference --checkpoint checkpoints/smplx_teachers/sub2.pth --num_envs 16
"""

import os
import sys
import torch
import argparse
from typing import Optional, Dict
from pathlib import Path

# Add paths for imports
REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_SRC = REPO_ROOT / "isaaclab" / "src"
ISAACGYM_SRC = REPO_ROOT / "isaacgym" / "src"

os.environ.setdefault("INTERMIMIC_PATH", str(REPO_ROOT))
for candidate in (REPO_ROOT, ISAACGYM_SRC, ISAACLAB_SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def load_isaac_gym_checkpoint(checkpoint_path: str) -> Dict:
    """Load policy from Isaac Gym checkpoint.

    Args:
        checkpoint_path: Path to the .pth checkpoint file

    Returns:
        Dictionary containing checkpoint data
    """
    print(f"[Test] Loading checkpoint from: {checkpoint_path}")

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    print(f"[Test] Checkpoint keys: {list(checkpoint.keys())}")

    return checkpoint


def test_environment_creation(cfg, headless: bool = False, video_recording: bool = False):
    """Test environment creation and basic functionality.

    Args:
        cfg: Environment configuration
        headless: Whether to run without rendering
        video_recording: Whether video recording is enabled

    Returns:
        Created environment instance

    Note:
        intermimic_lab module must be imported AFTER AppLauncher is initialized
    """
    # Import must happen after AppLauncher is created (done in run_full_test_suite)
    from intermimic_lab import InterMimicEnv

    print(f"[Test] Creating environment with {cfg.num_envs} environments...")

    try:
        # Determine render mode: rgb_array for video recording or non-headless, None for headless
        if video_recording:
            render_mode = "rgb_array"
        elif headless:
            render_mode = None
        else:
            render_mode = "rgb_array"

        env = InterMimicEnv(cfg, render_mode=render_mode)
        print(f"[Test] Environment created successfully!")
        print(f"[Test] Observation space: {env.cfg.num_observations}")
        print(f"[Test] Action space: {env.cfg.num_actions}")
        print(f"[Test] Number of environments: {env.num_envs}")
        print(f"[Test] Device: {env.device}")
        return env
    except Exception as e:
        print(f"[Test] Failed to create environment: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_environment_stepping(env, num_steps: int = 10):
    """Test environment stepping with random actions.

    Args:
        env: Environment instance
        num_steps: Number of steps to test

    Returns:
        True if successful, False otherwise
    """
    print(f"\n[Test] Running environment steps...")

    try:
        # Reset environment
        obs, _ = env.reset()
        print(f"[Test] Environment reset successfully!")
        print(f"[Test] Initial observation shape: {obs['policy'].shape}")

        # Run steps with random actions
        for step in range(num_steps):
            # Generate random actions
            actions = torch.randn(env.num_envs, env.cfg.num_actions, device=env.device)

            # Step environment
            obs, rewards, terminated, truncated, info = env.step(actions)

            if step % 5 == 0:
                print(f"[Test] Step {step}: obs_shape={obs['policy'].shape}, "
                      f"reward_mean={rewards.mean().item():.3f}, "
                      f"reward_min={rewards.min().item():.3f}, "
                      f"reward_max={rewards.max().item():.3f}")

        print(f"\n[Test] Environment stepping successful!")
        return True

    except Exception as e:
        print(f"[Test] Failed during environment stepping: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_observation_dimensions(env):
    """Test and verify observation dimensions.

    Args:
        env: Environment instance

    Returns:
        True if dimensions match expected, False otherwise
    """
    print(f"\n[Test] Verifying observation dimensions...")

    try:
        # Get observation
        obs, _ = env.reset()
        obs_policy = obs['policy']

        # Check dimensions
        expected_obs_dim = env.cfg.num_observations
        actual_obs_dim = obs_policy.shape[1]

        print(f"[Test] Expected observation dim: {expected_obs_dim}")
        print(f"[Test] Actual observation dim: {actual_obs_dim}")

        # Check if _curr_obs is built correctly (should be 1211)
        if hasattr(env, '_curr_obs'):
            curr_obs_dim = env._curr_obs.shape[1]
            print(f"[Test] HOI observation dim (_curr_obs): {curr_obs_dim}")
            print(f"[Test] Expected HOI dim: 1211 (root:7, dof:306, body:676, obj:13, ig:156, contact:53)")

        if actual_obs_dim == expected_obs_dim:
            print(f"[Test] ✓ Observation dimensions match!")
            return True
        else:
            print(f"[Test] ✗ Observation dimension mismatch!")
            return False

    except Exception as e:
        print(f"[Test] Failed during observation dimension check: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_policy_inference_with_checkpoint(env, checkpoint_path: str, num_steps: int = 100, config_path: str = None,
                                          continuous: bool = True, video_path: str = None, video_fps: int = 30):
    """Test policy inference with loaded checkpoint.

    Args:
        env: Environment instance
        checkpoint_path: Path to checkpoint file
        num_steps: Number of steps to run per episode (ignored if continuous=True)
        config_path: Optional path to training YAML config
        continuous: If True, run continuously with auto-reset until Ctrl+C (default: True)
        video_path: Optional path to save video recording (disables continuous mode)
        video_fps: Frame rate for video recording (default: 30)

    Returns:
        True if successful, False otherwise
    """
    print(f"\n[Test] Testing policy inference with checkpoint...")

    # Video recording setup (matching data_replay.py scheme)
    video_writer = None
    record_path = None
    if video_path:
        continuous = False  # Disable continuous mode when recording
        record_path = Path(video_path).expanduser()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "[Test] The --record-video flag requires the 'imageio' package. "
                "Install it inside your IsaacLab Python environment (pip install imageio imageio-ffmpeg)."
            ) from exc

    if continuous:
        print(f"[Test] Running in continuous mode - press Ctrl+C to stop")
    elif video_path:
        print(f"[Test] Video recording mode - will record to {record_path}")

    try:
        # Import policy loader (after Isaac Sim initialized)
        from intermimic_lab.policy_loader import load_policy

        # Get observation and action shapes from environment config
        obs_shape = env.cfg.num_observations
        action_shape = env.cfg.num_actions
        print(f"[Test] Environment shapes - obs: {obs_shape}, actions: {action_shape}")

        # Load policy from checkpoint
        print(f"[Test] Loading policy from checkpoint...")
        policy = load_policy(checkpoint_path, obs_shape, action_shape, device=str(env.device), config_path=config_path)
        print(f"[Test] Policy loaded successfully!")

        # Reset environment
        obs, _ = env.reset()
        print(f"[Test] Environment reset successfully!")
        print(f"[Test] Initial observation shape: {obs['policy'].shape}")

        # Initialize video writer if recording (matching data_replay.py)
        if record_path is not None:
            import imageio
            import numpy as np
            video_writer = imageio.get_writer(str(record_path), fps=video_fps)
            print(f"[Test] ✓ Recording video to {record_path} @ {video_fps} FPS")

        if continuous:
            # Continuous mode: run until Ctrl+C
            # Track per-environment statistics
            num_envs = env.num_envs
            episode_count = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
            episode_rewards = torch.zeros(num_envs, dtype=torch.float32, device=env.device)
            episode_steps = torch.zeros(num_envs, dtype=torch.int32, device=env.device)
            total_step = 0

            try:
                while True:
                    # Get action from policy
                    obs_tensor = obs['policy']
                    action = policy.get_action(obs_tensor, deterministic=True)

                    # Step environment
                    obs, rewards, terminated, truncated, info = env.step(action)

                    # Accumulate rewards and steps for all environments
                    episode_rewards += rewards
                    episode_steps += 1
                    total_step += 1

                    # Log progress
                    if total_step % 100 == 0:
                        avg_reward = episode_rewards.mean().item() / max(episode_steps.float().mean().item(), 1.0)
                        print(f"[Step {total_step}] "
                              f"avg_reward={avg_reward:.3f}, "
                              f"action_mean={action.mean().item():.3f}, "
                              f"action_std={action.std().item():.3f}, "
                              f"completed_episodes={episode_count.sum().item()}")

                    # Check if any environment needs reset
                    done = terminated | truncated
                    if done.any():
                        # Get indices of environments that are done
                        done_ids = done.nonzero(as_tuple=False).squeeze(-1)

                        # Log stats for completed episodes
                        for env_id in done_ids:
                            env_id_item = env_id.item()
                            ep_reward = episode_rewards[env_id].item()
                            ep_steps = episode_steps[env_id].item()
                            ep_count = episode_count[env_id].item()
                            avg_reward = ep_reward / max(ep_steps, 1)

                            print(f"\n[Env {env_id_item}] Episode {ep_count} completed: "
                                  f"{ep_steps} steps, avg_reward={avg_reward:.3f}")

                            # Reset stats for this environment
                            episode_count[env_id] += 1
                            episode_rewards[env_id] = 0.0
                            episode_steps[env_id] = 0

                        # Reset only the environments that are done
                        # Note: IsaacLab's DirectRLEnv handles selective resets automatically
                        # The observations are already updated for reset environments

            except KeyboardInterrupt:
                print(f"\n\n[Test] Stopped by user (Ctrl+C)")
                print(f"[Test] Total episodes completed: {episode_count.sum().item()}")
                print(f"[Test] Total steps: {total_step}")
                print(f"[Test] Episodes per environment: {episode_count.cpu().tolist()}")
                return True

        else:
            # Fixed number of steps mode or video recording mode
            total_reward = 0.0
            # Track completion for each environment separately
            episode_completed = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
            step = 0
            while True:
                step += 1
                # Get action from policy
                obs_tensor = obs['policy']
                action = policy.get_action(obs_tensor, deterministic=True)

                # Step environment
                obs, rewards, terminated, truncated, info = env.step(action)

                # Capture video frame if recording (matching data_replay.py)
                if video_writer is not None:
                    frame = env.render(recompute=True)
                    if frame is not None:
                        # Convert to RGB if needed
                        frame_rgb = frame[..., :3] if frame.shape[-1] >= 3 else frame
                        # Convert to uint8 if needed
                        if frame_rgb.dtype != np.uint8:
                            frame_rgb = np.clip(frame_rgb, 0.0, 1.0)
                            frame_rgb = (frame_rgb * 255).astype(np.uint8)
                        video_writer.append_data(frame_rgb)

                # Accumulate rewards
                total_reward += rewards.mean().item()

                # Track which episodes have completed
                done = terminated | truncated
                episode_completed = episode_completed | done

                # Log when first environment completes
                if done.any() and episode_completed.sum() == done.sum():
                    completed_count = episode_completed.sum().item()
                    print(f"[Test] {completed_count} environment(s) completed at step {step}")

                # For video recording, stop after ALL episodes have completed at least once
                if video_writer and episode_completed.all():
                    print(f"[Test] All {env.num_envs} environments completed - stopping video recording")
                    break

                # Log progress
                if step % 20 == 0:
                    print(f"[Test] Step {step}/{num_steps}: "
                          f"reward_mean={rewards.mean().item():.3f}, "
                          f"action_mean={action.mean().item():.3f}, "
                          f"action_std={action.std().item():.3f}, "
                          f"completed={episode_completed.sum().item()}/{env.num_envs}")

            avg_reward = total_reward / (step + 1) if step >= 0 else 0.0
            print(f"\n[Test] Policy inference test completed!")
            print(f"[Test] Average reward over {step + 1} steps: {avg_reward:.3f}")
            print(f"[Test] ✓ Policy is generating actions and controlling the humanoid")

        return True

    except Exception as e:
        print(f"[Test] Failed during policy inference test: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Clean up video writer (matching data_replay.py)
        if video_writer is not None:
            video_writer.close()
            print(f"[Test] ✓ Video saved to {record_path}")


def run_full_test_suite(cfg, checkpoint_path: Optional[str] = None, config_path: Optional[str] = None,
                        headless: bool = False, simulation_app=None, continuous: bool = True,
                        video_path: str = None, video_fps: int = 30):
    """Run the full test suite.

    Args:
        cfg: Environment configuration
        checkpoint_path: Optional path to checkpoint file
        config_path: Optional path to training YAML config
        headless: Whether to run without rendering
        simulation_app: Optional simulation app instance (if None, will create one)
        continuous: If True, run policy inference continuously until Ctrl+C (default: True)
        video_path: Optional path to save video recording (disables continuous mode)
        video_fps: Frame rate for video recording (default: 30)

    Returns:
        True if all tests pass, False otherwise
    """
    print("="*80)
    print("InterMimic Policy Inference Test Suite - IsaacLab")
    print("="*80)

    # Initialize AppLauncher if not provided (before importing intermimic_lab)
    created_app = False
    if simulation_app is None:
        print("\n[Test] Initializing Isaac Sim...")
        from isaaclab.app import AppLauncher
        app_launcher = AppLauncher({"headless": headless})
        simulation_app = app_launcher.app
        created_app = True
        print("[Test] ✓ Isaac Sim initialized")
    else:
        print("\n[Test] Using existing Isaac Sim instance")

    # Test 1: Environment creation
    print("\n" + "="*80)
    print("Test 1: Environment Creation")
    print("="*80)
    env = test_environment_creation(cfg, headless, video_recording=(video_path is not None))
    if env is None:
        print("\n[Test] ✗ Environment creation failed!")
        simulation_app.close()
        return False
    print("[Test] ✓ Environment creation passed!")

    # Skip other tests if in continuous mode or video recording - go straight to policy inference
    if (continuous or video_path) and checkpoint_path:
        print("\n" + "="*80)
        if video_path:
            print("Video Recording Mode")
        else:
            print("Continuous Policy Inference Mode")
        print("="*80)
        policy_test = test_policy_inference_with_checkpoint(
            env, checkpoint_path, config_path=config_path,
            continuous=continuous, video_path=video_path, video_fps=video_fps
        )

        # Cleanup
        env.close()
        if created_app:
            simulation_app.close()

        return policy_test

    # Test 2: Observation dimensions
    print("\n" + "="*80)
    print("Test 2: Observation Dimensions")
    print("="*80)
    obs_test = test_observation_dimensions(env)
    if not obs_test:
        print("[Test] ✗ Observation dimension test failed!")
        return False
    print("[Test] ✓ Observation dimension test passed!")

    # Test 3: Environment stepping
    print("\n" + "="*80)
    print("Test 3: Environment Stepping")
    print("="*80)
    step_test = test_environment_stepping(env, num_steps=10)
    if not step_test:
        print("[Test] ✗ Environment stepping test failed!")
        return False
    print("[Test] ✓ Environment stepping test passed!")

    # Test 4: Policy inference (if checkpoint provided)
    if checkpoint_path:
        print("\n" + "="*80)
        print("Test 4: Policy Inference with Checkpoint")
        print("="*80)
        policy_test = test_policy_inference_with_checkpoint(
            env, checkpoint_path, num_steps=50, config_path=config_path,
            continuous=False, video_path=None, video_fps=video_fps
        )
        if not policy_test:
            print("[Test] ✗ Policy inference test failed!")
            return False
        print("[Test] ✓ Policy inference test passed!")

    # All tests passed
    print("\n" + "="*80)
    print("All tests passed successfully!")
    print("="*80)
    print("\nNext steps:")
    print("1. Implement policy network reconstruction from checkpoint")
    print("2. Add reference motion comparison for full observation")
    print("3. Validate policy behavior matches Isaac Gym")
    print("4. Run full evaluation on test dataset")

    # Cleanup
    env.close()
    if created_app:
        simulation_app.close()

    return True


def main():
    """Main entry point when run as a module."""
    parser = argparse.ArgumentParser(description="Test InterMimic policy inference on IsaacLab")
    parser.add_argument("--checkpoint", type=str,
                        default="checkpoints/smplx_teachers/sub2.pth",
                        help="Path to Isaac Gym checkpoint (relative to INTERMIMIC_PATH)")
    parser.add_argument("--config", type=str, default=None,
                        help="Path to training YAML config (e.g., isaacgym/src/intermimic/data/cfg/train/rlg/omomo.yaml)")
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of environments")
    parser.add_argument("--headless", action="store_true",
                        help="Run without rendering")
    parser.add_argument("--test_steps", type=int, default=100,
                        help="Number of steps to test (for single episode mode)")
    parser.add_argument("--no-continuous", action="store_true",
                        help="Disable continuous mode (run single episode instead)")
    parser.add_argument("--record-video", type=str, default=None,
                        help="Path to save video recording (disables continuous mode, requires imageio)")
    parser.add_argument("--video-fps", type=int, default=30,
                        help="Frame rate for video recording (default: 30)")
    args = parser.parse_args()

    # Continuous is enabled by default, unless --no-continuous or --record-video is specified
    continuous = not args.no_continuous and not args.record_video

    # Initialize AppLauncher FIRST
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app

    # NOW we can import intermimic_lab (after Isaac Sim is initialized)
    from intermimic_lab.config import INTERMIMIC_ENV_CFG

    # Update checkpoint path relative to REPO_ROOT
    checkpoint_path = REPO_ROOT / args.checkpoint

    # Update config path if provided
    config_path = None
    if args.config:
        config_path = REPO_ROOT / args.config if not Path(args.config).is_absolute() else args.config

    # Create environment configuration
    cfg = INTERMIMIC_ENV_CFG
    cfg.scene.num_envs = args.num_envs
    cfg.num_envs = args.num_envs
    # Enable motion dataset loading for proper reference observations
    cfg.play_dataset = True

    # Run test suite (pass simulation_app to avoid double initialization)
    success = run_full_test_suite(
        cfg=cfg,
        checkpoint_path=str(checkpoint_path) if checkpoint_path.exists() else None,
        config_path=str(config_path) if config_path else None,
        headless=args.headless,
        simulation_app=simulation_app,
        continuous=continuous,
        video_path=args.record_video,
        video_fps=args.video_fps
    )

    # Cleanup (since we created the app in main())
    simulation_app.close()

    # Exit with appropriate code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
