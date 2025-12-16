"""IsaacLab data replay demo for InterMimic."""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
ISAACLAB_SRC = REPO_ROOT / "isaaclab" / "src"
ISAACGYM_SRC = REPO_ROOT / "isaacgym" / "src"

os.environ.setdefault("INTERMIMIC_PATH", str(REPO_ROOT))
for candidate in (REPO_ROOT, ISAACGYM_SRC, ISAACLAB_SRC):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def main():
    parser = argparse.ArgumentParser(description="InterMimic IsaacLab data replay demo")
    parser.add_argument("--num-envs", type=int, default=4, help="Number of parallel environments")
    parser.add_argument("--headless", action="store_true", help="Run Isaac Sim headless")
    parser.add_argument(
        "--motion-dir",
        type=str,
        default="InterAct/OMOMO_new",
        help="Motion directory relative to INTERMIMIC_PATH",
    )
    parser.add_argument("--no-playback", action="store_true", help="Disable dataset playback")
    parser.add_argument(
        "--record-video",
        type=str,
        default=None,
        help="Optional path to an MP4 file for saving a viewport capture (requires imageio).",
    )
    parser.add_argument("--video-fps", type=int, default=30, help="Frame rate for --record-video captures.")
    args = parser.parse_args()

    video_writer = None
    record_path: Optional[Path] = None
    if args.record_video:
        record_path = Path(args.record_video).expanduser()
        record_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            import imageio  # noqa: F401
        except ImportError as exc:
            raise SystemExit(
                "[InterMimic] The --record-video flag requires the 'imageio' package. "
                "Install it inside your IsaacLab Python environment (pip install imageio imageio-ffmpeg)."
            ) from exc

    print("=" * 70)
    print("InterMimic IsaacLab Data Replay")
    print("=" * 70)

    print("\n1. Importing IsaacLab modules...")
    from isaaclab.app import AppLauncher

    app_launcher = AppLauncher({"headless": args.headless})
    simulation_app = app_launcher.app
    print("   ✓ Isaac Sim started")

    print("\n2. Importing InterMimic modules...")
    from intermimic_lab import InterMimicEnv, InterMimicEnvCfg

    print("   ✓ Modules imported")

    print("\n3. Creating configuration...")
    cfg = InterMimicEnvCfg()
    cfg.num_envs = args.num_envs
    cfg.scene.num_envs = args.num_envs
    cfg.motion_file = args.motion_dir
    cfg.play_dataset = not args.no_playback
    cfg.replay_root_height_offset = 0.08
    print(f"   ✓ Config created ({cfg.num_envs} envs, playback={'on' if cfg.play_dataset else 'off'})")

    print("\n4. Creating environment (this may take a few moments)...")
    render_mode = "rgb_array" if args.record_video else (None if args.headless else "human")
    env = InterMimicEnv(cfg=cfg, render_mode=render_mode)
    print("   ✓ Environment created!")
    print(f"      - Device: {env.device}")
    print(f"      - Scene entities: {len(env.scene.keys())}")

    print("\n5. Streaming data...  (Ctrl+C to exit)")
    frame_idx = 0
    max_dataset_frames = env.dataset_max_length if cfg.play_dataset else None
    if record_path is not None:
        import imageio
        import numpy as np

        video_writer = imageio.get_writer(str(record_path), fps=args.video_fps)
        print(f"   ✓ Recording video to {record_path} @ {args.video_fps} FPS")

    try:
        while simulation_app.is_running():
            if cfg.play_dataset:
                env.play_dataset_step(frame_idx)
                frame_idx += 1
            else:
                env.sim.step()
                env.scene.update(env.physics_dt)
            if video_writer is not None:
                frame = env.render(recompute=True)
                if frame is not None:
                    frame_rgb = frame[..., :3] if frame.shape[-1] >= 3 else frame
                    if frame_rgb.dtype != np.uint8:
                        frame_rgb = np.clip(frame_rgb, 0.0, 1.0)
                        frame_rgb = (frame_rgb * 255).astype(np.uint8)
                    video_writer.append_data(frame_rgb)
            time.sleep(0.01)
            if max_dataset_frames is not None and frame_idx >= max_dataset_frames:
                print(f"\n   ✓ Completed dataset playback ({max_dataset_frames} frames).")
                break
    except KeyboardInterrupt:
        print("\n   Exiting...")
    finally:
        if video_writer is not None:
            video_writer.close()
        env.close()
        simulation_app.close()

    print("\n✅ Test completed!")


if __name__ == "__main__":
    main()
