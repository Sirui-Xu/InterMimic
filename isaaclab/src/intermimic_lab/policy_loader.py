"""
Policy loader for Isaac Gym checkpoints in IsaacLab.

This module provides a lightweight wrapper around the existing Isaac Gym
player infrastructure, allowing policies trained in Isaac Gym to be used
in IsaacLab environments.
"""

import torch
from pathlib import Path
import sys

# Add isaacgym to path to import existing infrastructure
isaacgym_path = Path(__file__).parent.parent.parent.parent / "isaacgym" / "src"
if str(isaacgym_path) not in sys.path:
    sys.path.insert(0, str(isaacgym_path))

from intermimic.learning.intermimic_network_builder import InterMimicBuilder
from rl_games.algos_torch import torch_ext
from rl_games.algos_torch.running_mean_std import RunningMeanStd
import yaml
import os


class IsaacGymPolicyWrapper:
    """Wrapper for Isaac Gym trained policies to use in IsaacLab.

    This class reuses the Isaac Gym network loading infrastructure,
    providing a simple interface for policy inference.

    Args:
        checkpoint_path: Path to the .pth checkpoint file
        obs_shape: Observation space dimension
        action_shape: Action space dimension
        device: Device to run inference on (default: "cuda:0")
        config_path: Optional path to training YAML config (e.g., omomo.yaml).
                     If not provided, will search for config in standard locations.
    """

    def __init__(self, checkpoint_path: str, obs_shape: int, action_shape: int, device: str = "cuda:0", config_path: str = None):
        self.device = torch.device(device)
        self.checkpoint_path = checkpoint_path
        self.config_path = config_path  # Optional explicit config path
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.model = None
        self.normalize_input = False
        self.running_mean_std = None

        self._load_checkpoint()

    def _load_checkpoint(self):
        """Load checkpoint and build network using Isaac Gym infrastructure."""
        print(f"[PolicyLoader] Loading checkpoint from: {self.checkpoint_path}")

        # Load checkpoint using rl_games utilities
        checkpoint = torch_ext.load_checkpoint(self.checkpoint_path)

        # Extract model state
        model_state = checkpoint['model']

        # Load network configuration from training config file
        # (checkpoint doesn't contain config in older format)
        network_config = self._load_training_config()
        self.normalize_input = True  # Default for InterMimic

        # Use shapes provided from environment
        obs_shape = self.obs_shape
        action_shape = self.action_shape

        print(f"[PolicyLoader] Network config loaded - obs: {obs_shape}, actions: {action_shape}")

        # Build network using InterMimicBuilder
        builder = InterMimicBuilder()
        builder.load(network_config)

        net_config = {
            'actions_num': action_shape,
            'input_shape': (obs_shape,),  # Must be a tuple, not an int
            'num_seqs': 1  # Not used for inference
        }

        self.model = builder.build('intermimic', **net_config)

        # Load model weights
        # Strip 'a2c_network.' prefix if present
        cleaned_state = {}
        for key, value in model_state.items():
            if key.startswith('a2c_network.'):
                cleaned_key = key.replace('a2c_network.', '')
            else:
                cleaned_key = key
            cleaned_state[cleaned_key] = value

        self.model.load_state_dict(cleaned_state)
        self.model.to(self.device, dtype=torch.float32)
        self.model.eval()

        # Load running mean/std for observation normalization
        if self.normalize_input and 'running_mean_std' in checkpoint:
            self.running_mean_std = RunningMeanStd((obs_shape,)).to(self.device)
            self.running_mean_std.load_state_dict(checkpoint['running_mean_std'])
            self.running_mean_std.eval()

        print(f"[PolicyLoader] Policy network loaded successfully!")

    def _load_training_config(self) -> dict:
        """Load network config from original training YAML or return defaults."""
        # Try to locate a training YAML near the checkpoint (Omomo defaults).
        # If not found, fall back to hard-coded defaults from omomo.yaml.
        default_network_config = {
            "name": "intermimic",
            "separate": True,
            "space": {
                "continuous": {
                    "mu_activation": None,
                    "sigma_activation": None,
                    "mu_init": {"name": "default"},
                    "sigma_init": {"name": "const_initializer", "val": -2.9},
                    "fixed_sigma": True,
                    "learn_sigma": False,
                }
            },
            "mlp": {
                "units": [1024, 1024, 512],
                "activation": "relu",
                "d2rl": False,
                "initializer": {"name": "default"},
                "regularizer": {"name": "None"},
            },
        }

        # Attempt to find a YAML config alongside the checkpoint
        candidates = []
        # 0) explicit config path if provided
        if self.config_path is not None:
            candidates.append(Path(self.config_path))
            print(f"[PolicyLoader] Using explicit config path: {self.config_path}")
        # 1) repo-relative isaacgym training config
        repo_root = Path(__file__).resolve().parents[3]
        candidates.append(repo_root / "isaacgym" / "src" / "intermimic" / "data" / "cfg" / "train" / "rlg" / "omomo.yaml")
        # 2) alongside checkpoint (if stored near training outputs)
        ck_dir = Path(self.checkpoint_path).resolve().parent
        candidates.append(ck_dir / "omomo.yaml")

        for candidate_yaml in candidates:
            if candidate_yaml.exists():
                try:
                    print(f"[PolicyLoader] Loading network config from: {candidate_yaml}")
                    with open(candidate_yaml, "r") as f:
                        cfg = yaml.safe_load(f) or {}
                    net_cfg = cfg.get("params", {}).get("network", {})
                    # Merge defaults with loaded config (defaults fill missing keys)
                    merged = {**default_network_config, **net_cfg}
                    merged["mlp"] = {**default_network_config["mlp"], **net_cfg.get("mlp", {})}
                    merged["space"] = {**default_network_config["space"], **net_cfg.get("space", {})}
                    print(f"[PolicyLoader] Successfully loaded config from: {candidate_yaml}")
                    return merged
                except Exception as err:
                    print(f"[PolicyLoader] Failed to load training YAML ({candidate_yaml}): {err}")

        # Fallback to defaults
        print(f"[PolicyLoader] No config file found, using default network config")
        return default_network_config

    @torch.no_grad()
    def get_action(self, obs: torch.Tensor, deterministic: bool = True) -> torch.Tensor:
        """Get action from policy for given observation.

        Args:
            obs: Observation tensor (batch_size, obs_dim)
            deterministic: If True, return mean action. If False, sample from distribution.

        Returns:
            Action tensor (batch_size, action_dim)
        """
        # Convert to float32 if needed
        if obs.dtype != torch.float32:
            obs = obs.float()

        # Normalize observation if needed
        if self.normalize_input and self.running_mean_std is not None:
            obs = self.running_mean_std(obs)

        # Run actor forward pass only (skip critic for inference)
        mu, sigma = self.model.eval_actor(obs)

        if deterministic:
            action = mu
        else:
            action = mu + sigma * torch.randn_like(sigma)

        return action


def load_policy(checkpoint_path: str, obs_shape: int, action_shape: int, device: str = "cuda:0", config_path: str = None) -> IsaacGymPolicyWrapper:
    """Load policy from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        obs_shape: Observation space dimension from environment
        action_shape: Action space dimension from environment
        device: Device to run inference on
        config_path: Optional path to training YAML config (e.g., omomo.yaml)

    Returns:
        Loaded policy wrapper
    """
    return IsaacGymPolicyWrapper(checkpoint_path, obs_shape, action_shape, device, config_path)
