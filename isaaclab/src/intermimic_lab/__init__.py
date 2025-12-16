"""InterMimic for IsaacLab - Whole-body human-object interaction."""

from .intermimic_env import InterMimicEnv
from .config import (
    InterMimicEnvCfg,
    INTERMIMIC_ENV_CFG,
    SMPLXHumanoidCfg,
    SMPLX_HUMANOID_CFG,
    InterMimicSceneCfg,
    INTERMIMIC_SCENE_CFG,
)

__version__ = "0.1.0"

__all__ = [
    "InterMimicEnv",
    "InterMimicEnvCfg",
    "INTERMIMIC_ENV_CFG",
    "SMPLXHumanoidCfg",
    "SMPLX_HUMANOID_CFG",
    "InterMimicSceneCfg",
    "INTERMIMIC_SCENE_CFG",
]
