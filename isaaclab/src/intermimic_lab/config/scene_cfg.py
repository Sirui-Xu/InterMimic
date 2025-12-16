"""Scene configuration for InterMimic environment."""

from typing import Optional

from isaaclab.utils import configclass
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.sim.spawners import GroundPlaneCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg

from .smplx_humanoid_cfg import SMPLX_HUMANOID_CFG


@configclass
class InterMimicSceneCfg(InteractiveSceneCfg):
    """Configuration for the InterMimic scene with humanoid and objects.

    The scene consists of:
    - Ground plane with configurable friction
    - SMPL-X humanoid robot
    - Dynamic objects for interaction (loaded from motion data)
    """

    # Ground plane is spawned explicitly in _setup_scene() instead
    # ground: AssetBaseCfg | None = sim_utils.GroundPlaneCfg()

    # SMPL-X Humanoid robot will be spawned programmatically in InterMimicEnv
    # to allow custom template creation and cloning.
    robot: Optional[SMPLX_HUMANOID_CFG.__class__] = None

    # Note: Dynamic objects will be spawned programmatically based on motion data
    # They are not part of the static scene configuration


# Default instance
INTERMIMIC_SCENE_CFG = InterMimicSceneCfg(num_envs=4096, env_spacing=2.0)
