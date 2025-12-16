"""SMPL-X Humanoid Asset Configuration for IsaacLab."""

from isaaclab.utils import configclass
from isaaclab.assets import ArticulationCfg
from isaaclab.actuators import ImplicitActuatorCfg
import isaaclab.sim as sim_utils
from ..path_utils import resolve_data_path

SMPLX_ASSET_PATH = resolve_data_path("assets", "smplx", "omomo_isaaclab.xml")


@configclass
class SMPLXHumanoidCfg(ArticulationCfg):
    """Configuration for SMPL-X humanoid robot.

    SMPL-X (SMPL eXpressive) is a parametric body model with:
    - 51 body joints with 3 DOFs each = 153 total DOFs
    - Full body including hands with articulated fingers
    - Used for whole-body human-object interaction

    Original asset: intermimic/data/assets/smplx/omomo.xml
    """

    @configclass
    class MetaInfoCfg:
        """Meta information about the SMPL-X asset."""
        asset_path: str = str(SMPLX_ASSET_PATH)
        usd_path: str = ""  # Will be generated from XML if needed

    meta_info: MetaInfoCfg = MetaInfoCfg()

    # Spawn configuration - Load from MuJoCo XML file
    # Using omomo_isaaclab.xml (no embedded floor/light - IsaacLab provides these)
    spawn: sim_utils.MjcfFileCfg = sim_utils.MjcfFileCfg(
        asset_path=str(SMPLX_ASSET_PATH),
        make_instanceable=True,
        fix_base=False,
    )

    # Articulation root (MJCF import creates /worldBody)
    articulation_root_prim_path: str = "/Pelvis/Pelvis"

    # Initial state
    init_state: ArticulationCfg.InitialStateCfg = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.0),  # Start 1m above ground
        rot=(1.0, 0.0, 0.0, 0.0),  # Identity quaternion (w, x, y, z)
        joint_pos={
            ".*": 0.0,  # All joints start at zero
        },
        joint_vel={
            ".*": 0.0,  # All velocities start at zero
        },
    )

    # Articulation properties
    articulation_props: sim_utils.ArticulationRootPropertiesCfg = sim_utils.ArticulationRootPropertiesCfg(
        enabled_self_collisions=False,  # Disable self-collisions for performance
        solver_position_iteration_count=4,  # Position solver iterations
        solver_velocity_iteration_count=1,  # Velocity solver iterations
    )

    # Actuator configuration - PD control
    actuators: dict = {
        "body": ImplicitActuatorCfg(
            joint_names_expr=[".*"],  # All joints
            stiffness={
                ".*": 500.0,  # Default stiffness for all joints
            },
            damping={
                ".*": 50.0,  # Default damping for all joints
            },
            effort_limit_sim={
                ".*": 500.0,  # Max torque per joint
            },
            velocity_limit_sim={
                ".*": 50.0,  # Conservative max velocity (rad/s)
            },
        ),
    }

    # Rigid body properties
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=100.0,
        enable_gyroscopic_forces=True,
        angular_damping=0.01,  # From original config
        max_angular_velocity=100.0,  # From original config
    )

    # Collision properties
    collision_props: sim_utils.CollisionPropertiesCfg = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.02,  # From PhysX config
        rest_offset=0.0,
    )


# Default instance for easy import
SMPLX_HUMANOID_CFG = SMPLXHumanoidCfg()
