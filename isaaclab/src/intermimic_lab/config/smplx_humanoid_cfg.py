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
        activate_contact_sensors=True,  # Enable contact reporting for ContactSensor
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

    actuators = {
    "legs": ImplicitActuatorCfg(
        joint_names_expr=["L_Hip_.*", "R_Hip_.*", "L_Knee_.*", "R_Knee_.*", "L_Ankle_.*", "R_Ankle_.*", "L_Toe_.*", "R_Toe_.*"],
        stiffness=800.0, damping=80.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "torso": ImplicitActuatorCfg(
        joint_names_expr=["Torso_.*", "Spine_.*", "Chest_.*"],
        stiffness=1000.0, damping=100.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "arms": ImplicitActuatorCfg(
        joint_names_expr=["L_Thorax_.*","R_Thorax_.*","L_Shoulder_.*","R_Shoulder_.*","L_Elbow_.*","R_Elbow_.*","L_Wrist_.*","R_Wrist_.*"],
        stiffness=500.0, damping=50.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "fingers": ImplicitActuatorCfg(
        joint_names_expr=[".*Index.*",".*Middle.*",".*Ring.*",".*Pinky.*",".*Thumb.*"],
        stiffness=100.0, damping=10.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    "head": ImplicitActuatorCfg(
        joint_names_expr=["Neck_.*","Head_.*"],
        stiffness=500.0, damping=50.0, effort_limit_sim=3000.0, velocity_limit_sim=50.0,
    ),
    }



    # # Actuator configuration - use PD gains from MJCF (do not override)
    # actuators: dict = {}

    # Rigid body properties
    rigid_props: sim_utils.RigidBodyPropertiesCfg = sim_utils.RigidBodyPropertiesCfg(
        disable_gravity=False,
        max_depenetration_velocity=100.0,
        enable_gyroscopic_forces=True,
        angular_damping=0.01,  # From original config
        max_linear_velocity=50.0,        # clamp crazy velocities
        max_angular_velocity=50.0,       # 100 is usually unnecessary
    )

    # Collision properties
    collision_props: sim_utils.CollisionPropertiesCfg = sim_utils.CollisionPropertiesCfg(
        contact_offset=0.02,  # From PhysX config
        rest_offset=0.0,
    )
    


# Default instance for easy import
SMPLX_HUMANOID_CFG = SMPLXHumanoidCfg()
