"""Configuration for InterMimic environment - migrated from Isaac Gym to IsaacLab."""

from isaaclab.utils import configclass
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg
from .smplx_humanoid_cfg import SMPLX_HUMANOID_CFG


@configclass
class InterMimicEnvCfg(DirectRLEnvCfg):
    """Configuration for the InterMimic whole-body HOI environment.

    This environment trains humanoid characters (SMPL-X or Unitree G1) to perform
    physics-based human-object interactions using motion retargeting and imitation.

    Original Isaac Gym config: intermimic/data/cfg/omomo_train.yaml
    """

    # Simulation parameters
    sim: SimulationCfg = SimulationCfg(
        dt=1/120,  # Physics simulation timestep (60 Hz)
        render_interval=4,  # Render every 2 physics steps
        physx=PhysxCfg(
            solver_type=1,  # 0: PGS, 1: TGS (Temporal Gauss-Seidel)
            min_position_iteration_count=4,
            max_position_iteration_count=4,
            min_velocity_iteration_count=1,
            max_velocity_iteration_count=1,
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=34603008,  # 8*1024*1024,
            enable_ccd=True,
        ),
    )

    # Scene parameters
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=4096,
        env_spacing=2.0,
        replicate_physics=True,
    )

    # Environment parameters
    episode_length_s: float = 10.0  # Episode duration in seconds (300 steps @ 30Hz)
    decimation: int = 4  # Control frequency = physics_freq / decimation = 60/2 = 30 Hz
    num_envs: int = 4096  # Number of parallel environments
    observation_space: int = 3198  # Flat observation vector for policy
    num_observations: int = 3198  # Observation space size
    action_space: int = 153  # Flat action vector
    num_actions: int = 153  # Action space size: 51 joints × 3 DOFs
    play_dataset: bool = False  # Enable motion replay instead of control
    replay_root_height_offset: float = 0.03  # Optional Z offset during mocap replay
    replay_object_height_offset: float = 0.03  # Optional Z offset for objects during replay

    # State initialization
    state_init: str = "Hybrid"  # Options: "Default", "Start", "Random", "Hybrid"
    hybrid_init_prob: float = 0.1  # Probability of using reference state in hybrid mode

    # Motion data parameters
    motion_file: str = "InterAct/OMOMO"  # Path to motion data
    data_fps: int = 30  # Motion data frame rate
    data_sub: list = ["sub2"]  # Data subset to use
    rollout_length: int = 300  # Maximum rollout length

    # Robot configuration
    robot_type: str = "smplx/omomo.xml"  # SMPL-X humanoid model
    robot_cfg: ArticulationCfg = SMPLX_HUMANOID_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Object parameters
    object_density: float = 200.0  # Density for dynamic objects (kg/m^3)
    ball_size: float = 1.0  # Size scaling for ball objects

    # Object physics material (matching Isaac Gym intermimic.py:333-340)
    @configclass
    class ObjectPhysicsCfg:
        static_friction: float = 0.6
        dynamic_friction: float = 0.6
        restitution: float = 0.05  # Bounciness
        # rest_offset varies by object type (0.015 for plasticbox/trashcan, 0.002 for others)

    object_physics: ObjectPhysicsCfg = ObjectPhysicsCfg()

    # Termination conditions
    termination_height: float = 0.15  # Terminate if root below this height (meters)
    enable_early_termination: bool = True

    # Control parameters
    pd_control: bool = True  # Use PD control for joints
    power_scale: float = 1.0  # Action scaling factor

    # Key bodies for tracking and contact
    key_bodies: list = None  # Will be set in __post_init__
    contact_bodies: list = None  # Will be set in __post_init__

    # Observation flags
    local_root_obs: bool = False  # Use local or global root observations

    # Visualization
    enable_reference_markers: bool = True  # Show reference motion markers (joints and object)

    # Ground plane friction
    @configclass
    class PlaneCfg:
        static_friction: float = 0.9
        dynamic_friction: float = 0.9
        restitution: float = 0.1  # Low bounce

    plane: PlaneCfg = PlaneCfg()

    # Reward configuration
    @configclass
    class RewardsCfg:
        """Reward weights for different components."""

        # Humanoid pose tracking
        pose_weight: float = 30.0  # Root and joint position tracking
        rotation_weight: float = 2.5  # Root and joint rotation tracking
        pose_vel_weight: float = 0.0  # Velocity matching (disabled)
        rotation_vel_weight: float = 0.0  # Angular velocity matching (disabled)

        # Object tracking
        object_pose_weight: float = 5.0  # Object position tracking
        object_rotation_weight: float = 0.1  # Object orientation tracking
        object_pose_vel_weight: float = 0.1  # Object velocity tracking
        object_rotation_vel_weight: float = 0.0  # Object angular velocity (disabled)

        # Interaction rewards
        interaction_geometry_weight: float = 5.0  # HOI geometry matching
        contact_hand_weight: float = 5.0  # Hand contact rewards
        contact_other_weight: float = 5.0  # Other body contact rewards
        contact_all_weight: float = 3.0  # Overall contact matching

        # Energy/action penalties
        energy_penalty_1: float = 0.00002  # Action magnitude penalty
        energy_penalty_2: float = 0.00002  # Action change penalty
        energy_penalty_3: float = 0.000000001  # Additional regularization

    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """Post-initialization to set up derived parameters."""
        # Define key bodies for observation and rewards
        self.key_bodies = [
            # Lower body
            "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
            "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
            # Upper body
            "Torso", "Spine", "Chest", "Neck", "Head",
            # Arms
            "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
            "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist"
        ]

        # Contact bodies include key bodies plus hand fingers
        self.contact_bodies = self.key_bodies + [
            # Left hand fingers
            "L_Index3", "L_Middle3", "L_Pinky3", "L_Ring3", "L_Thumb3",
            # Right hand fingers
            "R_Index3", "R_Middle3", "R_Pinky3", "R_Ring3", "R_Thumb3"
        ]

        # Verify episode length matches expected steps
        expected_steps = self.episode_length_s / (self.sim.dt * self.decimation)
        assert abs(expected_steps - 300) < 1e-6, \
            f"Episode length mismatch: {expected_steps} steps != 300 steps"

        # Verify control frequency
        control_freq = 1.0 / (self.sim.dt * self.decimation)
        assert abs(control_freq - 30.0) < 1e-6, \
            f"Control frequency mismatch: {control_freq} Hz != 30 Hz"


# Create default configuration instance
INTERMIMIC_ENV_CFG = InterMimicEnvCfg()
