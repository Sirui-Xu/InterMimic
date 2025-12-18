"""InterMimic Environment for IsaacLab - Direct RL Environment."""

import os
from pathlib import Path
from typing import Dict
import xml.etree.ElementTree as ET

import torch
import torch.nn.functional as F

from isaaclab.envs import DirectRLEnv
from isaaclab.assets import Articulation, RigidObject, RigidObjectCfg
import isaaclab.sim as sim_utils
from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg, PreviewSurfaceCfg, spawn_rigid_body_material
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.sim.converters import MeshConverter, MeshConverterCfg
from isaaclab.sim.schemas import schemas_cfg
from isaaclab.utils.math import quat_from_euler_xyz
from isaaclab.sim.utils.prims import bind_visual_material, bind_physics_material, is_prim_path_valid
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG

from .config import InterMimicEnvCfg, InterMimicSceneCfg
from .path_utils import resolve_data_path
from . import torch_utils
from . import observation_utils


class InterMimicEnv(DirectRLEnv):
    """InterMimic environment for whole-body human-object interaction.

    This environment trains SMPL-X humanoid to perform physics-based interactions
    with dynamic objects using motion imitation and retargeting.

    Key features:
    - SMPL-X humanoid with 51 joints (153 DOFs)
    - Dynamic object interaction from motion capture data
    - Hybrid state initialization (Default/Start/Random/Hybrid)
    - Physics-based contact rewards
    """

    cfg: InterMimicEnvCfg

    def __init__(self, cfg: InterMimicEnvCfg, render_mode: str | None = None, **kwargs):
        """Initialize InterMimic environment.

        Args:
            cfg: Configuration for the environment.
            render_mode: Render mode for the environment.
            **kwargs: Additional keyword arguments.
        """
        # Store configuration
        self.cfg = cfg

        # Load motion data
        self._load_motion_data()

        # Load object assets
        self._setup_object_assets()

        # Dataset playback buffers
        self._motion_dataset = None
        self._motion_lengths = None
        self._motion_max_length = 0
        self._motion_object_ids = None
        self._playback_assignments = None
        self._playback_frame = 0
        self._motion_dataset_aligned = False
        self._dof_reorder_indices: torch.Tensor | None = None
        self._body_reorder_indices: torch.Tensor | None = None
        self._body_reorder_inv_indices: torch.Tensor | None = None
        self._env_motion_ids_tensor: torch.Tensor | None = None
        self._env_object_names: list[str] = []
        self._env_rigid_objects: list[RigidObject | None] = []
        self._env_object_id_tensor: torch.Tensor | None = None
        self._object_mesh_usd_cache: dict[tuple[str, tuple[float, float, float], float | None], Path] = {}
        self._object_contact_sensor = None
        # Visualization markers for reference motion (may be created in _setup_scene)
        self._reference_joint_markers = None
        self._reference_object_markers = None
        # Debug storage for observation alignment dumps

        # Build HOI data and motion dataset from motion files
        # This creates _motion_dataset, hoi_data, and sets metadata
        self._build_hoi_data_from_motion()

        # Initialize parent
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        # Move object points to device after sim is initialized
        self.object_points = self.object_points.to(self.device)

        # Build DOF reorder indices for action/obs remapping
        self._build_dof_reorder_indices()

        # Move motion dataset to device and align to articulation order
        self._move_dataset_to_device(self.device)
        self._align_motion_dataset_to_robot()

        # Build additional tensors
        self._build_target_tensors()

        # Initialize observation and reward buffers
        self._init_buffers()

    def _setup_scene(self):
        """Setup the scene with humanoid and objects.

        This replaces the create_sim() method from Isaac Gym.
        """
        # IsaacLab scene already contains the default ground plane; no custom ground spawned here.
        # Import spawners
        from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

        # Step 1: Spawn ground plane
        plane_material = RigidBodyMaterialCfg(
            static_friction=self.cfg.plane.static_friction,
            dynamic_friction=self.cfg.plane.dynamic_friction,
            restitution=self.cfg.plane.restitution,
        )
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(physics_material=plane_material),
        )
        from isaaclab.sim.schemas import schemas
        ground_collision_path = "/World/ground/GroundPlane/CollisionPlane"

        # Set collision properties for better thin object detection
        schemas.define_collision_properties(
            prim_path=ground_collision_path,
            cfg=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.02,   # Increased for better collision detection with thin objects
                rest_offset=0.0,       # Ground stays at 0
            ),
        )

        # Step 2: Create SMPL-X humanoid articulation
        self._robot = Articulation(self.cfg.robot_cfg)

        # Step 3: Clone environments (creates num_envs copies)
        # Use copy_from_source=True so each environment can hold different assets (objects).
        self.scene.clone_environments(copy_from_source=True)

        # Step 4: Add dynamic objects per environment (cyclic over object types)
        self._spawn_objects_for_environments()

        # Step 5: Remove any decorative ground planes embedded in the robot asset
        self._remove_embedded_ground_geometry()

        # Step 6: Filter collisions for CPU simulation
        if self.sim.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        # Step 7: Register robot with the scene
        self.scene.articulations["robot"] = self._robot

        # Step 8: Create contact sensor for the robot (after robot is spawned)
        # Note: MJCF imports create structure like /Robot/Pelvis/Pelvis/body_parts
        from isaaclab.sensors import ContactSensor, ContactSensorCfg
        contact_sensor_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/Pelvis/Pelvis",  # Robot articulation root
            update_period=0.0,  # Update every step
            history_length=0,  # No history needed
            track_air_time=False,
            debug_vis=False,
        )
        self._contact_sensor = ContactSensor(contact_sensor_cfg)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        # Contact sensor for spawned objects
        object_contact_cfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/object",
            update_period=0.0,
            history_length=0,
            track_air_time=False,
            debug_vis=False,
        )
        self._object_contact_sensor = ContactSensor(object_contact_cfg)
        self.scene.sensors["object_contact_sensor"] = self._object_contact_sensor

        # Step 9: Setup reference motion visualization markers
        self._setup_reference_markers()

        # Step 10: Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # Setup debug visualization if enabled
        if getattr(self.cfg, "debug_viz", False):
            self._setup_debug_viz()

    def _spawn_objects_for_environments(self):
        """Spawn interaction objects so each environment has exactly one object."""
        asset_root = resolve_data_path("assets", "objects")
        mesh_root = resolve_data_path("assets", "objects", "objects", must_exist=False)

        color_palette = [
            (0.85, 0.33, 0.1),
            (0.2, 0.6, 0.85),
            (0.55, 0.7, 0.2),
            (0.7, 0.3, 0.8),
        ]

        self._env_rigid_objects = []
        self._env_object_names = []
        self._env_motion_ids_tensor = None
        if not self.object_names:
            self._env_object_id_tensor = None
            return

        num_motions = len(self.motion_object_names)
        object_to_motion_indices: dict[str, list[int]] = {}
        for motion_idx, obj in enumerate(self.motion_object_names):
            object_to_motion_indices.setdefault(obj, []).append(motion_idx)
        object_ids = {name: idx for idx, name in enumerate(self.object_names)}
        env_object_ids = torch.full((self.num_envs,), -1, dtype=torch.long, device=self.device)
        env_motion_ids: list[int] = []
        material_registry: dict[str, str] = {}

        motion_counters = {obj: 0 for obj in self.object_names}
        rigid_props_cfg = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.01,
            angular_damping=0.01,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=100.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=1,
        )
        collision_props_cfg = sim_utils.CollisionPropertiesCfg(
            contact_offset=0.02,
            rest_offset=0.0,
        )
        for env_id in range(self.num_envs):
            if num_motions > 0:
                preferred_obj = self.object_names[env_id % len(self.object_names)]
                indices = object_to_motion_indices.get(preferred_obj, [])
                if indices:
                    idx = motion_counters[preferred_obj] % len(indices)
                    motion_idx = indices[idx]
                    motion_counters[preferred_obj] += 1
                else:
                    motion_idx = env_id % num_motions
                env_motion_ids.append(motion_idx)
                obj_name = self.motion_object_names[motion_idx]
            else:
                env_motion_ids.append(-1)
                obj_name = self.object_names[env_id % len(self.object_names)]

            if obj_name not in object_ids:
                print(f"[InterMimic] Unknown object '{obj_name}' for env {env_id}, skipping spawn.")
                self._env_object_names.append(obj_name)
                self._env_rigid_objects.append(None)
                continue

            self._env_object_names.append(obj_name)
            env_object_ids[env_id] = object_ids[obj_name]

            color = color_palette[object_ids[obj_name] % len(color_palette)]
            if obj_name not in material_registry:
                material_registry[obj_name] = f"/World/materials/{obj_name}"
            material_path = material_registry[obj_name]
            create_material = not is_prim_path_valid(material_path)

            mass_props = (
                sim_utils.MassPropertiesCfg(density=self.cfg.object_density)
                if self.cfg.object_density is not None
                else None
            )
            scale_tuple = (
                (self.cfg.ball_size, self.cfg.ball_size, self.cfg.ball_size)
                if self.cfg.ball_size not in (None, 1.0)
                else None
            )

            if mesh_root is None:
                print(f"[InterMimic] Mesh directory missing. Cannot spawn '{obj_name}'.")
                self._env_rigid_objects.append(None)
                continue
            mesh_path = mesh_root / obj_name / f"{obj_name}.obj"
            if not mesh_path.exists():
                print(f"[InterMimic] Mesh file not found: {mesh_path}, skipping object")
                self._env_rigid_objects.append(None)
                continue

            scale_for_cache = scale_tuple if scale_tuple is not None else (1.0, 1.0, 1.0)
            density_key = float(self.cfg.object_density) if self.cfg.object_density is not None else None
            cache_key = (obj_name, scale_for_cache, density_key)
            usd_path = self._object_mesh_usd_cache.get(cache_key)
            if usd_path is None:
                usd_path = self._convert_object_mesh_to_usd(
                    obj_name=obj_name,
                    mesh_path=mesh_path,
                    scale=scale_for_cache,
                    rigid_props=rigid_props_cfg,
                    collision_props=collision_props_cfg,
                    mass_props=mass_props,
                )
                if usd_path is None:
                    self._env_rigid_objects.append(None)
                    continue
                self._object_mesh_usd_cache[cache_key] = usd_path

            obj_cfg = RigidObjectCfg(
                prim_path=f"/World/envs/env_{env_id}/object",
                spawn=UsdFileCfg(
                    usd_path=str(usd_path),
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=color,
                        roughness=0.35,
                    )
                    if create_material
                    else None,
                    activate_contact_sensors=True,
                ),
                init_state=RigidObjectCfg.InitialStateCfg(
                    pos=(0.0, 0.0, 1.0),
                    rot=(1.0, 0.0, 0.0, 0.0),
                ),
            )
            obj = RigidObject(cfg=obj_cfg)

            # Bind visual material if needed
            if not create_material and is_prim_path_valid(material_path):
                bind_visual_material(obj_cfg.prim_path, material_path)

            # Create and bind physics material (matching Isaac Gym intermimic.py:333-340)
            physics_material_path = f"/World/PhysicsMaterials/ObjectMaterial_{env_id}"
            physics_material_cfg = RigidBodyMaterialCfg(
                static_friction=self.cfg.object_physics.static_friction,
                dynamic_friction=self.cfg.object_physics.dynamic_friction,
                restitution=self.cfg.object_physics.restitution,
                friction_combine_mode="average",
                restitution_combine_mode="average",
            )
            spawn_rigid_body_material(physics_material_path, physics_material_cfg)
            bind_physics_material(obj_cfg.prim_path, physics_material_path)

            self._env_rigid_objects.append(obj)
            self.scene.rigid_objects[f"env_{env_id}_object"] = obj

        self._env_object_id_tensor = env_object_ids
        if env_motion_ids:
            self._env_motion_ids_tensor = torch.tensor(env_motion_ids, dtype=torch.long, device=self.device)

    def _convert_object_mesh_to_usd(
        self,
        obj_name: str,
        mesh_path: Path,
        scale: tuple[float, float, float],
        rigid_props,
        collision_props,
        mass_props,
    ) -> Path | None:
        """Convert an OBJ mesh to USD using MeshConverter and return the USD path."""
        cache_dir = (Path(__file__).resolve().parent / "assets" / "usd" / obj_name)
        cache_dir.mkdir(parents=True, exist_ok=True)
        usd_file_name = f"{obj_name}_{'_'.join(f'{s:.3f}' for s in scale)}"
        mesh_cfg = MeshConverterCfg(
            asset_path=str(mesh_path),
            usd_dir=str(cache_dir),
            usd_file_name=usd_file_name,
            make_instanceable=False,
            scale=scale,
            rigid_props=rigid_props,
            collision_props=collision_props,
            mass_props=mass_props,
            mesh_collision_props=schemas_cfg.ConvexDecompositionPropertiesCfg(
                hull_vertex_limit=64,
                max_convex_hulls=64,
                voxel_resolution=300000,
            ),
        )
        try:
            converter = MeshConverter(mesh_cfg)
        except Exception as err:
            print(f"[InterMimic] Failed to convert mesh '{mesh_path}' to USD: {err}")
            return None
        return Path(converter.usd_path)

    def _load_motion_data(self):
        """Load motion capture data for imitation.

        Loads HOI motion sequences from the specified data directory.
        """
        motion_dir = os.path.join(
            os.environ.get('INTERMIMIC_PATH', os.getcwd()),
            self.cfg.motion_file
        )

        # Find all motion files for the specified data subset
        motion_files = []
        if os.path.exists(motion_dir):
            all_files = sorted(os.listdir(motion_dir))
            for filename in all_files:
                # Filter by data subset (e.g., 'sub2')
                if any(sub in filename for sub in self.cfg.data_sub):
                    motion_files.append(os.path.join(motion_dir, filename))

        print(f"[InterMimic] Found {len(motion_files)} motion files")

        # Extract object names from motion files
        self.object_names = []
        self.motion_object_names = []
        for filepath in motion_files:
            # Format: seq###_objectname_*.pt
            filename = os.path.basename(filepath)
            parts = filename.split('_')
            if len(parts) >= 2:
                obj_name = parts[-2]
                self.motion_object_names.append(obj_name)
                if obj_name not in self.object_names:
                    self.object_names.append(obj_name)
            else:
                self.motion_object_names.append("unknown")

        print(f"[InterMimic] Unique objects: {self.object_names}")

        # Store motion file paths
        self.motion_files = motion_files
        self.num_motions = len(motion_files)

        # Motion data will be loaded lazily or in batches
        # For now, just store the file paths
        self.motion_data = {}

    def _setup_object_assets(self):
        """Setup object asset properties.

        Loads object meshes and computes surface points for contact rewards.
        """
        import trimesh
        from isaaclab.utils.math import convert_quat

        self.object_points = []
        mesh_root = resolve_data_path("assets", "objects", "objects", must_exist=False)

        for obj_name in self.object_names:
            obj_file = mesh_root / obj_name / f"{obj_name}.obj"

            if obj_file.exists():
                # Load mesh and sample surface points
                mesh_obj = trimesh.load(str(obj_file), force='mesh')
                obj_verts = mesh_obj.vertices
                center = obj_verts.mean(axis=0)

                # Sample 1024 surface points
                object_points, _ = trimesh.sample.sample_surface_even(
                    mesh_obj, count=1024, seed=2024
                )
                object_points = torch.tensor(object_points - center, dtype=torch.float32)

                # Pad if needed
                while object_points.shape[0] < 1024:
                    object_points = torch.cat([
                        object_points,
                        object_points[:1024 - object_points.shape[0]]
                    ], dim=0)

                self.object_points.append(object_points)
            else:
                print(f"[Warning] Object file not found: {obj_file}")
                # Create dummy points
                self.object_points.append(torch.zeros((1024, 3)))

        if self.object_points:
            self.object_points = torch.stack(self.object_points, dim=0)
            print(f"[InterMimic] Loaded object points: {self.object_points.shape}")
        else:
            # Will be moved to device after parent __init__
            self.object_points = torch.zeros((0, 1024, 3), dtype=torch.float32)

    def _build_dof_reorder_indices(self):
        """Build a mapping from Isaac Gym joint order (dataset) to Isaac Lab articulation order."""
        self._dof_reorder_indices = None
        robot = self.scene.articulations.get("robot", None)
        if robot is None:
            print("[InterMimic] Robot articulation not available for DOF remapping.")
            return

        # Joint names in the articulation (Isaac Lab order)
        actual_names_raw = getattr(robot.data, "joint_names", None)
        if not actual_names_raw:
            print("[InterMimic] Failed to read articulation joint names for DOF remapping.")
            return
        actual_names = [name.split("/")[-1] for name in actual_names_raw]

        # Joint names as stored in the Isaac Gym dataset (MJCF declaration order)
        try:
            expected_names = self._load_mjcf_joint_order(self.cfg.robot_cfg.spawn.asset_path)
        except FileNotFoundError:
            print(f"[InterMimic] MJCF file not found for DOF remapping: {self.cfg.robot_cfg.spawn.asset_path}")
            return
        except Exception as err:
            print(f"[InterMimic] Failed to parse MJCF for DOF remapping: {err}")
            return

        if not expected_names:
            print("[InterMimic] No joint names found in MJCF for DOF remapping.")
            return

        dataset_index = {name: idx for idx, name in enumerate(expected_names)}
        reorder: list[int] = []
        missing_in_dataset: list[str] = []
        for name in actual_names:
            base = name.split(":")[-1]
            candidates = [base, base.replace("_joint", "")]
            matched_idx = None
            for candidate in candidates:
                if candidate in dataset_index:
                    matched_idx = dataset_index[candidate]
                    break
            if matched_idx is None:
                missing_in_dataset.append(name)
                continue
            reorder.append(matched_idx)

        if missing_in_dataset:
            print(f"[InterMimic] Missing joints in dataset for DOF remapping: {missing_in_dataset}")
            return

        if len(reorder) != len(actual_names):
            print(
                f"[InterMimic] DOF remapping size mismatch (dataset {len(expected_names)} vs articulation {len(actual_names)})."
            )
            return

        self._dof_reorder_indices = torch.tensor(reorder, device=self.device, dtype=torch.long) # gym2lab
        self._dof_reorder_inv_indices = torch.argsort(self._dof_reorder_indices) # lab2gym

    @staticmethod
    def _load_mjcf_joint_order(asset_path: str) -> list[str]:
        """Extract joint names from MJCF in declaration order (excluding freejoint/default)."""
        tree = ET.parse(asset_path)
        root = tree.getroot()
        joint_names: list[str] = []
        for joint in root.iter("joint"):
            name = joint.get("name")
            if not name:
                continue
            jtype = joint.get("type", "hinge").lower()
            if joint.tag == "freejoint" or jtype in {"free", "freejoint"}:
                continue
            joint_names.append(name)
        return joint_names

    @staticmethod
    def _load_mjcf_body_order(asset_path: str) -> list[str]:
        """Extract body names from MJCF in declaration (depth-first) order."""
        tree = ET.parse(asset_path)
        root = tree.getroot()
        worldbody = root.find("worldbody")
        if worldbody is None:
            return []

        body_names: list[str] = []

        def traverse(body_elem: ET.Element):
            name = body_elem.get("name")
            if name:
                body_names.append(name)
            for child in body_elem.findall("body"):
                traverse(child)

        for body in worldbody.findall("body"):
            traverse(body)

        return body_names

    def _align_motion_dataset_to_robot(self):
        """Reorder cached motion DOF tensors and convert quats to wxyz to match articulation."""
        if self._motion_dataset is None or self._dof_reorder_indices is None:
            return

        reorder = self._dof_reorder_indices
        for key in ("dof_pos", "dof_vel"):
            if key in self._motion_dataset:
                tensor = self._motion_dataset[key]
                if tensor.shape[-1] < reorder.shape[0]:
                    print(
                        f"[InterMimic] Skipping DOF reorder for {key}: dataset has {tensor.shape[-1]} DOFs,"
                        f" articulation expects {reorder.shape[0]}."
                    )
                    self._motion_dataset_aligned = False
                    return
                # tensor shape: (num_motions, max_len, num_dofs)
                self._motion_dataset[key] = tensor[:, :, reorder]

        # Convert quaternion ordering from Gym (xyzw) to IsaacLab (wxyz) for root and object
        if "root_rot" in self._motion_dataset:
            self._motion_dataset["root_rot"] = self._convert_gym_quaternion(self._motion_dataset["root_rot"])
        if "obj_rot" in self._motion_dataset:
            self._motion_dataset["obj_rot"] = self._convert_gym_quaternion(self._motion_dataset["obj_rot"])

        self._motion_dataset_aligned = True
        print("[InterMimic] Reordered motion dataset DOFs to match articulation.")

    def _remove_embedded_ground_geometry(self):
        """No-op: omomo_isaaclab.xml has no embedded floor/light geometry.

        The IsaacLab-specific MJCF file (omomo_isaaclab.xml) has the embedded
        floor and light removed, so no runtime cleanup is needed.
        """
        pass

    def _move_dataset_to_device(self, device: torch.device | str):
        """Move cached dataset tensors to the desired device."""
        if self._motion_dataset is None:
            return
        device = torch.device(device)
        for key, tensor in self._motion_dataset.items():
            self._motion_dataset[key] = tensor.to(device)
        if self._motion_lengths is not None:
            self._motion_lengths = self._motion_lengths.to(device)
        if self._motion_object_ids is not None:
            self._motion_object_ids = self._motion_object_ids.to(device)
        if self._playback_assignments is not None:
            self._playback_assignments = self._playback_assignments.to(device)
        if self._env_object_id_tensor is not None:
            self._env_object_id_tensor = self._env_object_id_tensor.to(device)
        if self._env_motion_ids_tensor is not None:
            self._env_motion_ids_tensor = self._env_motion_ids_tensor.to(device)
        if self.hoi_data is not None:
            self.hoi_data = self.hoi_data.to(device)

    def _build_hoi_data_from_motion(self):
        """Build HOI data tensor from motion files, following Isaac Gym _load_motion().

        This processes raw motion data to compute velocities and IG, then assembles
        the final hoi_data tensor that can be indexed for reference observations.
        Also sets metadata: _motion_lengths, _motion_max_length, _motion_object_ids.
        """
        if not self.motion_files:
            print("[InterMimic] No motion files loaded")
            self.hoi_data = None
            return

        print(f"[InterMimic] Building HOI data from {len(self.motion_files)} motion files...")

        hoi_datas = []
        fps_data = 30.0
        lengths = []  # Track motion lengths for metadata
        object_ids = []  # Track object IDs for metadata

        # Also collect individual components for _motion_dataset (used by markers, etc.)
        component_lists = {
            'root_pos': [], 'root_rot': [], 'dof_pos': [], 'dof_vel': [],
            'body_pos': [], 'body_rot': [], 'body_pos_vel': [], 'body_rot_vel': [],
            'obj_pos': [], 'obj_rot': [], 'obj_pos_vel': [], 'obj_rot_vel': [],
            'ig': [], 'contact_human': [], 'contact_obj': []
        }

        def quat_to_exp_map_wxyz(quat: torch.Tensor) -> torch.Tensor:
            """Compute exponential map from wxyz quaternion."""
            sin_theta = torch.sqrt(torch.clamp(1.0 - quat[..., 0] * quat[..., 0], min=0.0))
            angle = 2.0 * torch.acos(torch.clamp(quat[..., 0], -1.0, 1.0))
            axis = quat[..., 1:] / (sin_theta.unsqueeze(-1) + 1e-8)
            default_axis = torch.zeros_like(axis)
            default_axis[..., -1] = 1.0
            mask = sin_theta > 1e-5
            angle = torch.where(mask, angle, torch.zeros_like(angle))
            axis = torch.where(mask.unsqueeze(-1), axis, default_axis)
            return angle.unsqueeze(-1) * axis
        
        for idx, motion_file in enumerate(self.motion_files):
            loaded_dict = {}

            # Load raw HOI data from file (use CPU if device not yet initialized)
            device = self.device if hasattr(self, 'device') else torch.device('cpu')
            raw_hoi_data = torch.load(motion_file, map_location=device)
            loaded_dict['hoi_data'] = raw_hoi_data.detach()

            num_frames = loaded_dict['hoi_data'].shape[0]

            # Track metadata
            lengths.append(num_frames)
            motion_obj_name = self.motion_object_names[idx]
            obj_id = self.object_names.index(motion_obj_name) if motion_obj_name in self.object_names else -1
            object_ids.append(obj_id)

            # Extract components from raw HOI data (slicing as per Isaac Gym)
            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:] - loaded_dict['root_pos'][:-1,:]) * fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, 3), device=device), loaded_dict['root_pos_vel']), dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()  # keep xyzw for policy compatibility
            root_rot_wxyz = self._convert_gym_quaternion(loaded_dict['root_rot'])
            root_rot_exp = quat_to_exp_map_wxyz(root_rot_wxyz)
            loaded_dict['root_rot_vel'] = torch.zeros(num_frames, 3, device=device)
            loaded_dict['root_rot_vel'][1:] = (root_rot_exp[1:] - root_rot_exp[:-1]) * fps_data

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()
            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:] - loaded_dict['dof_pos'][:-1,:]) * fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, 153), device=device), loaded_dict['dof_vel']), dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162:162+52*3].clone()
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:] - loaded_dict['body_pos'][:-1,:]) * fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, 52*3), device=device), loaded_dict['body_pos_vel']), dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()
            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:] - loaded_dict['obj_pos'][:-1,:]) * fps_data
            loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, 3), device=device), loaded_dict['obj_pos_vel']), dim=0)

            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()  # keep xyzw
            obj_rot_wxyz = self._convert_gym_quaternion(loaded_dict['obj_rot'])
            obj_rot_exp = quat_to_exp_map_wxyz(obj_rot_wxyz)
            loaded_dict['obj_rot_vel'] = torch.zeros(num_frames, 3, device=device)
            loaded_dict['obj_rot_vel'][1:] = (obj_rot_exp[1:] - obj_rot_exp[:-1]) * fps_data

            # Compute interaction geometry (IG)
            # Get object points for this motion's object
            motion_obj_name = self.motion_object_names[idx]
            obj_idx = self.object_names.index(motion_obj_name) if motion_obj_name in self.object_names else 0
            obj_points_template = self.object_points[obj_idx:obj_idx+1]  # (1, 1024, 3)

            # Transform object points to world frame for each frame
            obj_rot_extend = obj_rot_wxyz.unsqueeze(1).repeat(1, obj_points_template.shape[1], 1).view(-1, 4)
            object_points_extend = obj_points_template.squeeze(0).unsqueeze(0).repeat(num_frames, 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(num_frames, obj_points_template.shape[1], 3) + loaded_dict['obj_pos'].unsqueeze(1)

            # Compute SDF from body positions to object points
            ref_ig = torch_utils.compute_sdf(loaded_dict['body_pos'].view(num_frames, 52, 3), obj_points).view(-1, 3)

            # Transform to heading frame
            heading_rot = torch_utils.calc_heading_quat_inv(root_rot_wxyz)
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, 52, 1).view(-1, 4)
            ref_ig = torch_utils.quat_rotate(heading_rot_extend, ref_ig).view(num_frames, -1)
            loaded_dict['ig'] = ref_ig

            # Extract contact data
            loaded_dict['contact_obj'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
            loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())
            body_rot_xyzw = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone().view(num_frames, 52, 4)
            body_rot_wxyz = self._convert_gym_quaternion(body_rot_xyzw.view(-1, 4)).view(num_frames, 52, 4)
            loaded_dict['body_rot'] = body_rot_xyzw.view(num_frames, -1)  # keep xyzw in stored data
            body_rot_exp = quat_to_exp_map_wxyz(body_rot_wxyz.view(-1, 4)).view(num_frames, 52, 3)
            body_rot_vel = torch.zeros(num_frames, 52, 3, device=device)
            body_rot_vel[1:] = (body_rot_exp[1:] - body_rot_exp[:-1]) * fps_data
            loaded_dict['body_rot_vel'] = body_rot_vel.view(num_frames, -1)

            # Reassemble HOI data
            loaded_dict['hoi_data'] = torch.cat((
                loaded_dict['root_pos'],
                loaded_dict['root_rot'],
                loaded_dict['dof_pos'],
                loaded_dict['dof_vel'],
                loaded_dict['body_pos'],
                loaded_dict['body_rot'],
                loaded_dict['body_pos_vel'],
                loaded_dict['body_rot_vel'],
                loaded_dict['obj_pos'],
                loaded_dict['obj_rot'],
                loaded_dict['obj_pos_vel'],
                loaded_dict['obj_rot_vel'],
                loaded_dict['ig'],
                loaded_dict['contact_human'],
                loaded_dict['contact_obj'],
            ), dim=-1)

            hoi_datas.append(loaded_dict['hoi_data'])

            # Also save individual components for visualization/debugging
            for key in component_lists.keys():
                if key in loaded_dict:
                    component_lists[key].append(loaded_dict[key])

        # Pad all motions to same length
        max_length = max(hoi_data.shape[0] for hoi_data in hoi_datas)
        padded_hoi_datas = []
        for hoi_data in hoi_datas:
            pad_size = (0, 0, 0, max_length - hoi_data.size(0))
            padded_data = F.pad(hoi_data, pad_size, "constant", 0)
            padded_hoi_datas.append(padded_data)

        # Stack into (num_motions, max_frames, 1211)
        self.hoi_data = torch.stack(padded_hoi_datas, dim=0)
        print(f"[InterMimic] Built HOI data tensor: {self.hoi_data.shape}")

        # Save only essential components that need special handling (DOF reordering, playback)
        # Other components can be extracted from hoi_data using extract_data_component()
        if not hasattr(self, '_motion_dataset') or self._motion_dataset is None:
            self._motion_dataset = {}

        def pad_and_stack(component_list):
            """Pad sequences to max_length and stack."""
            padded = []
            for seq in component_list:
                pad_size = (0, 0, 0, max_length - seq.shape[0])
                padded.append(F.pad(seq, pad_size, "constant", 0))
            return torch.stack(padded, dim=0)

        # Only store components needed for dataset playback and DOF alignment
        essential_keys = ['root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'obj_pos', 'obj_rot']
        for key in essential_keys:
            if key in component_lists and component_lists[key]:
                self._motion_dataset[key] = pad_and_stack(component_lists[key])

        # Set metadata for motion dataset
        device = self.device if hasattr(self, 'device') else torch.device('cpu')
        self._motion_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
        self._motion_max_length = int(max_length)
        self._motion_object_ids = torch.tensor(object_ids, dtype=torch.long, device=device)
        self._playback_assignments = None
        self._playback_frame = 0

        print(f"[InterMimic] Saved {len(self._motion_dataset)} components to motion dataset")
        print(f"[InterMimic] Motion metadata: {len(lengths)} motions (max length={max_length})")

    @property
    def dataset_max_length(self) -> int:
        """Maximum number of frames available in the loaded dataset."""
        return int(self._motion_max_length)

    def _apply_dataset_frame(self, frame_idx: int):
        """Apply dataset states to humanoid and objects for the given frame."""
        if self._motion_dataset is None:
            return

        if (
            self._playback_assignments is None
            or self._playback_assignments.shape[0] != self.num_envs
        ):
            num_motions = self._motion_dataset["root_pos"].shape[0]
            if self._env_motion_ids_tensor is not None and num_motions > 0:
                self._playback_assignments = self._env_motion_ids_tensor % num_motions
            else:
                self._playback_assignments = (
                    torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                    % max(num_motions, 1)
                )

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        motion_ids = self._playback_assignments
        max_indices = torch.clamp(self._motion_lengths[motion_ids] - 1, min=0)
        frame_tensor = torch.full_like(motion_ids, frame_idx)
        frame_tensor = torch.minimum(frame_tensor, max_indices)

        env_origins = self.scene.env_origins[env_ids]

        robot = self.scene.articulations["robot"]
        root_pos = self._motion_dataset["root_pos"][motion_ids, frame_tensor] + env_origins
        if self.cfg.replay_root_height_offset != 0.0:
            root_pos[:, 2] += self.cfg.replay_root_height_offset
        root_rot = self._motion_dataset["root_rot"][motion_ids, frame_tensor]
        root_pose = torch.cat([root_pos, root_rot], dim=-1)
        root_vel = torch.zeros((self.num_envs, 6), device=self.device)
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        joint_pos = self._motion_dataset["dof_pos"][motion_ids, frame_tensor]
        joint_vel = self._motion_dataset["dof_vel"][motion_ids, frame_tensor]
        if (
            not self._motion_dataset_aligned
            and self._dof_reorder_indices is not None
            and self._motion_dataset["dof_pos"].shape[-1] >= self._dof_reorder_indices.shape[0]
        ):
            joint_pos = joint_pos[:, self._dof_reorder_indices]
            joint_vel = joint_vel[:, self._dof_reorder_indices]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)

        if self._env_rigid_objects:
            obj_pos = self._motion_dataset["obj_pos"][motion_ids, frame_tensor] + env_origins
            if self.cfg.replay_object_height_offset != 0.0:
                obj_pos[:, 2] += self.cfg.replay_object_height_offset
            obj_rot = self._motion_dataset["obj_rot"][motion_ids, frame_tensor]
            obj_pose = torch.cat([obj_pos, obj_rot], dim=-1)
            obj_ids = self._motion_object_ids[motion_ids]
            zero_vel = torch.zeros((1, 6), device=self.device)
            env_object_ids = self._env_object_id_tensor if self._env_object_id_tensor is not None else None
            for env_idx in range(self.num_envs):
                obj_handle = self._env_rigid_objects[env_idx] if env_idx < len(self._env_rigid_objects) else None
                if obj_handle is None:
                    continue
                if env_object_ids is not None and obj_ids[env_idx] != env_object_ids[env_idx]:
                    continue
                pose = obj_pose[env_idx].unsqueeze(0)
                obj_handle.write_root_pose_to_sim(pose)
                obj_handle.write_root_com_velocity_to_sim(zero_vel)

    def play_dataset_step(self, frame_idx: int | None = None, step_sim: bool = True):
        """Replay dataset pose at the specified frame."""
        if not self.cfg.play_dataset:
            raise RuntimeError("Dataset playback is disabled. Set cfg.play_dataset=True to use this mode.")

        if frame_idx is None:
            frame_idx = self._playback_frame
            self._playback_frame += 1

        self._apply_dataset_frame(frame_idx)
        # Keep reference markers in sync during dataset playback (no RL step loop).
        if self.cfg.enable_reference_markers and self._playback_assignments is not None:
            self._update_reference_markers(frame_ids=frame_idx, motion_ids=self._playback_assignments)

        if step_sim:
            self.sim.step()
            self.scene.update(self.physics_dt)

    @staticmethod
    def _convert_gym_quaternion(quat: torch.Tensor) -> torch.Tensor:
        """Convert Isaac Gym quaternion ordering (xyzw) to Isaac Lab (wxyz)."""
        if quat.shape[-1] != 4:
            raise ValueError("Quaternion tensor must have last dimension of size 4.")
        return torch.cat([quat[..., 3:4], quat[..., 0:3]], dim=-1)

    @staticmethod
    def _convert_lab_quaternion(quat: torch.Tensor) -> torch.Tensor:
        """Convert Isaac Lab quaternion ordering (wxyz) to Isaac Lab (xyzw)."""
        if quat.shape[-1] != 4:
            raise ValueError("Quaternion tensor must have last dimension of size 4.")
        return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)
    
    def _build_target_tensors(self):
        """Build tensors for tracking object states."""
        num_actors = 2  # Humanoid + Object

        # Object states will be accessed through scene.rigid_objects
        # This is a placeholder for additional computed tensors
        pass

    def _init_buffers(self):
        """Initialize observation and reward buffers."""
        # Current and historical observations
        self._curr_obs = torch.zeros(
            (self.num_envs, self.cfg.num_observations),
            device=self.device,
            dtype=torch.float32
        )
        self._hist_obs = torch.zeros_like(self._curr_obs)

        # Reference observations for imitation
        self._curr_ref_obs = torch.zeros_like(self._curr_obs)
        self._hist_ref_obs = torch.zeros_like(self._curr_obs)

        # Reward tracking
        self._curr_reward = torch.zeros(
            (self.num_envs, self.cfg.rollout_length),
            device=self.device,
            dtype=torch.float32
        )

        # Tracking which motion each environment is following
        self.motion_ids = torch.zeros(
            self.num_envs, device=self.device, dtype=torch.long
        )

        # Build PD action offset and scale for action normalization
        if self.cfg.pd_control:
            self._build_pd_action_offset_scale()

        # Contact forces are read from PhysX via _get_contact_forces()
        # GPU-accelerated, no initialization needed

        # Store previous action for energy penalty
        self.prev_actions = torch.zeros_like(self._curr_obs[:, :self.cfg.num_actions])

        # Build body name to index mappings
        self._build_body_mappings()

        # Setup data component extraction system (for parsing HOI observations)
        self._setup_data_component_system()

        print(f"[InterMimic] Initialized buffers for {self.num_envs} environments")

    def _build_body_mappings(self):
        """Build mappings from body names to indices."""
        def _clean_name(name: str) -> str:
            return name.split("/")[-1].split(":")[-1]

        # Get body names from the robot and contact sensor (strip prim paths)
        robot_body_names_raw = list(self._robot.body_names)
        robot_body_names = [_clean_name(name) for name in robot_body_names_raw]
        robot_body_index = {name: idx for idx, name in enumerate(robot_body_names)}

        contact_sensor_body_names_raw = list(self._contact_sensor.body_names)
        contact_sensor_body_names = [_clean_name(name) for name in contact_sensor_body_names_raw]

        # Build mapping from ContactSensor body indices to Robot body indices
        self._contact_to_robot_body_indices = []
        for contact_body_name in contact_sensor_body_names:
            robot_idx = robot_body_index.get(contact_body_name, -1)
            if robot_idx == -1:
                print(f"[Warning] Contact sensor body '{contact_body_name}' not found in robot")
            self._contact_to_robot_body_indices.append(robot_idx)

        self._contact_to_robot_body_indices = torch.tensor(
            self._contact_to_robot_body_indices, dtype=torch.long, device=self.device
        )

        # Attempt to build mapping between IsaacLab body order and Isaac Gym/MJCF order
        dataset_body_index: dict[str, int] = {}
        prefer_dataset_indices = False
        try:
            dataset_body_names = self._load_mjcf_body_order(self.cfg.robot_cfg.spawn.asset_path)
            dataset_body_index = {name: idx for idx, name in enumerate(dataset_body_names)}
        except FileNotFoundError:
            print(f"[InterMimic] MJCF file not found for body remapping: {self.cfg.robot_cfg.spawn.asset_path}")
        except Exception as err:
            print(f"[InterMimic] Failed to parse MJCF for body remapping: {err}")

        if dataset_body_index:
            reorder: list[int] = []
            missing_bodies: list[str] = []
            for name in robot_body_names:
                dataset_idx = dataset_body_index.get(name)
                if dataset_idx is None:
                    missing_bodies.append(name)
                    break
                reorder.append(dataset_idx)
            if missing_bodies:
                print(f"[InterMimic] Missing bodies in dataset for remapping: {missing_bodies}")
                self._body_reorder_indices = None
                self._body_reorder_inv_indices = None
            else:
                reorder_tensor = torch.tensor(reorder, dtype=torch.long, device=self.device)
                self._body_reorder_indices = reorder_tensor
                self._body_reorder_inv_indices = torch.argsort(reorder_tensor)
                prefer_dataset_indices = True
                print("[InterMimic] Reordered body tensors to match Isaac Gym dataset order.")
        else:
            self._body_reorder_indices = None
            self._body_reorder_inv_indices = None

        index_lookup = dataset_body_index if prefer_dataset_indices else robot_body_index

        # Map key bodies to indices (order depends on whether remapping succeeded)
        self._key_body_ids = []
        for body_name in self.cfg.key_bodies:
            idx = index_lookup.get(body_name)
            if idx is None:
                print(f"[Warning] Key body '{body_name}' not found in {'dataset' if prefer_dataset_indices else 'robot'} body list")
                continue
            self._key_body_ids.append(idx)

        # Map contact bodies to indices
        self._contact_body_ids = []
        for body_name in self.cfg.contact_bodies:
            idx = index_lookup.get(body_name)
            if idx is None:
                print(f"[Warning] Contact body '{body_name}' not found in {'dataset' if prefer_dataset_indices else 'robot'} body list")
                continue
            self._contact_body_ids.append(idx)

        self._key_body_ids = torch.tensor(self._key_body_ids, dtype=torch.long, device=self.device)
        self._contact_body_ids = torch.tensor(self._contact_body_ids, dtype=torch.long, device=self.device)

        print(f"[InterMimic] Mapped {len(self._key_body_ids)} key bodies and {len(self._contact_body_ids)} contact bodies")
        print(f"[InterMimic] ContactSensor has {len(contact_sensor_body_names)} bodies, Robot has {len(robot_body_names)} bodies")

    def _build_pd_action_offset_scale(self):
        """Build PD action offset and scale from joint limits."""
        # Get joint limits from robot
        joint_limits = self._robot.root_physx_view.get_dof_limits().clone()  # (num_envs, num_dofs, 2)

        # Use first environment's limits (should be same for all)
        lim_low = joint_limits[0, :, 0]  # Lower limits
        lim_high = joint_limits[0, :, 1]  # Upper limits

        # Compute offset (center) and scale (half-range)
        self._pd_action_offset = 0.5 * (lim_high + lim_low)
        self._pd_action_scale = 0.5 * (lim_high - lim_low)

        # Handle joints with small range (< 0.1): set offset to 0 and use first scale
        small_range_mask = self._pd_action_offset.abs() > 0.1
        self._pd_action_scale[small_range_mask] = self._pd_action_scale[0].clone()
        self._pd_action_offset[small_range_mask] = 0

        # Special handling for knee joints (indices 5 and 17 for SMPL-X)
        # Scale them more aggressively
        if self.cfg.num_actions > 17:
            self._pd_action_scale[5] = 5.0  # L_Knee
            self._pd_action_scale[17] = 5.0  # R_Knee

        # Ensure parameters reside on the simulation device for action computation
        self._pd_action_offset = self._pd_action_offset.to(self.device)
        self._pd_action_scale = self._pd_action_scale.to(self.device)

        print(f"[InterMimic] Built PD action offset and scale")

    def _action_to_pd_targets(self, actions: torch.Tensor) -> torch.Tensor:
        """Convert normalized actions to PD targets.

        Args:
            actions: Normalized actions in [-1, 1]

        Returns:
            PD targets in joint space
        """
        pd_targets = self._pd_action_offset + self._pd_action_scale * actions
        return pd_targets

    def _setup_data_component_system(self):
        """Setup data component extraction system.

        Copied from Isaac Gym intermimic.py:214
        This defines the order and indices of components in the HOI observation vector.
        """
        # Define component order matching Isaac Gym format
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot',
            'body_pos_vel', 'body_rot_vel', 'obj_pos', 'obj_rot', 'obj_pos_vel',
            'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]

        # Define component sizes based on SMPL-X humanoid (52 bodies, 153 DOFs)
        data_component_sizes = [
            3,    # root_pos
            4,    # root_rot (quaternion)
            153,  # dof_pos (51 joints * 3 DOFs)
            153,  # dof_vel
            156,  # body_pos (52 bodies * 3)
            208,  # body_rot (52 bodies * 4 quaternion)
            156,  # body_pos_vel (52 bodies * 3)
            156,  # body_rot_vel (52 bodies * 3 angular vel)
            3,    # obj_pos
            4,    # obj_rot (quaternion)
            3,    # obj_pos_vel
            3,    # obj_rot_vel
            156,  # ig (52 bodies * 3 SDF vectors)
            52,   # contact_human (binary contact per body)
            1,    # contact_obj (binary contact with object)
        ]

        # Compute cumulative indices for fast extraction
        self.data_component_index = [
            sum(data_component_sizes[:i])
            for i in range(len(data_component_sizes) + 1)
        ]

        print(f"[InterMimic] Setup data component system with {len(self.data_component_order)} components")

    def extract_data_component(self, var_name, obs=None):
        """Extract a data component from observation vector.

        Copied from Isaac Gym intermimic.py:255

        Args:
            var_name: Name of component ('root_pos', 'body_rot', etc.)
            obs: Observation tensor to extract from

        Returns:
            Extracted component tensor
        """
        index = self.data_component_order.index(var_name)
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]

        if obs is not None:
            return obs[..., start:end]

        return None

    def _pre_physics_step(self, actions: torch.Tensor):
        """Process actions from policy before physics steps.

        Called ONCE per RL step. Processes actions and caches them for _apply_action().

        Args:
            actions: Actions from the policy (num_envs, num_actions).
        """
        # Ensure actions are on the sim device
        actions = actions.to(self.device)

        # Remap policy actions (dataset joint order) back to articulation order if needed
        if self._dof_reorder_indices is not None and actions.shape[-1] >= self._dof_reorder_indices.shape[0]:
            actions_robot = actions[:, self._dof_reorder_indices]
        else:
            # No remapping needed - use actions directly
            actions_robot = actions

        # Debug: Print action statistics every 100 steps
        if self.episode_length_buf[0] % 100 == 0:
            print(f"[_pre_physics_step] Step {self.episode_length_buf[0].item()}: "
                  f"actions shape={actions.shape}, "
                  f"mean={actions.mean().item():.4f}, "
                  f"std={actions.std().item():.4f}")

        if self.cfg.pd_control:
            # Convert actions to PD targets and cache them
            self._pd_targets = self._action_to_pd_targets(actions_robot)

            # Debug: Print PD target statistics every 100 steps
            if self.episode_length_buf[0] % 100 == 0:
                print(f"[_pre_physics_step] PD targets: "
                      f"mean={self._pd_targets.mean().item():.4f}, "
                      f"std={self._pd_targets.std().item():.4f}")
        else:
            # Cache scaled actions for torque control
            self._scaled_actions = actions_robot * self.cfg.power_scale

        # Update reference motion visualization markers
        self._update_reference_markers()

    def _apply_action(self):
        """Apply cached actions to the simulation.

        Called DECIMATION times per RL step (once per physics substep).
        """
        if self.cfg.pd_control:
            # Apply cached PD control targets
            self._robot.set_joint_position_target(self._pd_targets)
        else:
            # Apply cached torque control
            self._robot.set_joint_effort_target(self._scaled_actions)
        self._robot.write_data_to_sim()

    def _get_observations(self) -> Dict[str, torch.Tensor]:
        """Compute observations for the current state.

        Returns:
            Dictionary with 'policy' key containing observations.
        """
        # TODO: Implement full observation computation
        # For now, return zeros
        obs = self._compute_observations()

        return {"policy": obs}

    def _compute_observations(self) -> torch.Tensor:
        """Compute the actual observation vector matching Isaac Gym format.

        Format matches Isaac Gym InterMimic:
        - obs_buf = concatenate(compute_observations_iter(delta_t=1), compute_observations_iter(delta_t=16))
        - Each iteration includes: [humanoid_obs, task_obs, ig_all, ref_ig-ig]
        - Then convert to policy format with joint remapping and quaternion xyzw
        """
        # First, build the current HOI observation (_curr_obs) - this is the base 1211-dim obs
        self._build_hoi_observations()

        # If motion dataset or motion ids are missing, fall back to current obs only.
        if self.hoi_data is None or self._env_motion_ids_tensor is None or self._motion_lengths is None:
            return self._curr_obs

        obs_iter_1 = self._compute_observations_iter(delta_t=1)
        obs_iter_16 = self._compute_observations_iter(delta_t=16)

        # Concatenate like Isaac Gym line 677
        obs_buf = torch.cat([obs_iter_1, obs_iter_16], dim=-1)

        # obs_buf is already in Isaac Gym format (joints in IG order, wxyz quats)
        # Return directly without conversion
        return obs_buf

    def _compute_observations_iter(self, delta_t: int = 1) -> torch.Tensor:
        """Compute observations for one temporal iteration by indexing HOI data.

        Matches Isaac Gym _compute_observations_iter (line 648).

        Args:
            delta_t: Time delta for reference observation (1 or 16)

        Returns:
            Processed observation: [humanoid_obs, task_obs, ig_all, ref_ig-ig]
        """
        if (
            self.hoi_data is None
            or self._env_motion_ids_tensor is None
            or self._motion_lengths is None
        ):
            # Fallback: just return current obs if no HOI/motion ids are available
            return self._curr_obs

        env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)

        # Get current timestep (progress_buf in Isaac Gym)
        ts = self.episode_length_buf.clone()

        # Get max episode length for each environment from motion data
        motion_ids = self._env_motion_ids_tensor[env_ids]
        max_ep_lengths = self._motion_lengths[motion_ids]
        
        # Clamp current timestep to valid range for dataset sampling
        curr_ts = torch.clamp(ts, max=max_ep_lengths - 1)

        # Compute next timestep: clamp(ts + delta_t, max=max_episode_length-1)
        next_ts = torch.clamp(ts + delta_t, max=max_ep_lengths - 1)

        # Get reference observation from HOI data: hoi_data[motion_id, next_ts]
        motion_ids = self._env_motion_ids_tensor[env_ids]
        ref_obs = self.hoi_data[motion_ids, next_ts]

        # Compute humanoid observations
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)

        # Compute task observations
        task_obs = self._compute_task_obs(env_ids, ref_obs)

        # Concatenate
        obs = torch.cat([obs, task_obs], dim=-1)

        # Compute IG observations
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs)

        # Final concatenation
        return torch.cat((obs, ig_all, ref_ig - ig), dim=-1)

    def _compute_humanoid_obs(self, env_ids: torch.Tensor, ref_obs: torch.Tensor,
                             next_ts: torch.Tensor) -> torch.Tensor:
        """Compute humanoid observations using observation_utils.

        Args:
            env_ids: Environment IDs
            ref_obs: Reference observations
            next_ts: Next timestep indices

        Returns:
            Humanoid observation tensor
        """
        # Extract body states from current observation
        body_pos = self.extract_data_component('body_pos', self._curr_obs[env_ids]).view(-1, 52, 3)
        body_rot = self.extract_data_component('body_rot', self._curr_obs[env_ids]).view(-1, 52, 4)
        body_vel = self.extract_data_component('body_pos_vel', self._curr_obs[env_ids]).view(-1, 52, 3)
        body_ang_vel = self.extract_data_component('body_rot_vel', self._curr_obs[env_ids]).view(-1, 52, 3)

        # Get contact forces from PhysX (GPU-accelerated)
        contact_forces = self._get_contact_forces()
        # print("contact_forces", contact_forces)
        # Call observation_utils function
        obs = observation_utils.compute_humanoid_observations_max(
            body_pos=body_pos,
            body_rot=body_rot,
            body_vel=body_vel,
            body_ang_vel=body_ang_vel,
            local_root_obs=self.cfg.local_root_obs,
            root_height_obs=True,
            contact_forces=contact_forces,
            contact_body_ids=self._contact_body_ids,
            ref_obs=ref_obs,
            key_body_ids=self._key_body_ids,
            extract_data_component_fn=self.extract_data_component
        )

        return obs

    def _compute_task_obs(self, env_ids: torch.Tensor, ref_obs: torch.Tensor) -> torch.Tensor:
        """Compute task (object interaction) observations.

        Args:
            env_ids: Environment IDs
            ref_obs: Reference observations

        Returns:
            Task observation tensor
        """
        # Get root states
        root_pos = self.extract_data_component('root_pos', self._curr_obs[env_ids])
        root_rot = self.extract_data_component('root_rot', self._curr_obs[env_ids])
        root_states = torch.cat([root_pos, root_rot], dim=-1)

        # Get target (object) states
        tar_pos = self.extract_data_component('obj_pos', self._curr_obs[env_ids])
        tar_rot = self.extract_data_component('obj_rot', self._curr_obs[env_ids])
        tar_vel = self.extract_data_component('obj_pos_vel', self._curr_obs[env_ids])
        tar_ang_vel = self.extract_data_component('obj_rot_vel', self._curr_obs[env_ids])
        tar_states = torch.cat([tar_pos, tar_rot, tar_vel, tar_ang_vel], dim=-1)

        # Call observation_utils function
        task_obs = observation_utils.compute_obj_observations(
            root_states=root_states,
            tar_states=tar_states,
            ref_obs=ref_obs,
            extract_data_component_fn=self.extract_data_component
        )

        return task_obs

    def _compute_ig_obs(self, env_ids: torch.Tensor, ref_obs: torch.Tensor) -> tuple:
        """Compute processed interaction geometry observations.

        Matches Isaac Gym _compute_ig_obs (line 661).

        Args:
            env_ids: Environment IDs
            ref_obs: Reference observations

        Returns:
            Tuple of (ig_all, ig, ref_ig)
        """
        # Call observation_utils function
        ig_all, ig, ref_ig = observation_utils.compute_ig_obs(
            curr_obs=self._curr_obs[env_ids],
            ref_obs=ref_obs,
            key_body_ids=self._key_body_ids,
            extract_data_component_fn=self.extract_data_component
        )

        return ig_all, ig, ref_ig

    def _build_hoi_observations(self):
        """Build current HOI observation matching Isaac Gym format.

        Uses observation_utils.build_hoi_observations() copied from Isaac Gym.
        This creates _curr_obs with size 1211.
        """
        # Get environment origins for local coordinate conversion
        # In IsaacLab, envs are spawned at different locations, but observations should be env-local
        env_origins = self.scene.env_origins  # (num_envs, 3)

        # Get robot state from IsaacLab - adapting sensor access for IsaacLab
        root_pos_w = self._robot.data.root_pos_w  # (num_envs, 3) - world coordinates
        root_quat_w = self._robot.data.root_quat_w  # (num_envs, 4) - wxyz format
        root_lin_vel_w = self._robot.data.root_lin_vel_w  # (num_envs, 3)
        root_ang_vel_w = self._robot.data.root_ang_vel_w  # (num_envs, 3)

        # Convert root position from world to environment-local coordinates
        # This makes observations independent of env spawn location
        root_pos = root_pos_w - env_origins
        root_quat = self._convert_lab_quaternion(root_quat_w)
        
        # Get DOF states in IsaacLab order
        dof_pos_isaaclab = self._robot.data.joint_pos  # (num_envs, num_dofs)
        dof_vel_isaaclab = self._robot.data.joint_vel  # (num_envs, num_dofs)

        # Remap from IsaacLab order to Isaac Gym/dataset order for HOI observations
        dof_pos = dof_pos_isaaclab[:, self._dof_reorder_inv_indices]
        dof_vel = dof_vel_isaaclab[:, self._dof_reorder_inv_indices]

        # Get body states
        body_pos_w = self._robot.data.body_pos_w  # (num_envs, num_bodies, 3) - world coordinates
        body_quat_w = self._robot.data.body_quat_w  # (num_envs, num_bodies, 4)
        body_vel_w = self._robot.data.body_vel_w  # (num_envs, num_bodies, 6)

        # Convert body positions from world to environment-local coordinates
        body_pos = body_pos_w - env_origins.unsqueeze(1)  # Subtract env origin from each body
        body_quat = self._convert_lab_quaternion(body_quat_w)
        
        # Extract linear and angular velocities
        body_lin_vel_w = body_vel_w[..., :3]
        body_ang_vel_w = body_vel_w[..., 3:]

        # Reorder body tensors to Isaac Gym (dataset) order if mapping available
        if self._body_reorder_inv_indices is not None:
            idx = self._body_reorder_inv_indices
            body_pos = torch.index_select(body_pos, dim=1, index=idx)
            body_quat = torch.index_select(body_quat, dim=1, index=idx)
            body_lin_vel_w = torch.index_select(body_lin_vel_w, dim=1, index=idx)
            body_ang_vel_w = torch.index_select(body_ang_vel_w, dim=1, index=idx)

        # Get object states (returns 13-dim: pos + quat + lin_vel + ang_vel)
        # Object positions are also converted to env-local coordinates
        target_states = self._get_object_states()

        # Get contact forces from PhysX (GPU-accelerated, all environments in parallel)
        contact_buf = self._get_contact_forces()
        target_contact_buf = self._get_object_contact_forces()

        # Get object surface points for IG computation
        object_points = self._get_object_points_for_envs()
        # Call Isaac Gym's build_hoi_observations function
        # NOTE: Use env-local coordinates (root_pos, body_pos) not world coordinates
        self._curr_obs = observation_utils.build_hoi_observations(
            root_pos=root_pos,  # env-local
            root_rot=root_quat,
            root_vel=root_lin_vel_w,
            root_ang_vel=root_ang_vel_w,
            dof_pos=dof_pos,
            dof_vel=dof_vel,
            body_pos=body_pos,  # env-local
            local_root_obs=self.cfg.local_root_obs,
            root_height_obs=True,
            dof_obs_size=self.cfg.num_actions,
            target_states=target_states,  # Already env-local from _get_object_states
            target_contact_buf=target_contact_buf,
            contact_buf=contact_buf,
            object_points=object_points,
            body_rot=body_quat,
            body_vel=body_lin_vel_w,
            body_rot_vel=body_ang_vel_w,
            compute_sdf_fn=torch_utils.compute_sdf
        )

    def _get_contact_forces(self) -> torch.Tensor:
        """Get net contact forces on each rigid body (GPU-accelerated).

        Uses IsaacLab's ContactSensor which provides GPU-accelerated access to
        contact forces via PhysX ContactReporter API. This enables large-scale
        parallel RL training with thousands of environments.

        Returns:
            Contact forces tensor (num_envs, num_bodies, 3) in world frame
            Reordered to match robot's body order.
            All operations are GPU-accelerated for parallel training.
        """
        # GPU-accelerated: ContactSensor provides net forces on all bodies
        # net_forces_w shape: (num_envs, num_contact_bodies, 3)
        # No CPU transfer - fully parallel across all environments
        contact_forces_sensor = self._contact_sensor.data.net_forces_w

        # Reorder from ContactSensor's body order to Robot's body order
        # Create output tensor with robot's body count
        num_robot_bodies = len(self._robot.body_names)
        contact_forces_robot = torch.zeros(
            (self.num_envs, num_robot_bodies, 3),
            dtype=contact_forces_sensor.dtype,
            device=self.device
        )

        # Map forces from sensor order to robot order using precomputed indices
        # Only copy valid indices (where mapping exists)
        valid_mask = self._contact_to_robot_body_indices >= 0
        if valid_mask.any():
            valid_sensor_indices = torch.arange(len(self._contact_to_robot_body_indices), device=self.device)[valid_mask]
            valid_robot_indices = self._contact_to_robot_body_indices[valid_mask]
            contact_forces_robot[:, valid_robot_indices, :] = contact_forces_sensor[:, valid_sensor_indices, :]

        # Reorder to dataset body order if mapping is available
        if self._body_reorder_inv_indices is not None:
            contact_forces_robot = torch.index_select(
                contact_forces_robot, dim=1, index=self._body_reorder_inv_indices
            )

        return contact_forces_robot

    def _get_object_contact_forces(self) -> torch.Tensor:
        """Get net contact forces acting on the interaction objects."""
        if self._object_contact_sensor is None:
            return torch.zeros((self.num_envs, 3), device=self.device)

        contact_forces = self._object_contact_sensor.data.net_forces_w
        if contact_forces.ndim == 3:
            forces = contact_forces.sum(dim=1)
        elif contact_forces.ndim == 2:
            forces = contact_forces
        else:
            return torch.zeros((self.num_envs, 3), device=self.device, dtype=contact_forces.dtype)

        if forces.shape[0] != self.num_envs:
            out = torch.zeros((self.num_envs, 3), device=self.device, dtype=forces.dtype)
            count = min(forces.shape[0], self.num_envs)
            out[:count] = forces[:count]
            return out

        return forces

    def _get_object_points_for_envs(self) -> torch.Tensor:
        if self.object_points.shape[0] == 0:
            return torch.zeros((self.num_envs, 1024, 3), device=self.device)

        if self._env_object_id_tensor is not None:
            env_obj_points_local = self.object_points[self._env_object_id_tensor]  # (E, P, 3)
        else:
            env_obj_points_local = self.object_points[0:1].expand(self.num_envs, -1, -1)

        return env_obj_points_local

    def _get_object_states(self) -> torch.Tensor:
        """Get object states for all environments in env-local coordinates.

        Returns:
            Object states (num_envs, 13) containing pos(3) + quat(4) + vel(3) + ang_vel(3)
            Positions are relative to environment origins.
        """
        obj_states = torch.zeros((self.num_envs, 13), device=self.device)
        
        env_origins = self.scene.env_origins  # (num_envs, 3)

        for env_id, obj in enumerate(self._env_rigid_objects):
            if obj is not None:
                # Get object state in world coordinates
                obj_pos_w = obj.data.root_pos_w[0]  # Object only has 1 instance
                obj_quat_w = obj.data.root_quat_w[0]
                obj_lin_vel = obj.data.root_lin_vel_w[0]
                obj_ang_vel = obj.data.root_ang_vel_w[0]

                # Convert position to env-local coordinates
                obj_pos = obj_pos_w - env_origins[env_id]
                obj_quat = self._convert_lab_quaternion(obj_quat_w)
                obj_states[env_id] = torch.cat([obj_pos, obj_quat, obj_lin_vel, obj_ang_vel])
                
        return obj_states

    def _get_rewards(self) -> torch.Tensor:
        """Compute rewards for the current state.

        Returns:
            Reward tensor (num_envs,).
        """
        # TODO: Implement reward computation
        # For now, return zeros
        rewards = torch.zeros(self.num_envs, device=self.device)

        return rewards

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Check termination conditions.

        Returns:
            Tuple of (terminated, truncated) boolean tensors.
        """
        # Check for early termination (fallen)
        if self.cfg.enable_early_termination:
            root_pos = self._robot.data.root_pos_w
            terminated = root_pos[:, 2] < self.cfg.termination_height
        else:
            terminated = torch.zeros(
                self.num_envs, device=self.device, dtype=torch.bool
            )

        # Truncation based on episode length
        truncated = self.episode_length_buf >= self.max_episode_length

        return terminated, truncated

    def _reset_idx(self, env_ids: torch.Tensor):
        """Reset specified environments.

        Args:
            env_ids: Indices of environments to reset.
        """
        num_resets = len(env_ids)

        if num_resets == 0:
            return

        # NOTE: We don't call super()._reset_idx() because we have per-environment
        # RigidObjects which don't follow the standard IsaacLab pattern.
        # Instead, we manually reset robot and objects below.

        # Reset robot and object states based on state_init mode
        # Note: Objects are reset within _apply_reference_state()
        if self.cfg.state_init == "Default":
            self._reset_to_default(env_ids)
        elif self.cfg.state_init == "Random":
            self._reset_to_random(env_ids)
        elif self.cfg.state_init == "Hybrid":
            self._reset_to_hybrid(env_ids)
        else:
            self._reset_to_default(env_ids)

        # Reset episode tracking
        self.episode_length_buf[env_ids] = 0

        # Clear observation buffers
        self._curr_obs[env_ids] = 0
        self._hist_obs[env_ids] = 0

        if num_resets > 0 and env_ids[0] == 0:  # Only print for first env to avoid spam
            print(f"[InterMimic] Reset {num_resets} environments")

    def _reset_to_default(self, env_ids: torch.Tensor):
        """Reset to reference motion first frame if available, otherwise default pose."""
        if self._apply_reference_state(env_ids, frame_idx=0):
            return

        # Fallback: default configuration from asset
        self._robot.write_root_pose_to_sim(
            self._robot.data.default_root_state[env_ids, :7],
            env_ids=env_ids
        )
        self._robot.write_joint_state_to_sim(
            self._robot.data.default_joint_pos[env_ids],
            self._robot.data.default_joint_vel[env_ids],
            env_ids=env_ids
        )

    def _reset_to_random(self, env_ids: torch.Tensor):
        """Reset to random pose."""
        # TODO: Implement random pose sampling
        if not self._apply_reference_state(env_ids, frame_idx=0):
            self._reset_to_default(env_ids)

    def _reset_to_hybrid(self, env_ids: torch.Tensor):
        """Reset with hybrid initialization (mix of default and reference)."""
        num_resets = len(env_ids)

        # Randomly choose between default and reference initialization
        use_ref = torch.rand(num_resets, device=self.device) < self.cfg.hybrid_init_prob

        # Reset environments
        default_ids = env_ids[~use_ref]
        ref_ids = env_ids[use_ref]

        if len(default_ids) > 0:
            self._reset_to_default(default_ids)

        if len(ref_ids) > 0:
            if not self._apply_reference_state(ref_ids, frame_idx=0):
                self._reset_to_default(ref_ids)

    def _apply_reference_state(self, env_ids: torch.Tensor, frame_idx: int = 0) -> bool:
        """Apply reference motion state (first frame by default) to selected environments.

        Returns:
            True if reference state was applied, False if no motion data was available.
        """
        if self._motion_dataset is None or self._motion_lengths is None or len(self.motion_files) == 0:
            return False

        # Ensure we have motion assignments per env
        if (
            self._playback_assignments is None
            or self._playback_assignments.shape[0] != self.num_envs
        ):
            num_motions = self._motion_dataset["root_pos"].shape[0]
            if self._env_motion_ids_tensor is not None and num_motions > 0:
                self._playback_assignments = self._env_motion_ids_tensor % num_motions
            else:
                self._playback_assignments = (
                    torch.arange(self.num_envs, device=self.device, dtype=torch.long)
                    % max(num_motions, 1)
                )

        motion_ids = self._playback_assignments[env_ids]
        max_indices = torch.clamp(self._motion_lengths[motion_ids] - 1, min=0)
        frame_tensor = torch.full_like(motion_ids, frame_idx)
        frame_tensor = torch.minimum(frame_tensor, max_indices)

        env_origins = self.scene.env_origins[env_ids]

        robot = self.scene.articulations["robot"]
        root_pos = self._motion_dataset["root_pos"][motion_ids, frame_tensor] + env_origins
        if self.cfg.replay_root_height_offset != 0.0:
            root_pos[:, 2] += self.cfg.replay_root_height_offset
        root_rot = self._motion_dataset["root_rot"][motion_ids, frame_tensor]
        root_pose = torch.cat([root_pos, root_rot], dim=-1)
        root_vel = torch.zeros((len(env_ids), 6), device=self.device)
        robot.write_root_link_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_link_velocity_to_sim(root_vel, env_ids=env_ids)

        joint_pos = self._motion_dataset["dof_pos"][motion_ids, frame_tensor]
        joint_vel = self._motion_dataset["dof_vel"][motion_ids, frame_tensor]
        if (
            not self._motion_dataset_aligned
            and self._dof_reorder_indices is not None
            and self._motion_dataset["dof_pos"].shape[-1] >= self._dof_reorder_indices.shape[0]
        ):
            joint_pos = joint_pos[:, self._dof_reorder_indices]
            joint_vel = joint_vel[:, self._dof_reorder_indices]
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids=env_ids)

        if self._env_rigid_objects:
            obj_pos = self._motion_dataset["obj_pos"][motion_ids, frame_tensor] + env_origins
            if self.cfg.replay_object_height_offset != 0.0:
                obj_pos[:, 2] += self.cfg.replay_object_height_offset
            obj_rot = self._motion_dataset["obj_rot"][motion_ids, frame_tensor]
            obj_pose = torch.cat([obj_pos, obj_rot], dim=-1)
            obj_ids = self._motion_object_ids[motion_ids]
            zero_vel = torch.zeros((1, 6), device=self.device)
            env_object_ids = self._env_object_id_tensor if self._env_object_id_tensor is not None else None
            for idx, env_idx in enumerate(env_ids.tolist()):
                obj_handle = self._env_rigid_objects[env_idx] if env_idx < len(self._env_rigid_objects) else None
                if obj_handle is None:
                    continue
                if env_object_ids is not None and obj_ids[idx] != env_object_ids[env_idx]:
                    continue
                pose = obj_pose[idx].unsqueeze(0)
                obj_handle.write_root_pose_to_sim(pose)
                obj_handle.write_root_com_velocity_to_sim(zero_vel)
                # obj_handle.write_data_to_sim()  # Actually apply the changes to simulation

        return True

    def _setup_reference_markers(self):
        """Setup visualization markers for reference motion (joints and object)."""
        from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

        # Only setup markers if visualization is enabled
        if not self.cfg.enable_reference_markers:
            return

        # Create marker configuration for reference joints (small blue spheres)
        joint_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/ReferenceJoints",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.02,  # 2cm spheres
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(0.8, 0.5, 0.3),  # Blue
                        opacity=0.6,
                    ),
                )
            },
        )

        # Create marker configuration for reference object (red sphere, one per env)
        object_marker_cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/ReferenceObject",
            markers={
                "sphere": sim_utils.SphereCfg(
                    radius=0.04,
                    visual_material=PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.2, 0.2),  # Red
                        opacity=0.7,
                    ),
                )
            },
        )

        # Initialize marker managers with proper counts
        self._reference_joint_markers = VisualizationMarkers(joint_marker_cfg)
        self._reference_object_markers = VisualizationMarkers(object_marker_cfg)

    def _update_reference_markers(
        self,
        frame_ids: int | torch.Tensor | None = None,
        motion_ids: torch.Tensor | None = None,
    ):
        """Update reference motion visualization markers with current reference pose.

        Args:
            frame_ids: Optional frame index (scalar) or per-env indices (shape: (num_envs,)).
                Defaults to the current episode step counter (`episode_length_buf`).
            motion_ids: Optional per-env motion indices (shape: (num_envs,)).
                Defaults to `_env_motion_ids_tensor` if available.
        """
        if self._reference_joint_markers is None or self._reference_object_markers is None:
            return
        # Resolve motion ids
        if motion_ids is None:
            motion_ids = self._env_motion_ids_tensor
        if motion_ids is None or self._motion_lengths is None:
            return
        motion_ids = motion_ids.to(device=self.device, dtype=torch.long)

        # Resolve frame ids
        if frame_ids is None:
            frame_ids_tensor = self.episode_length_buf.to(device=self.device, dtype=torch.long)
        elif isinstance(frame_ids, int):
            frame_ids_tensor = torch.full((motion_ids.shape[0],), frame_ids, device=self.device, dtype=torch.long)
        else:
            frame_ids_tensor = frame_ids.to(device=self.device, dtype=torch.long)
            if frame_ids_tensor.ndim == 0:
                frame_ids_tensor = frame_ids_tensor.expand(motion_ids.shape[0])
            elif frame_ids_tensor.shape[0] != motion_ids.shape[0]:
                if frame_ids_tensor.numel() == 1:
                    frame_ids_tensor = frame_ids_tensor.reshape(1).expand(motion_ids.shape[0])
                else:
                    raise ValueError(
                        f"frame_ids must be scalar or shape ({motion_ids.shape[0]},), got {tuple(frame_ids_tensor.shape)}"
                    )
        # Get environment origins for positioning markers in world space
        env_origins = self.scene.env_origins  # Shape: (num_envs, 3)

        # Get reference body positions from hoi_data using extract_data_component
        if self.hoi_data is not None:

            # Extract body_pos from hoi_data: (num_motions, max_length, 1211)
            # body_pos is at indices determined by data_component_order
            body_pos_flat = self.extract_data_component("body_pos", self.hoi_data)  # (num_motions, max_length, 156)
            num_motions, max_length, _ = body_pos_flat.shape
            body_pos = body_pos_flat.view(num_motions, max_length, 52, 3)

            # Frame ids clamped to motion length
            clamped_frame_ids = torch.minimum(
                frame_ids_tensor,
                (self._motion_lengths[motion_ids] - 1).clamp(min=0),
            )

            # Gather body positions for all envs and key bodies
            # body_pos_sel: (num_envs, 52, 3)
            body_pos_sel = body_pos[motion_ids, clamped_frame_ids]
            joint_positions = body_pos_sel[:, :, :] + env_origins.unsqueeze(1)

            num_envs = joint_positions.shape[0]
            num_key_bodies = joint_positions.shape[1]
            joint_positions_flat = joint_positions.reshape(num_envs * num_key_bodies, 3)
            joint_orientations_flat = torch.zeros((num_envs * num_key_bodies, 4), device=self.device)
            joint_orientations_flat[:, 0] = 1.0  # Identity quaternion (w, x, y, z)
            
            # Update joint markers
            self._reference_joint_markers.visualize(
                translations=joint_positions_flat,
                orientations=joint_orientations_flat,
            )

        # Get reference object pose
        if "obj_pos" in self._motion_dataset:
            num_envs = motion_ids.shape[0]

            # Clamp frames to valid range
            clamped_frame_ids = torch.minimum(
                frame_ids_tensor,
                (self._motion_lengths[motion_ids] - 1).clamp(min=0),
            )

            object_positions = self._motion_dataset["obj_pos"][motion_ids, clamped_frame_ids] + env_origins
            if "obj_rot" in self._motion_dataset:
                object_orientations = self._motion_dataset["obj_rot"][motion_ids, clamped_frame_ids]
            else:
                object_orientations = torch.zeros((num_envs, 4), device=self.device)
                object_orientations[:, 0] = 1.0  # Identity quaternion (w, x, y, z)

            # Update object markers
            self._reference_object_markers.visualize(
                translations=object_positions,
                orientations=object_orientations,
            )

    def _setup_debug_viz(self):
        """Setup debug visualization markers."""
        # TODO: Add visualization for contact points, reference poses, etc.
        pass


# Register environment
# This allows loading with: env = gym.make("InterMimic-v0")
import gymnasium as gym
from isaaclab.envs import ManagerBasedRLEnvCfg

# Note: Proper registration will be done separately
