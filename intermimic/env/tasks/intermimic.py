from enum import Enum
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *

from utils import torch_utils
import torch.nn.functional as F
from env.tasks.humanoid import *
import trimesh



class InterMimic(Humanoid_SMPLX):
    class StateInit(Enum):
        Default = 0
        Start = 1
        Random = 2
        Hybrid = 3

    def __init__(self, cfg, sim_params, physics_engine, device_type, device_id, headless):
        state_init = cfg["env"]["stateInit"]
        self._state_init = InterMimic.StateInit[state_init]
        self._hybrid_init_prob = cfg["env"]["hybridInitProb"]

        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []
        self.motion_file = cfg['env']['motion_file']
        self.play_dataset = cfg['env']['playdataset']
        self.reward_weights = cfg["env"]["rewardWeights"]
        self.save_images = cfg['env']['saveImages']
        self.init_vel = cfg['env']['initVel']
        self.ball_size = cfg['env']['ballSize']
        self.more_rigid = cfg['env']['moreRigid']
        self.rollout_length = cfg['env']['rolloutLength']
        self.psi = cfg['env'].get('physicalBufferSize', 1)
        motion_file = os.listdir(self.motion_file)
        self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file if data_path.split('_')[0] in cfg['env']['dataSub']])
        self.object_name = [motion_example.split('_')[-2] for motion_example in self.motion_file]
        object_name_set = sorted(list(set(self.object_name)))
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).cuda()
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']
        self.ref_hoi_obs_size = 7 + 51 * 6 + 52 * 13 + 13 + 52 * 3 + 52 + 1
        self.num_motions = len(self.motion_file)
        self.dataset_index = to_torch([int(data_path.split('/')[-1].split('_')[0][3:]) for data_path in self.motion_file], dtype=torch.long).cuda()

        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self.hoi_data = self._load_motion(self.motion_file, topk=self.psi)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self.kinematic_reset = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self.contact_reset = torch.zeros((self.num_envs, 2), device=self.device, dtype=torch.float)
        self.dataset_id = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self._curr_reward = torch.zeros([self.num_envs, cfg['env']['rolloutLength']], device=self.device, dtype=torch.float)
        self._sum_reward = torch.zeros([self.num_envs], device=self.device, dtype=torch.float)
        self._curr_state = torch.zeros([self.num_envs, cfg['env']['rolloutLength'], 332], device=self.device, dtype=torch.float)
        self._build_target_tensors()

        return

    def post_physics_step(self):
        super().post_physics_step()
        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return

    def _load_motion(self, motion_file, startk=0, topk=1, initk=0):

        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path)[startk:]
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

        
            max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

            loaded_dict['dof_vel'] = []

            loaded_dict['dof_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_vel'].shape[-1])).to('cuda'),loaded_dict['dof_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone()
            loaded_dict['body_pos_vel'] = (loaded_dict['body_pos'][1:,:].clone() - loaded_dict['body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['body_pos_vel']),dim=0)

            loaded_dict['obj_pos'] = loaded_dict['hoi_data'][:, 318:321].clone()

            loaded_dict['obj_pos_vel'] = (loaded_dict['obj_pos'][1:,:].clone() - loaded_dict['obj_pos'][:-1,:].clone())*self.fps_data
            if self.init_vel:
                loaded_dict['obj_pos_vel'] = torch.cat((loaded_dict['obj_pos_vel'][:1],loaded_dict['obj_pos_vel']),dim=0)
            else:
                loaded_dict['obj_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_pos_vel'].shape[-1])).to('cuda'),loaded_dict['obj_pos_vel']),dim=0)


            loaded_dict['obj_rot'] = loaded_dict['hoi_data'][:, 321:325].clone()
            obj_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['obj_rot'])
            loaded_dict['obj_rot_vel'] = (obj_rot_exp_map[1:,:].clone() - obj_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['obj_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['obj_rot_vel'].shape[-1])).to('cuda'),loaded_dict['obj_rot_vel']),dim=0)


            obj_rot_extend = loaded_dict['obj_rot'].unsqueeze(1).repeat(1, self.object_points[self.object_id[idx]].shape[0], 1).view(-1, 4)
            object_points_extend = self.object_points[self.object_id[idx]].unsqueeze(0).repeat(loaded_dict['obj_rot'].shape[0], 1, 1).view(-1, 3)
            obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(loaded_dict['obj_rot'].shape[0], self.object_points[self.object_id[idx]].shape[0], 3) + loaded_dict['obj_pos'].unsqueeze(1)

            ref_ig = compute_sdf(loaded_dict['body_pos'].view(max_episode_length[-1],52,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, loaded_dict['body_pos'].shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['ig'] = ref_ig
            loaded_dict['contact_obj'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
            loaded_dict['contact_human'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())
            loaded_dict['body_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone()

            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['body_rot'].view(-1, 4)).view(-1, 52*3)
            loaded_dict['body_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['body_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['body_rot_vel'].shape[-1])).to('cuda'),loaded_dict['body_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), 
                                                    loaded_dict['root_rot'].clone(), 
                                                    loaded_dict['dof_pos'].clone(), 
                                                    loaded_dict['dof_vel'].clone(),
                                                    loaded_dict['body_pos'].clone(),
                                                    loaded_dict['body_rot'].clone(),
                                                    loaded_dict['body_pos_vel'].clone(),
                                                    loaded_dict['body_rot_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(), 
                                                    loaded_dict['obj_rot_vel'].clone(),
                                                    loaded_dict['ig'].clone(),
                                                    loaded_dict['contact_human'].clone(),
                                                    loaded_dict['contact_obj'].clone(),
                                                    ),dim=-1)
            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            loaded_dict['hoi_data'] = torch.cat([loaded_dict['hoi_data'][0:1] for _ in range(initk)]+[loaded_dict['hoi_data']], dim=0)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), 
                                loaded_dict['root_rot'].clone(), 
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(), 
                                loaded_dict['dof_pos'].clone(), 
                                loaded_dict['dof_vel'].clone(), 
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(max_episode_length) + initk
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(max_episode_length, dtype=torch.long) + initk
        hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        hoi_data = torch.stack(hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, topk, 1, 1)

        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        if not hasattr(self, 'data_component_order'):
            self.create_component_stat(loaded_dict)
        return hoi_data

    def create_component_stat(self, loaded_dict):
        self.data_component_order = [
            'root_pos', 'root_rot', 'dof_pos', 'dof_vel', 'body_pos', 'body_rot', 'body_pos_vel', 'body_rot_vel',
            'obj_pos', 'obj_rot', 'obj_pos_vel', 'obj_rot_vel', 'ig', 'contact_human', 'contact_obj'
        ]

        # Precompute the sizes for each component.
        data_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.data_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.data_component_index = [sum(data_component_sizes[:i]) for i in range(len(data_component_sizes) + 1)]

        self.ref_component_order = [
            'root_pos', 'root_rot', 'root_pos_vel', 'root_rot_vel', 'dof_pos', 'dof_vel', 'obj_pos', 'obj_rot', 
            'obj_pos_vel', 'obj_rot_vel'
        ]

        # Precompute the sizes for each component.
        ref_component_sizes = [
            loaded_dict[name].shape[1]
            for name in self.ref_component_order
        ]

        # Precompute cumulative indices. The first index is zero.
        # For each i, calculate the sum of component_sizes[:i] to determine the starting index for that component.
        self.ref_component_index = [sum(ref_component_sizes[:i]) for i in range(len(ref_component_sizes) + 1)]

    def extract_ref_component(self, var_name, data_id, ref_index, t):
        index = self.ref_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.ref_component_index[index]
        end = self.ref_component_index[index+1]
        
        return self.hoi_refs[data_id, ref_index, t, start:end]


    def extract_data_component(self, var_name, ref=False, data_id=None, t=None, obs=None):
        index = self.data_component_order.index(var_name)
        
        # The number of columns to extract for this component.
        start = self.data_component_index[index]
        end = self.data_component_index[index+1]
        
        if ref and data_id is not None and t is not None:
            return self.hoi_data[data_id, t, start:end]
        
        if obs is not None:
            return obs[..., start:end]

    def _create_envs(self, num_envs, spacing, num_per_row):

        self._target_handles = []
        self._load_target_asset()
        super()._create_envs(num_envs, spacing, num_per_row)
        return

    def _build_env(self, env_id, env_ptr, humanoid_asset):
        super()._build_env(env_id, env_ptr, humanoid_asset)

        self._build_target(env_id, env_ptr)
        return   

    def _load_target_asset(self): # smplx
        asset_root = "intermimic/data/assets/objects/"
        self._target_asset = []
        points_num = []
        self.object_points = []
        for i, object_name in enumerate(self.object_name):

            asset_file = object_name + ".urdf"
            obj_file = asset_root + 'objects/' + object_name + '/' + object_name + '.obj'
            max_convex_hulls = 64
            density = self.object_density
        
            asset_options = gymapi.AssetOptions()
            asset_options.angular_damping = 0.01
            asset_options.linear_damping = 0.01

            asset_options.density = density
            asset_options.default_dof_drive_mode = gymapi.DOF_MODE_NONE
            asset_options.vhacd_enabled = True
            asset_options.vhacd_params.max_convex_hulls = max_convex_hulls
            asset_options.vhacd_params.max_num_vertices_per_ch = 64
            asset_options.vhacd_params.resolution = 300000


            self._target_asset.append(self.gym.load_asset(self.sim, asset_root, asset_file, asset_options))

            mesh_obj = trimesh.load(obj_file, force='mesh')
            obj_verts = mesh_obj.vertices
            center = np.mean(obj_verts, 0)
            object_points, object_faces = trimesh.sample.sample_surface_even(mesh_obj, count=1024, seed=2024)

            object_points = to_torch(object_points - center)
            

            while object_points.shape[0] < 1024:
                object_points = torch.cat([object_points, object_points[:1024 - object_points.shape[0]]], dim=0)
            self.object_points.append(to_torch(object_points))
        
        self.object_points = torch.stack(self.object_points, dim=0)
        return

    def _build_target(self, env_id, env_ptr):
        col_group = env_id
        col_filter = 0
        segmentation_id = 0

        default_pose = gymapi.Transform()
        
        target_handle = self.gym.create_actor(env_ptr, self._target_asset[env_id % len(self.object_name)], default_pose, self.object_name[env_id % len(self.object_name)], col_group, col_filter, segmentation_id)

        props = self.gym.get_actor_rigid_shape_properties(env_ptr, target_handle)
        for p_idx in range(len(props)):
            props[p_idx].restitution = 0.6
            props[p_idx].friction = 0.8
            props[p_idx].rolling_friction = 0.01
            props[p_idx].torsion_friction = 0.8
        self.gym.set_actor_rigid_shape_properties(env_ptr, target_handle, props)

        self._target_handles.append(target_handle)
        self.gym.set_actor_scale(env_ptr, target_handle, self.ball_size)

        return

    def _build_target_tensors(self):
        num_actors = self.get_num_actors_per_env()
        self._target_states = self._root_states.view(self.num_envs, num_actors, self._root_states.shape[-1])[..., 1, :]
        
        self._tar_actor_ids = to_torch(num_actors * np.arange(self.num_envs), device=self.device, dtype=torch.int32) + 1
        
        bodies_per_env = self._rigid_body_state.shape[0] // self.num_envs
        contact_force_tensor = self.gym.acquire_net_contact_force_tensor(self.sim)
        contact_force_tensor = gymtorch.wrap_tensor(contact_force_tensor)
        self._tar_contact_forces = contact_force_tensor.view(self.num_envs, bodies_per_env, 3)[..., self.num_bodies, :]
        return
    
    def _reset_target(self, env_ids):
        self._target_states[env_ids, :3] = self.extract_ref_component('obj_pos', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 3:7] = self.extract_ref_component('obj_rot', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 7:10] = self.extract_ref_component('obj_pos_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        self._target_states[env_ids, 10:13] = self.extract_ref_component('obj_rot_vel', self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids])
        return  

    def _reset_env_tensors(self, env_ids):
        super()._reset_env_tensors(env_ids)


        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
    
        return

    def _reset_envs(self, env_ids):
        self._reset_default_env_ids = []
        self._reset_ref_env_ids = []

        super()._reset_envs(env_ids)

        return

    def _reset_actors(self, env_ids):
        if (self._state_init == InterMimic.StateInit.Default):
            self._reset_default(env_ids)
        elif (self._state_init == InterMimic.StateInit.Start
              or self._state_init == InterMimic.StateInit.Random):
            self._reset_ref_state_init(env_ids)
        elif (self._state_init == InterMimic.StateInit.Hybrid):
            self._reset_hybrid_state_init(env_ids)
        else:
            assert(False), "Unsupported state initialization strategy: {:s}".format(str(self._state_init))
        self._reset_target(env_ids)

        return

    def _reset_default(self, env_ids):
        self._humanoid_root_states[env_ids] = self._initial_humanoid_root_states[env_ids]
        self._dof_pos[env_ids] = self._initial_dof_pos[env_ids]
        self._dof_vel[env_ids] = self._initial_dof_vel[env_ids]
        self._reset_default_env_ids = env_ids
        return

    def _reset_ref_state_init(self, env_ids):
        num_envs = env_ids.shape[0]

        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)

        if (self._state_init == InterMimic.StateInit.Random
            or self._state_init == InterMimic.StateInit.Hybrid):
            motion_times = torch.cat([torch.randint(0, max(1, self.max_episode_length[i[e]]-self.rollout_length), (1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        elif (self._state_init == InterMimic.StateInit.Start):
            motion_times = torch.zeros(num_envs, device=self.device, dtype=torch.long)#.int()

        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )

        return

    def cal_cdf(self, i, e):
        rewards = self.ref_reward[i[e], :, :max(1, self.max_episode_length[i[e]]-self.rollout_length)].clone() 
        ref_reward_sum = 1 / (rewards.sum(dim=0)) 
        prob = ref_reward_sum / ref_reward_sum.sum()
        cdf = torch.cumsum(prob, 0)
        return cdf

    def _reset_hybrid_state_init(self, env_ids):
        num_envs = env_ids.shape[0]
        i = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in env_ids], device=self.device, dtype=torch.long)
        ref_probs = to_torch(np.array([self._hybrid_init_prob] * num_envs), device=self.device)
        ref_init_mask = torch.bernoulli(ref_probs) == 1.0

        ref_reset_ids = env_ids[ref_init_mask]

        motion_times = torch.cat([torch.searchsorted(self.cal_cdf(i, e), torch.rand(1).to(self.device)) if env_ids[e] not in ref_reset_ids else torch.zeros((1,), device=self.device, dtype=torch.long) for e in range(num_envs)]) 
        ref_reward = self.ref_reward[i, :, motion_times] 
        prob = ref_reward / ref_reward.sum(1, keepdim=True)

        cdf = torch.cumsum(prob, dim=1)
        idx = torch.searchsorted(cdf, torch.rand((cdf.shape[0], 1)).to(cdf.device)).squeeze(1)
        self.ref_index[env_ids] = idx
        self.progress_buf[env_ids] = motion_times.clone()
        self.start_times[env_ids] = motion_times.clone()
        self.data_id[env_ids] = i
        self.dataset_id[env_ids] = self.dataset_index[self.data_id[env_ids]]
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.extract_ref_component('root_pos', i, idx, motion_times),
                            root_rot=self.extract_ref_component('root_rot', i, idx, motion_times),
                            dof_pos=self.extract_ref_component('dof_pos', i, idx, motion_times),
                            root_vel=self.extract_ref_component('root_pos_vel', i, idx, motion_times),
                            root_ang_vel=self.extract_ref_component('root_rot_vel', i, idx, motion_times),
                            dof_vel=self.extract_ref_component('dof_vel', i, idx, motion_times),
                            )
        return

    def _set_env_state(self, env_ids, root_pos, root_rot, dof_pos, root_vel, root_ang_vel, dof_vel):
        self._humanoid_root_states[env_ids, 0:3] = root_pos
        self._humanoid_root_states[env_ids, 3:7] = root_rot
        self._humanoid_root_states[env_ids, 7:10] = root_vel
        self._humanoid_root_states[env_ids, 10:13] = root_ang_vel
        
        self._dof_pos[env_ids] = dof_pos
        self._dof_vel[env_ids] = dof_vel
        return

    def _compute_task_obs(self, env_ids=None, ref_obs=None):
        if (env_ids is None):
            root_states = self._humanoid_root_states
            tar_states = self._target_states
        else:
            root_states = self._humanoid_root_states[env_ids]
            tar_states = self._target_states[env_ids]
        
        obs = self.compute_obj_observations(root_states, tar_states, ref_obs)
        return obs

    def compute_humanoid_observations_max(self, body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs, contact_forces, contact_body_ids, ref_obs, key_body_ids):
        # type: (Tensor, Tensor, Tensor, Tensor, bool, bool, Tensor, Tensor, Tensor, Tensor) -> Tensor
        root_pos = body_pos[:, 0, :]
        root_rot = body_rot[:, 0, :]

        root_h = root_pos[:, 2:3]
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        if (not root_height_obs):
            root_h_obs = torch.zeros_like(root_h)
        else:
            root_h_obs = root_h

        len_keypos = len(key_body_ids)
        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_2 = heading_rot_expand.repeat((1, len_keypos, 1))
        flat_heading_rot_2 = heading_rot_expand_2.reshape(heading_rot_expand_2.shape[0] * heading_rot_expand_2.shape[1], 
                                                heading_rot_expand_2.shape[2])
        
        heading_rot_expand = heading_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_rot = heading_rot_expand.reshape(heading_rot_expand.shape[0] * heading_rot_expand.shape[1], 
                                                heading_rot_expand.shape[2])

        heading_rot_expand = heading_rot.unsqueeze(-2)
        heading_rot_expand_no_hand = heading_rot_expand.repeat((1, 22, 1))
        flat_heading_rot_no_hand = heading_rot_expand_no_hand.reshape(heading_rot_expand_no_hand.shape[0] * heading_rot_expand_no_hand.shape[1], 
                                                heading_rot_expand_no_hand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand = heading_inv_rot_expand.repeat((1, body_pos.shape[1], 1))
        flat_heading_inv_rot = heading_inv_rot_expand.reshape(heading_inv_rot_expand.shape[0] * heading_inv_rot_expand.shape[1], 
                                                heading_inv_rot_expand.shape[2])

        heading_inv_rot_expand = heading_inv_rot.unsqueeze(-2)
        heading_inv_rot_expand_no_hand = heading_inv_rot_expand.repeat((1, 22, 1))
        flat_heading_inv_rot_no_hand = heading_inv_rot_expand_no_hand.reshape(heading_inv_rot_expand_no_hand.shape[0] * heading_inv_rot_expand_no_hand.shape[1], 
                                                heading_inv_rot_expand_no_hand.shape[2])
        
        _ref_body_pos = self.extract_data_component('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_pos = body_pos[:, key_body_ids, :]

        diff_global_body_pos = _ref_body_pos - _body_pos
        diff_local_body_pos_flat = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
        
        local_ref_body_pos = _body_pos - root_pos.unsqueeze(1)  # preserves the body position
        local_ref_body_pos = torch_utils.quat_rotate(flat_heading_rot_2, local_ref_body_pos.view(-1, 3)).view(-1, len_keypos * 3)
    
        root_pos_expand = root_pos.unsqueeze(-2)
        local_body_pos = body_pos - root_pos_expand
        flat_local_body_pos = local_body_pos.reshape(local_body_pos.shape[0] * local_body_pos.shape[1], local_body_pos.shape[2])
        flat_local_body_pos = quat_rotate(flat_heading_rot, flat_local_body_pos)
        local_body_pos = flat_local_body_pos.reshape(local_body_pos.shape[0], local_body_pos.shape[1] * local_body_pos.shape[2])
        local_body_pos = local_body_pos[..., 3:] # remove root pos

        flat_body_rot = body_rot.reshape(body_rot.shape[0] * body_rot.shape[1], body_rot.shape[2])
        flat_local_body_rot = quat_mul(flat_heading_rot, flat_body_rot)
        flat_local_body_rot_obs = torch_utils.quat_to_tan_norm(flat_local_body_rot)
        local_body_rot_obs = flat_local_body_rot_obs.reshape(body_rot.shape[0], body_rot.shape[1] * flat_local_body_rot_obs.shape[1])
        
        ref_body_rot = self.extract_data_component('body_rot', obs=ref_obs)
        ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1) 
        body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
        diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
        diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
        diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
        diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

        local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
        local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

        ref_body_vel = self.extract_data_component('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
        _body_vel = body_vel[:, key_body_ids, :]
        diff_global_vel = ref_body_vel - _body_vel
        diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

        ref_body_ang_vel = self.extract_data_component('body_rot_vel', obs=ref_obs)
        ref_body_ang_vel_no_hand = torch.cat((ref_body_ang_vel[:, :18*3], ref_body_ang_vel[:, 33*3:37*3]), dim=-1)
        body_ang_vel_no_hand = torch.cat((body_ang_vel[:, :18], body_ang_vel[:, 33:37]), dim=1)
        diff_global_ang_vel = ref_body_ang_vel_no_hand.view(-1, 22, 3) - body_ang_vel_no_hand
        diff_local_ang_vel = torch_utils.quat_rotate(flat_heading_rot_no_hand, diff_global_ang_vel.view(-1, 3)).view(-1, 22 * 3)

        if (local_root_obs):
            root_rot_obs = torch_utils.quat_to_tan_norm(root_rot)
            local_body_rot_obs[..., 0:6] = root_rot_obs

        flat_body_vel = body_vel.reshape(body_vel.shape[0] * body_vel.shape[1], body_vel.shape[2])
        flat_local_body_vel = quat_rotate(flat_heading_rot, flat_body_vel)
        local_body_vel = flat_local_body_vel.reshape(body_vel.shape[0], body_vel.shape[1] * body_vel.shape[2])
        
        flat_body_ang_vel = body_ang_vel.reshape(body_ang_vel.shape[0] * body_ang_vel.shape[1], body_ang_vel.shape[2])
        flat_local_body_ang_vel = quat_rotate(flat_heading_rot, flat_body_ang_vel)
        local_body_ang_vel = flat_local_body_ang_vel.reshape(body_ang_vel.shape[0], body_ang_vel.shape[1] * body_ang_vel.shape[2])

        body_contact_buf = contact_forces[:, contact_body_ids, :].clone() #.view(contact_forces.shape[0],-1)
        contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
        ref_body_contact = self.extract_data_component('contact_human', obs=ref_obs)[:, contact_body_ids]
        diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

        obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact, diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos, local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
        return obs
    
    def compute_obj_observations(self, root_states, tar_states, ref_obs):
        root_pos = root_states[:, 0:3]
        root_rot = root_states[:, 3:7]

        tar_pos = tar_states[:, 0:3]
        tar_rot = tar_states[:, 3:7]
        tar_vel = tar_states[:, 7:10]
        tar_ang_vel = tar_states[:, 10:13]

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_inv_rot = torch_utils.calc_heading_quat(root_rot)

        local_tar_pos = tar_pos - root_pos
        local_tar_pos[..., -1] = tar_pos[..., -1]
        local_tar_pos = quat_rotate(heading_rot, local_tar_pos)
        local_tar_vel = quat_rotate(heading_rot, tar_vel)
        local_tar_ang_vel = quat_rotate(heading_rot, tar_ang_vel)

        local_tar_rot = quat_mul(heading_rot, tar_rot)
        local_tar_rot_obs = torch_utils.quat_to_tan_norm(local_tar_rot)

        _ref_obj_pos = self.extract_data_component('obj_pos', obs=ref_obs)
        diff_global_obj_pos = _ref_obj_pos - tar_pos
        diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

        local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
        local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

        ref_obj_rot = self.extract_data_component('obj_rot', obs=ref_obs)
        diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
        diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)  # Need to be change of basis
        diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

        local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
        local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

        ref_obj_vel = self.extract_data_component('obj_pos_vel', obs=ref_obs)
        diff_global_vel = ref_obj_vel - tar_vel
        diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

        ref_obj_ang_vel = self.extract_data_component('obj_rot_vel', obs=ref_obs)
        diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
        diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

        obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs, diff_local_vel, diff_local_ang_vel], dim=-1)
        return obs
    
    def _compute_observations_iter(self, hoi_data, env_ids=None, delta_t=1):
        if (env_ids is None):
            env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)

        ts = self.progress_buf[env_ids].clone() 
        next_ts = torch.clamp(ts + delta_t, max=self.max_episode_length[self.data_id[env_ids]]-1)
        ref_obs = hoi_data[self.data_id[env_ids], next_ts].clone()
        obs = self._compute_humanoid_obs(env_ids, ref_obs, next_ts)
        task_obs = self._compute_task_obs(env_ids, ref_obs)
        obs = torch.cat([obs, task_obs], dim=-1)    
        ig_all, ig, ref_ig = self._compute_ig_obs(env_ids, ref_obs)
        return torch.cat((obs,ig_all,ref_ig-ig),dim=-1)
        
    def _compute_ig_obs(self, env_ids, ref_obs):
        ig = self.extract_data_component('ig', obs=self._curr_obs[env_ids]).view(env_ids.shape[0], -1, 3)
        ig_norm = ig.norm(dim=-1, keepdim=True)
        ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
        ig = ig_all[:, self._key_body_ids, :].view(env_ids.shape[0], -1)
        ig_all = ig_all.view(env_ids.shape[0], -1)    
        ref_ig = self.extract_data_component('ig', obs=ref_obs)
        ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, self._key_body_ids, :]
        ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
        ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()  
        ref_ig = ref_ig.view(env_ids.shape[0], -1)
        return ig_all, ig, ref_ig
        
    def _compute_observations(self, env_ids=None):
        if (env_ids is None):
            self._curr_ref_obs[:] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[:] = torch.cat((self._compute_observations_iter(self.hoi_data, None, 1), self._compute_observations_iter(self.hoi_data, None, 16)), dim=-1)

        else:
            self._curr_ref_obs[env_ids] = self.hoi_data[self.data_id[env_ids], self.progress_buf[env_ids]].clone()
            self.obs_buf[env_ids] = torch.cat((self._compute_observations_iter(self.hoi_data, env_ids, 1), self._compute_observations_iter(self.hoi_data, env_ids, 16)), dim=-1)
            
        return
    
    def _compute_hoi_observations(self, env_ids=None):
        self._curr_obs[:] = self.build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                        self._rigid_body_rot[:, 0, :],
                                                        self._rigid_body_vel[:, 0, :],
                                                        self._rigid_body_ang_vel[:, 0, :],
                                                        self._dof_pos, self._dof_vel, self._rigid_body_pos,
                                                        self._local_root_obs, self._root_height_obs, 
                                                        self._dof_obs_size, self._target_states,
                                                        self._tar_contact_forces,
                                                        self._contact_forces,
                                                        self.object_points[self.object_id[self.data_id]],
                                                        self._rigid_body_rot,
                                                        self._rigid_body_vel,
                                                        self._rigid_body_ang_vel
                                                        )
        return

    def build_hoi_observations(self, root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos, 
                            local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):

        contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
        target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

        tar_pos = target_states[:, 0:3]
        tar_rot = target_states[:, 3:7]
        obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
        ig = compute_sdf(body_pos, obj_points).view(-1, 3)
        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
        ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)    
        
        obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel, 
                         body_pos.reshape(body_pos.shape[0],-1), body_rot.reshape(body_rot.shape[0],-1), body_vel.reshape(body_vel.shape[0],-1), body_rot_vel.reshape(body_rot_vel.shape[0],-1),
                         target_states, ig, contact, target_contact), dim=-1)
        return obs
    
    def _compute_reset(self):
        self.reset_buf[:], self._terminate_buf[:] = self.compute_hoi_reset(self.reset_buf, self.progress_buf, self.obs_buf,
                                                                           self._rigid_body_pos, self.max_episode_length[self.data_id],
                                                                           self._enable_early_termination, self._termination_heights, self.start_times, 
                                                                           self.rollout_length, self.kinematic_reset, torch.any(self.contact_reset > 10, dim=-1)
                                                                          )
        if self.reset_buf.sum() > 0 and self.psi > 1:
            reset_ind = (self.reset_buf == 1)
            data_id = self.data_id[reset_ind]
            max_episode_length = self.max_episode_length[data_id]
            if (max_episode_length < self.rollout_length).all():
                self._sum_reward[reset_ind] = 0
                return
            start_index, end_index = self.start_times[reset_ind], self.progress_buf[reset_ind]
            sum_reward = self._sum_reward[reset_ind].mean()
            if torch.rand(1)[0] < 0:
                self._sum_reward[reset_ind] = 0
                return
            self._sum_reward[reset_ind] = 0
            reset_ind = torch.logical_and(reset_ind, self.max_episode_length[self.data_id] > self.rollout_length)
            if reset_ind.sum() < 0.995:
                return
            curr_reward = self._curr_reward[reset_ind]
            state = self._curr_state[reset_ind]
            # Initialize the reward tensor with zeros
            reward = torch.zeros((curr_reward.shape[0], self.hoi_refs.shape[0], self.hoi_refs.shape[2]), device=curr_reward.device)
            end_i = torch.minimum(max_episode_length, self.rollout_length + start_index)

            assert (end_index < end_i).all()
            # Loop through each example in the batch to assign the values from curr_reward to the correct slices in reward

            # data_num, sample_choice, time, feature

            for i in range(curr_reward.shape[0]):
                if end_index[i] > start_index[i]+30:  # Ensure the indices are valid
                    index_tensor = torch.arange(start_index[i]+10, end_index[i]-10, device=start_index.device)
                    reward[i, data_id[i], start_index[i]+10:end_index[i]-10] = ((end_index[i] - index_tensor) / (end_i[i] - index_tensor))

            adjust_reward, adjust_reward_index = reward.max(dim=0)
            for i in range(reward.shape[1]):
                if self.max_episode_length[i] < self.rollout_length:
                    continue
                for j in range(reward.shape[2]):
                    if self.max_episode_length[i] - j < self.rollout_length:
                        break
                    value, index = self.ref_reward[i, 1:, j].min(dim=0)
                    index = index + 1
                    id1 = adjust_reward_index[i, j]
                    idx = j - start_index[adjust_reward_index[i, j]]

                    if idx > 0 and idx < self.rollout_length and adjust_reward[i, j] > 0.5:
                        self.ref_reward[i, index, j] = adjust_reward[i, j]
                        self.hoi_refs[i, index, j] = state[id1, idx]
            self.ref_reward[:, 1:, :] = self.ref_reward[:, 1:, :] * (1 - 1e-5)
        return

    def compute_hoi_reset(self, reset_buf, progress_buf, obs_buf, rigid_body_pos,
                          max_episode_length, enable_early_termination, termination_heights, 
                          start_times, rollout_length, reset_ig, contact_reset):

        reset, terminated = self.compute_humanoid_reset(reset_buf, progress_buf, obs_buf, rigid_body_pos,
                                                        max_episode_length, enable_early_termination, termination_heights, 
                                                        start_times, rollout_length)

        reset_ig *= (progress_buf > 1 + start_times)
        contact_reset *= (progress_buf > 1 + start_times)
                
        terminated = torch.where(torch.logical_or(reset_ig, contact_reset), torch.ones_like(reset_buf), terminated)
        reset = torch.where(reset.bool(), torch.ones_like(reset_buf), terminated)

        return reset, terminated

    def _compute_reward(self, actions):
        rb, human_reset, key_pos, ref_key_pos = self.compute_humanoid_reward(self.reward_weights)
        ro, object_reset, obj_points, ref_obj_points = self.compute_obj_reward(self.reward_weights)
        rig, ig_reset = self.compute_ig_reward(self.reward_weights, key_pos, ref_key_pos, obj_points, ref_obj_points)
        rcg, contact_reset = self.compute_cg_reward(self.reward_weights)
        self.rew_buf[:] = rb * ro * rig * rcg
        kinematic_reset = torch.logical_or(human_reset, object_reset)
        self.contact_reset = (self.contact_reset + contact_reset) * contact_reset
        self.kinematic_reset = torch.logical_or(ig_reset, kinematic_reset)
        index = torch.arange(self._curr_reward.shape[0])
        # # print(self._humanoid_root_states.dtype)
        self._curr_reward[index, self.progress_buf - self.start_times] = self.rew_buf
        self._sum_reward[index] += self.rew_buf
        self._curr_state[index, self.progress_buf - self.start_times, :] = torch.cat([
            self._humanoid_root_states,
            self._dof_pos,
            self._dof_vel,
            self._target_states,
        ], dim=1)
        return
    
    def compute_humanoid_reward(self, w):
        # body pos reward
        len_keypos = len(self._key_body_ids)
        key_pos = self.extract_data_component('body_pos', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_key_pos = self.extract_data_component('body_pos', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)[:, self._key_body_ids]
        
        ref_ig = self.extract_data_component('ig', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 3)
        ref_ig_norm = ref_ig.norm(dim=-1)
        weight_h = (-5 * ref_ig_norm).exp()
        weight_hp = weight_h.clone().detach()  
        ancle_toe_ids = [i+1 for i in range(len_keypos) if 'Ankle' in self.key_bodies[i] or 'Toe' in self.key_bodies[i]]
        weight_hp[:, ancle_toe_ids] = 1

        ep = torch.mean(((ref_key_pos - key_pos)**2).sum(dim=-1) * weight_hp[:, self._key_body_ids],dim=-1)
        rp = torch.exp(-ep*w['p'])

        body_rot = self.extract_data_component('body_rot', obs=self._curr_obs).view(self._curr_obs.shape[0], -1, 4)
        ref_body_rot = self.extract_data_component('body_rot', obs=self._curr_ref_obs).view(self._curr_ref_obs.shape[0], -1, 4)
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot.reshape(-1, 4)), body_rot.reshape(-1, 4))
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 52)
        weight_hr = 1 - weight_h
        
        er = torch.mean(diff[:, :] * weight_hr, dim=-1)
        rr = torch.exp(-er*w['r'])
        
        body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_obs)
        ref_body_pos_vel = self.extract_data_component('body_pos_vel', obs=self._curr_ref_obs)
        # body pos vel reward
        epv = torch.mean((ref_body_pos_vel - body_pos_vel)**2,dim=-1)
        # epv = torch.mean(pos_vel ,dim=-1) # torch.zeros_like(ep)
        rpv = torch.exp(-epv*w['pv'])

        dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_obs)
        ref_dof_pos_vel = self.extract_data_component('body_rot_vel', obs=self._curr_ref_obs)
        # body rot vel reward
        erv = torch.mean((ref_dof_pos_vel - dof_pos_vel)**2,dim=-1)
        rrv = torch.exp(-erv*w['rv'])

        # energy penalty
        hist_dof_vel = self.extract_data_component('dof_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('dof_vel', obs=self._curr_obs) - hist_dof_vel)*self.fps_data
        dof_diffacc = (local_vel.view(-1, 51*3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)).clone()
        energy = dof_diffacc.pow(2).mean(dim=-1).mul(-w['eg1']).exp()

        rb = rp*rr*rpv*rrv*energy
        human_reset = (ref_key_pos - key_pos).norm(dim=-1).mean(dim=-1) > 0.5
        
        return rb, human_reset, key_pos, ref_key_pos
    
    def compute_obj_reward(self, w):
        # object pos reward
        root_pos = self.extract_data_component('root_pos', obs=self._curr_obs)
        root_rot = self.extract_data_component('root_rot', obs=self._curr_obs)

        heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
        
        obj_pos = self.extract_data_component('obj_pos', obs=self._curr_obs)
        obj_rot = self.extract_data_component('obj_rot', obs=self._curr_obs)
        local_obj_pos = obj_pos - root_pos
        local_obj_pos[..., -1] = obj_pos[..., -1]
        local_obj_pos = quat_rotate(heading_rot, local_obj_pos)

        local_obj_rot = quat_mul(heading_rot, obj_rot)

        object_points = self.object_points[self.object_id[self.data_id]]
        obj_rot_extend = obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        object_points_extend = object_points.view(-1, 3)
        obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + obj_pos.unsqueeze(1)

        ref_root_pos = self.extract_data_component('root_pos', obs=self._curr_ref_obs)
        ref_root_rot = self.extract_data_component('root_rot', obs=self._curr_ref_obs)

        ref_heading_rot = torch_utils.calc_heading_quat_inv(ref_root_rot)

        ref_obj_pos = self.extract_data_component('obj_pos', obs=self._curr_ref_obs)
        ref_obj_rot = self.extract_data_component('obj_rot', obs=self._curr_ref_obs)

        ref_local_obj_pos = ref_obj_pos - ref_root_pos
        ref_local_obj_pos[..., -1] = ref_obj_pos[..., -1]
        ref_local_obj_pos = quat_rotate(ref_heading_rot, ref_local_obj_pos)

        ref_local_obj_rot = quat_mul(ref_heading_rot, ref_obj_rot)

        ref_obj_rot_extend = ref_obj_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
        ref_obj_points = torch_utils.quat_rotate(ref_obj_rot_extend, object_points_extend).view(obj_rot.shape[0], object_points.shape[1], 3) + ref_obj_pos.unsqueeze(1)

        eop = torch.mean(((ref_local_obj_pos - local_obj_pos)**2),dim=-1) # * (1 - weight_h.max(dim=-1)[0])
        rop = torch.exp(-eop*w['op'])

        # object rot reward
        diff_quat_data = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_local_obj_rot), local_obj_rot)
        diff_angle, diff_axis = torch_utils.quat_to_angle_axis(diff_quat_data)
        diff = diff_angle.view(-1, 1)
        
        eor = torch.mean(diff,dim=-1)
        ror = torch.exp(-eor*w['or'])

        obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_obs)
        ref_obj_pos_vel = self.extract_data_component('obj_pos_vel', obs=self._curr_ref_obs)
        # object pos vel reward
        eopv = torch.mean((ref_obj_pos_vel - obj_pos_vel)**2,dim=-1)
        ropv = torch.exp(-eopv*w['opv'])

        obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_obs)
        ref_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._curr_ref_obs)
        # object rot vel reward
        eorv = torch.mean((ref_obj_rot_vel - obj_rot_vel)**2,dim=-1)
        rorv = torch.exp(-eorv*w['orv'])
        
        hist_obj_vel = self.extract_data_component('obj_pos_vel', obs=self._hist_obs)
        obj_diffacc = (self.extract_data_component('obj_pos_vel', obs=self._curr_obs) - hist_obj_vel)*self.fps_data
        obj_diffacc = obj_diffacc*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)

        hist_obj_rot_vel = self.extract_data_component('obj_rot_vel', obs=self._hist_obs)
        local_vel = (self.extract_data_component('obj_rot_vel', obs=self._curr_obs) - hist_obj_rot_vel)*self.fps_data
        obj_rot_diffacc = local_vel.view(-1, 3)*(self.progress_buf-self.start_times>2).float().unsqueeze(dim=-1)
        
        obj_energy = (obj_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp()) * (obj_rot_diffacc.pow(2).mean(dim=-1).mul(-w['eg2']).exp())
        ro = rop*ror*ropv*rorv*obj_energy
        object_reset = (obj_points - ref_obj_points).norm(dim=-1).mean(dim=-1) > 0.5
        return ro, object_reset, obj_points, ref_obj_points
    
    def compute_ig_reward(self, w, key_pos, ref_key_pos, obj_points, ref_obj_points):
        len_keypos = len(self._key_body_ids)
        ig = key_pos.view(-1,len_keypos,3).unsqueeze(2) - obj_points.unsqueeze(1)
        ref_ig = ref_key_pos.view(-1,len_keypos,3).unsqueeze(2) - ref_obj_points.unsqueeze(1)
        ### interaction graph reward ###
        weight_1 = (1 / torch.clamp((ig**2).sum(dim=-1), min=0.01))
        weight_1 = weight_1 / weight_1.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)
        weight_2 = (1 / torch.clamp((ref_ig**2).sum(dim=-1), min=0.01))
        weight_2 = weight_2 / weight_2.sum(dim=-1, keepdim=True).sum(dim=-2, keepdim=True)

        eig = ((ig - ref_ig)**2).sum(dim=-1) * (weight_1 + weight_2)  

        rig = torch.exp(-w['ig'] * (eig.sum(dim=-1).sum(dim=-1) * 0.5))

        reset_ig_1 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ref_ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig_2 = (((ig - ref_ig)**2).sum(dim=-1).sqrt() / torch.clamp((ig**2).sum(dim=-1).sqrt(), min=0.5)).max(dim=-1)[0].max(dim=-1)[0] > 2
        reset_ig = torch.logical_or(reset_ig_1, reset_ig_2)
        return rig, reset_ig
    
    def compute_cg_reward(self, w):    
        contact_thres = 0.1
        ref_human_contact = self.extract_data_component('contact_human', obs=self._curr_ref_obs)
        human_contact = self.extract_data_component('contact_human', obs=self._curr_obs)
        left_contact_hand_ids = list(range(17, 33))
        
        ref_left_contact_hand = ref_human_contact[:, left_contact_hand_ids]
        ref_left_contact_hand_any = torch.any(ref_left_contact_hand > contact_thres, dim=-1).float()
        left_hand_contact = human_contact[:, left_contact_hand_ids].clone()
        left_hand_contact_any = torch.any(left_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        ecg_left = (((ref_left_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(left_hand_contact - ref_left_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_left = 0.5 * (1 + torch.exp(-ecg_left*w['cg_hand'])) * (ref_left_contact_hand_any) + (1 - ref_left_contact_hand_any)


        right_contact_hand_ids = list(range(36, 52))
        
        ref_right_contact_hand = ref_human_contact[:, right_contact_hand_ids]
        ref_right_contact_hand_any = torch.any(ref_right_contact_hand > contact_thres, dim=-1).float()
        right_hand_contact = human_contact[:, right_contact_hand_ids].clone()
        right_hand_contact_any = torch.any(right_hand_contact > contact_thres, dim=-1, keepdim=True).float()

        contact_reset = torch.cat([ 
                                torch.abs(ref_left_contact_hand_any.unsqueeze(-1) - left_hand_contact_any) * ref_left_contact_hand_any.unsqueeze(-1), 
                                torch.abs(ref_right_contact_hand_any.unsqueeze(-1) - right_hand_contact_any) * ref_right_contact_hand_any.unsqueeze(-1),
                                ], dim=-1)
        
        ecg_right = (((ref_right_contact_hand_any.unsqueeze(-1) > contact_thres) * torch.abs(right_hand_contact - ref_right_contact_hand_any.unsqueeze(-1))).mean(dim=-1))
        rcg_right = 0.5 * (1 + torch.exp(-ecg_right*w['cg_hand'])) * (ref_right_contact_hand_any) + (1 - ref_right_contact_hand_any)
        
        rcg_hand = rcg_left * rcg_right

        other_ids = [i for i in range(len(self.contact_bodies)) if i not in left_contact_hand_ids and i not in right_contact_hand_ids]
        ref_other_contact = ref_human_contact[:, other_ids]
        other_contact = human_contact[:, other_ids]
        ecg_other = ((torch.abs(other_contact - ref_other_contact) * (ref_other_contact > contact_thres))).mean(dim=-1)
        rcg_other = torch.exp(-ecg_other*w['cg_other'])
        
        no_contact = torch.abs(human_contact) < contact_thres
        ecg_all = (torch.abs(no_contact + ref_human_contact) * (ref_human_contact < -contact_thres)).mean(dim=-1)
        rcg_all = torch.exp(-ecg_all*w['cg_all'])

        contact_all = self._contact_forces.clone().abs().sum(dim=-1).sum(dim=-1)
        contact_energy = contact_all.pow(2).mul(-w['eg3']).exp()

        rcg = rcg_hand*rcg_other*rcg_all*contact_energy
        return rcg, contact_reset
    
    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        t = to_torch(
                [
                    t if t < self.max_episode_length[self.data_id[i]] else self.max_episode_length[self.data_id[i]]-1
                    for i in range(self.num_envs)
                ],
                device=self.device,
                dtype=torch.long
            )
        ### update object ###
        self._target_states[env_ids, :3] = self.extract_data_component('obj_pos', True, self.data_id[env_ids], t)
        self._target_states[env_ids, 3:7] = self.extract_data_component('obj_rot', True, self.data_id[env_ids], t)
        self._target_states[env_ids, 7:10] = torch.zeros_like(self._target_states[env_ids, 7:10])
        self._target_states[env_ids, 10:13] = torch.zeros_like(self._target_states[env_ids, 10:13])

        ### update subject ###   
        _humanoid_root_pos = self.extract_data_component('root_pos', True, self.data_id[env_ids], t)
        _humanoid_root_rot = self.extract_data_component('root_rot', True, self.data_id[env_ids], t)
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])
        
        self._dof_pos[env_ids] = self.extract_data_component('dof_pos', True, self.data_id[env_ids], t)
        self._dof_vel[env_ids] = self.extract_data_component('dof_vel', True, self.data_id[env_ids], t)


        env_ids_int32 = self._humanoid_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim,
                                                     gymtorch.unwrap_tensor(self._root_states),
                                                     gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))
        
        env_ids_int32 = self._tar_actor_ids[env_ids]
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(self._root_states),
                                                    gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))

        self._refresh_sim_tensors()
        obj_contact = self.extract_data_component('contact_obj', True, self.data_id[env_ids], t)
        obj_contact = torch.any(obj_contact > 0.1, dim=-1)
        human_contact = self.extract_data_component('contact_human', True, self.data_id[env_ids], t)
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]

                if obj_contact[env_id] == True:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                else:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(0., 0., 1.))
                    
                handle = self.humanoid_handles[env_id]
                for j in range(self.num_bodies):
                    if human_contact[env_id, j] > 0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(1., 0., 0.))
                    elif human_contact[env_id, j] > -0.5:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 1., 0.))
                    else:
                        self.gym.set_rigid_body_color(env_ptr, handle, j, gymapi.MESH_VISUAL,
                                                    gymapi.Vec3(0., 0., 1.))
        self.render(t=t)
        self.gym.simulate(self.sim)

        return
    

    def render(self, sync_frame_time=False, t=0):
        super().render(sync_frame_time)

        if self.viewer:  
            if self.save_images:
                env_ids = 0
                if self.play_dataset:
                    frame_id = t
                else:
                    frame_id = self.progress_buf[env_ids]
                dataname = self.motion_file[-1][6:-3]
                rgb_filename = "intermimic/data/images/" + dataname + "/rgb_env%d_frame%05d.png" % (env_ids, frame_id)
                os.makedirs("intermimic/data/images/" + dataname, exist_ok=True)
                self.gym.write_viewer_image_to_file(self.viewer,rgb_filename)
        return
    
@torch.jit.script
def compute_sdf(points1, points2):
    # type: (Tensor, Tensor) -> Tensor
    dis_mat = points1.unsqueeze(2) - points2.unsqueeze(1)
    dis_mat_lengths = torch.norm(dis_mat, dim=-1)
    min_length_indices = torch.argmin(dis_mat_lengths, dim=-1)
    B_indices, N_indices = torch.meshgrid(torch.arange(points1.shape[0]), torch.arange(points1.shape[1]), indexing='ij')
    min_dis_mat = dis_mat[B_indices, N_indices, min_length_indices].contiguous()
    return min_dis_mat