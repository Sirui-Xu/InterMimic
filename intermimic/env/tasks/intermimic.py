from enum import Enum
import numpy as np
import torch
import os

from isaacgym import gymtorch
from isaacgym import gymapi
from isaacgym.torch_utils import *
import torch.utils

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
        motion_file = os.listdir(self.motion_file)
        self.motion_file = sorted([os.path.join(self.motion_file, data_path) for data_path in motion_file if data_path.split('_')[0] in cfg['env']['dataSub']])
        self.object_name = [motion_example.split('_')[-2] for motion_example in self.motion_file]
        object_name_set = sorted(list(set(self.object_name)))
        print(self.motion_file, object_name_set)
        self.object_id = to_torch([object_name_set.index(name) for name in self.object_name], dtype=torch.long).cuda()
        self.obj2motion = torch.stack([self.object_id == k for k in range(len(object_name_set))], dim=0)
        self.object_name = object_name_set
        self.robot_type = cfg['env']['robotType']
        self.object_density = cfg['env']['objectDensity']

        print(self.robot_type)
        self.num_motions = len(self.motion_file)
        super().__init__(cfg=cfg,
                         sim_params=sim_params,
                         physics_engine=physics_engine,
                         device_type=device_type,
                         device_id=device_id,
                         headless=headless)
        
        self._load_motion(self.motion_file)

        self._curr_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_ref_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._curr_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._hist_obs = torch.zeros((self.num_envs, self.ref_hoi_obs_size), device=self.device, dtype=torch.float)
        self._tar_pos = torch.zeros([self.num_envs, 3], device=self.device, dtype=torch.float)
        self._reset_ig = torch.zeros([self.num_envs], device=self.device, dtype=torch.bool)
        self._build_target_tensors()

        return

    def _compute_reward(self, actions):
        super()._compute_reward(actions)
        return

    def _compute_reset(self):
        super()._compute_reset()


    def post_physics_step(self):

        super().post_physics_step()
        env_ids = to_torch(np.arange(self.num_envs), device=self.device, dtype=torch.long)
        self._update_hist_hoi_obs()
        self._compute_hoi_observations(env_ids)

        return

    def _update_hist_hoi_obs(self, env_ids=None):
        self._hist_obs = self._curr_obs.clone()
        return
        
    def _setup_character_props(self, key_bodies):
        super()._setup_character_props(key_bodies)
        return


    def _load_motion(self, motion_file):

        self.hoi_data_dict = []
        hoi_datas = []
        hoi_refs = []
        if type(motion_file) != type([]):
            motion_file = [motion_file]
        self.max_episode_length = []
        for idx, data_path in enumerate(motion_file):
            loaded_dict = {}
            hoi_data = torch.load(data_path)
            loaded_dict['hoi_data'] = hoi_data.detach().to('cuda')

        
            self.max_episode_length.append(loaded_dict['hoi_data'].shape[0])
            self.fps_data = 30.

            loaded_dict['root_pos'] = loaded_dict['hoi_data'][:, 0:3].clone()
            loaded_dict['root_pos_vel'] = (loaded_dict['root_pos'][1:,:].clone() - loaded_dict['root_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['root_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_pos_vel'].shape[-1])).to('cuda'),loaded_dict['root_pos_vel']),dim=0)

            loaded_dict['root_rot'] = loaded_dict['hoi_data'][:, 3:7].clone()
            root_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['root_rot'])
            loaded_dict['root_rot_vel'] = (root_rot_exp_map[1:,:].clone() - root_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['root_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['root_rot_vel'].shape[-1])).to('cuda'),loaded_dict['root_rot_vel']),dim=0)

            loaded_dict['dof_pos'] = loaded_dict['hoi_data'][:, 9:9+153].clone()

            loaded_dict['dof_pos_vel'] = []

            loaded_dict['dof_pos_vel'] = (loaded_dict['dof_pos'][1:,:].clone() - loaded_dict['dof_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['dof_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['dof_pos_vel'].shape[-1])).to('cuda'),loaded_dict['dof_pos_vel']),dim=0)

            loaded_dict['body_pos'] = loaded_dict['hoi_data'][:, 162: 162+52*3].clone().view(self.max_episode_length[-1],52,3)
            loaded_dict['key_body_pos'] = loaded_dict['body_pos'][:, self._key_body_ids, :].view(self.max_episode_length[-1],-1).clone()
            loaded_dict['key_body_pos_vel'] = (loaded_dict['key_body_pos'][1:,:].clone() - loaded_dict['key_body_pos'][:-1,:].clone())*self.fps_data
            loaded_dict['key_body_pos_vel'] = torch.cat((torch.zeros((1, loaded_dict['key_body_pos_vel'].shape[-1])).to('cuda'),loaded_dict['key_body_pos_vel']),dim=0)

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
            key_body_pose = loaded_dict['key_body_pos'][:,:].clone()
            ref_ig = compute_sdf(key_body_pose.view(loaded_dict['obj_rot'].shape[0],-1,3), obj_points).view(-1, 3)
            heading_rot = torch_utils.calc_heading_quat_inv(loaded_dict['root_rot'])
            heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pose.shape[1] // 3, 1).view(-1, 4)
            ref_ig = quat_rotate(heading_rot_extend, ref_ig).view(loaded_dict['obj_rot'].shape[0], -1)    
            loaded_dict['contact'] = torch.round(loaded_dict['hoi_data'][:, 330:331].clone())
            loaded_dict['contact_parts'] = torch.round(loaded_dict['hoi_data'][:, 331:331+52].clone())
            loaded_dict['human_rot'] = loaded_dict['hoi_data'][:, 331+52:331+52+52*4].clone()

            human_rot_exp_map = torch_utils.quat_to_exp_map(loaded_dict['human_rot'].view(-1, 4)).view(-1, 52*3)
            loaded_dict['human_rot_vel'] = (human_rot_exp_map[1:,:].clone() - human_rot_exp_map[:-1,:].clone())*self.fps_data
            loaded_dict['human_rot_vel'] = torch.cat((torch.zeros((1, loaded_dict['human_rot_vel'].shape[-1])).to('cuda'),loaded_dict['human_rot_vel']),dim=0)

            loaded_dict['hoi_data'] = torch.cat((
                                                    loaded_dict['root_pos'].clone(), 
                                                    loaded_dict['root_rot'].clone(), 
                                                    loaded_dict['dof_pos'].clone(), 
                                                    loaded_dict['dof_pos_vel'].clone(),
                                                    loaded_dict['obj_pos'].clone(),
                                                    loaded_dict['obj_rot'].clone(),
                                                    loaded_dict['obj_pos_vel'].clone(), 
                                                    loaded_dict['obj_rot_vel'].clone(),
                                                    loaded_dict['key_body_pos'][:,:].clone(),
                                                    loaded_dict['contact'].clone(),
                                                    loaded_dict['contact_parts'].clone(),
                                                    ref_ig.clone(),
                                                    loaded_dict['human_rot'].clone(),
                                                    loaded_dict['key_body_pos_vel'].clone(),
                                                    loaded_dict['human_rot_vel'],
                                                    ),dim=-1)

            assert(self.ref_hoi_obs_size == loaded_dict['hoi_data'].shape[-1])
            self.hoi_data_dict.append(loaded_dict)
            hoi_datas.append(loaded_dict['hoi_data'])

            hoi_ref = torch.cat((
                                loaded_dict['root_pos'].clone(), 
                                loaded_dict['root_rot'].clone(), 
                                loaded_dict['dof_pos'].clone(), 
                                loaded_dict['dof_pos_vel'].clone(), 
                                loaded_dict['root_pos_vel'].clone(),
                                loaded_dict['root_rot_vel'].clone(), 
                                loaded_dict['obj_pos'].clone(),
                                loaded_dict['obj_rot'].clone(),
                                loaded_dict['obj_pos_vel'].clone(),
                                loaded_dict['obj_rot_vel'].clone(),
                                ),dim=-1)
            hoi_refs.append(hoi_ref)
        max_length = max(self.max_episode_length)
        self.num_motions = len(hoi_refs)
        self.max_episode_length = to_torch(self.max_episode_length, dtype=torch.long)
        self.hoi_data = []
        self.hoi_refs = []
        for i, data in enumerate(hoi_datas):
            pad_size = (0, 0, 0, max_length - data.size(0))
            padded_data = F.pad(data, pad_size, "constant", 0)
            self.hoi_data.append(padded_data)
            self.hoi_refs.append(F.pad(hoi_refs[i], pad_size, "constant", 0))
        self.hoi_data = torch.stack(self.hoi_data, dim=0)
        self.hoi_refs = torch.stack(self.hoi_refs, dim=0).unsqueeze(1).repeat(1, 3, 1, 1)

        self.ref_reward = torch.zeros((self.hoi_refs.shape[0], self.hoi_refs.shape[1], self.hoi_refs.shape[2])).to(self.hoi_refs.device)
        self.ref_reward[:, 0, :] = 1.0

        self.ref_index = torch.zeros((self.num_envs, )).long().to(self.hoi_refs.device)
        return


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
        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 319:322]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 322:326]
        self._target_states[env_ids, 7:10] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 326:329]
        self._target_states[env_ids, 10:13] = self.hoi_refs[self.data_id[env_ids], self.ref_index[env_ids], self.progress_buf[env_ids], 329:332]
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
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.hoi_refs[i, idx, motion_times, 0:3],
                            root_rot=self.hoi_refs[i, idx, motion_times, 3:7],
                            dof_pos=self.hoi_refs[i, idx, motion_times, 7:160],
                            root_vel=self.hoi_refs[i, idx, motion_times, 313:316],
                            root_ang_vel=self.hoi_refs[i, idx, motion_times, 316:319],
                            dof_vel=self.hoi_refs[i, idx, motion_times, 160:313],
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
        self._hist_obs[env_ids] = 0
        self.contact_reset[env_ids] = 0 
        self._set_env_state(env_ids=env_ids,
                            root_pos=self.hoi_refs[i, idx, motion_times, 0:3],
                            root_rot=self.hoi_refs[i, idx, motion_times, 3:7],
                            dof_pos=self.hoi_refs[i, idx, motion_times, 7:160],
                            root_vel=self.hoi_refs[i, idx, motion_times, 313:316],
                            root_ang_vel=self.hoi_refs[i, idx, motion_times, 316:319],
                            dof_vel=self.hoi_refs[i, idx, motion_times, 160:313],
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


    
    def _compute_hoi_observations(self, env_ids=None):
        key_body_pos = self._rigid_body_pos[:, self._key_body_ids, :]
        key_body_vel = self._rigid_body_vel[:, self._key_body_ids, :]
        if (env_ids is None):
            self._curr_obs[:] = build_hoi_observations(self._rigid_body_pos[:, 0, :],
                                                               self._rigid_body_rot[:, 0, :],
                                                               self._rigid_body_vel[:, 0, :],
                                                               self._rigid_body_ang_vel[:, 0, :],
                                                               self._dof_pos, self._dof_vel, key_body_pos,
                                                               self._local_root_obs, self._root_height_obs, 
                                                               self._dof_obs_size, self._target_states,
                                                               self._tar_contact_forces,
                                                               self._contact_forces,
                                                               self.object_points[self.object_id[self.data_id]],
                                                               self._rigid_body_rot,
                                                               key_body_vel,
                                                               self._rigid_body_ang_vel
                                                               )
        else:
            self._curr_obs[env_ids] = build_hoi_observations(self._rigid_body_pos[env_ids][:, 0, :],
                                                                   self._rigid_body_rot[env_ids][:, 0, :],
                                                                   self._rigid_body_vel[env_ids][:, 0, :],
                                                                   self._rigid_body_ang_vel[env_ids][:, 0, :],
                                                                   self._dof_pos[env_ids], self._dof_vel[env_ids], key_body_pos[env_ids],
                                                                   self._local_root_obs, self._root_height_obs, 
                                                                   self._dof_obs_size, self._target_states[env_ids],
                                                                   self._tar_contact_forces[env_ids],
                                                                   self._contact_forces[env_ids],
                                                                   self.object_points[self.object_id[self.data_id[env_ids]]],
                                                                   self._rigid_body_rot[env_ids],
                                                                   key_body_vel[env_ids],
                                                                   self._rigid_body_ang_vel[env_ids]).float()
        return


    def play_dataset_step(self, time):

        t = time
        if t == 0:
            self.data_id = to_torch([torch.where(self.obj2motion[i % len(self.object_name)] == 1)[0][torch.randint(self.obj2motion[i % len(self.object_name)].sum(), ())] for i in range(self.num_envs)], device=self.device, dtype=torch.long)
        env_ids = to_torch([i for i in range(self.num_envs) if t < self.max_episode_length[self.data_id[i]]], device=self.device, dtype=torch.long)

        ### update object ###
        self._target_states[env_ids, :3] = self.hoi_refs[self.data_id[env_ids], 0, t, 319:322]
        self._target_states[env_ids, 3:7] = self.hoi_refs[self.data_id[env_ids], 0, t, 322:326]
        self._target_states[env_ids, 7:10] = torch.zeros_like(self._target_states[env_ids, 7:10])# self.hoi_refs[self.data_id[env_ids], 0, t, 326:329]
        self._target_states[env_ids, 10:13] = torch.zeros_like(self._target_states[env_ids, 10:13])# self.hoi_refs[self.data_id[env_ids], 0, t, 329:332]

        ### update subject ###   
        _humanoid_root_pos = self.hoi_refs[self.data_id[env_ids], 0, t, 0:3]
        _humanoid_root_rot = self.hoi_refs[self.data_id[env_ids], 0, t, 3:7]
        self._humanoid_root_states[env_ids, 0:3] = _humanoid_root_pos
        self._humanoid_root_states[env_ids, 3:7] = _humanoid_root_rot
        self._humanoid_root_states[:, 7:10] = torch.zeros_like(self._humanoid_root_states[:, 7:10])
        self._humanoid_root_states[:, 10:13] = torch.zeros_like(self._humanoid_root_states[:, 10:13])
        
        self._dof_pos[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 7:160]
        self._dof_vel[env_ids] = self.hoi_refs[self.data_id[env_ids], 0, t, 160:313]


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
        # ### draw contact label ###
        for env_id, env_ptr in enumerate(self.envs):
            if env_id in env_ids:
                contact = self.hoi_data_dict[self.data_id[env_id]]['contact'][t,:]
                obj_contact = torch.any(contact > 0.1, dim=-1)
                env_ptr = self.envs[env_id]
                handle = self._target_handles[env_id]

                if obj_contact == True:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
                                                gymapi.Vec3(1., 0., 0.))
                else:
                    self.gym.set_rigid_body_color(env_ptr, handle, 0, gymapi.MESH_VISUAL,
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



def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, key_body_pos, 
                           local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf, contact_buf, object_points, body_rot, body_vel, body_rot_vel):

    contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
    target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
    ig = compute_sdf(key_body_pos, obj_points).view(-1, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, key_body_pos.shape[1], 1).view(-1, 4)
    ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)    
    
    obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel, target_states, key_body_pos.contiguous().view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), target_contact, contact, ig, body_rot.view(-1, 52*4), body_vel.view(-1,key_body_pos.shape[1]*key_body_pos.shape[2]), body_rot_vel.view(-1, 52*3)), dim=-1)
    return obs
