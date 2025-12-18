"""Observation computation utilities - copied from Isaac Gym InterMimic.

This module contains the exact observation computation logic from Isaac Gym,
with only the sensor data access adapted for IsaacLab.
"""

import torch
import numpy as np
from . import torch_utils_gym as torch_utils

quat_rotate = torch_utils.quat_rotate
quat_mul = torch_utils.quat_mul


def compute_humanoid_observations_max(body_pos, body_rot, body_vel, body_ang_vel, local_root_obs, root_height_obs,
                                     contact_forces, contact_body_ids, ref_obs, key_body_ids, extract_data_component_fn):
    """Compute humanoid observations with reference motion comparison.

    Copied directly from Isaac Gym intermimic.py:499
    """
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

    _ref_body_pos = extract_data_component_fn('body_pos', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
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

    ref_body_rot = extract_data_component_fn('body_rot', obs=ref_obs)
    ref_body_rot_no_hand = torch.cat((ref_body_rot[:, :18*4], ref_body_rot[:, 33*4:37*4]), dim=-1)
    body_rot_no_hand = torch.cat((body_rot[:, :18], body_rot[:, 33:37]), dim=1)
    diff_global_body_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_body_rot_no_hand.reshape(-1, 4)), body_rot_no_hand.reshape(-1, 4))
    diff_local_body_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(flat_heading_rot_no_hand, diff_global_body_rot.view(-1, 4)), flat_heading_inv_rot_no_hand)
    diff_local_body_rot_obs = torch_utils.quat_to_tan_norm(diff_local_body_rot_flat)
    diff_local_body_rot_obs = diff_local_body_rot_obs.view(body_rot_no_hand.shape[0], body_rot_no_hand.shape[1] * diff_local_body_rot_obs.shape[-1])

    local_ref_body_rot = torch_utils.quat_mul(flat_heading_rot_no_hand, ref_body_rot_no_hand.reshape(-1, 4))
    local_ref_body_rot = torch_utils.quat_to_tan_norm(local_ref_body_rot).view(ref_body_rot_no_hand.shape[0], -1)

    ref_body_vel = extract_data_component_fn('body_pos_vel', obs=ref_obs).view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
    _body_vel = body_vel[:, key_body_ids, :]
    diff_global_vel = ref_body_vel - _body_vel
    diff_local_vel = torch_utils.quat_rotate(flat_heading_rot_2, diff_global_vel.view(-1, 3)).view(-1, len_keypos * 3)

    ref_body_ang_vel = extract_data_component_fn('body_rot_vel', obs=ref_obs)
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

    body_contact_buf = contact_forces[:, contact_body_ids, :].clone()
    contact = torch.any(torch.abs(body_contact_buf) > 0.1, dim=-1).float()
    ref_body_contact = extract_data_component_fn('contact_human', obs=ref_obs)[:, contact_body_ids]
    diff_body_contact = ref_body_contact * ((ref_body_contact + 1) / 2 - contact)

    obs = torch.cat((root_h_obs, local_body_pos, local_body_rot_obs, local_body_vel, local_body_ang_vel, contact,
                    diff_local_body_pos_flat, diff_local_body_rot_obs, diff_body_contact, local_ref_body_pos,
                    local_ref_body_rot, diff_local_vel, diff_local_ang_vel), dim=-1)
    return obs


def compute_obj_observations(root_states, tar_states, ref_obs, extract_data_component_fn):
    """Compute object observations with reference comparison.

    Copied directly from Isaac Gym intermimic.py:601
    """
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

    _ref_obj_pos = extract_data_component_fn('obj_pos', obs=ref_obs)
    diff_global_obj_pos = _ref_obj_pos - tar_pos
    diff_local_obj_pos_flat = torch_utils.quat_rotate(heading_rot, diff_global_obj_pos)

    local_ref_obj_pos = _ref_obj_pos - root_pos  # preserves the body position
    local_ref_obj_pos = torch_utils.quat_rotate(heading_rot, local_ref_obj_pos)

    ref_obj_rot = extract_data_component_fn('obj_rot', obs=ref_obs)
    diff_global_obj_rot = torch_utils.quat_mul_norm(torch_utils.quat_inverse(ref_obj_rot), tar_rot)
    diff_local_obj_rot_flat = torch_utils.quat_mul(torch_utils.quat_mul(heading_rot, diff_global_obj_rot.view(-1, 4)), heading_inv_rot)
    diff_local_obj_rot_obs = torch_utils.quat_to_tan_norm(diff_local_obj_rot_flat)

    local_ref_obj_rot = torch_utils.quat_mul(heading_rot, ref_obj_rot)
    local_ref_obj_rot = torch_utils.quat_to_tan_norm(local_ref_obj_rot)

    ref_obj_vel = extract_data_component_fn('obj_pos_vel', obs=ref_obs)
    diff_global_vel = ref_obj_vel - tar_vel
    diff_local_vel = torch_utils.quat_rotate(heading_rot, diff_global_vel)

    ref_obj_ang_vel = extract_data_component_fn('obj_rot_vel', obs=ref_obs)
    diff_global_ang_vel = ref_obj_ang_vel - tar_ang_vel
    diff_local_ang_vel = torch_utils.quat_rotate(heading_rot, diff_global_ang_vel)

    obs = torch.cat([local_tar_vel, local_tar_ang_vel, diff_local_obj_pos_flat, diff_local_obj_rot_obs,
                    diff_local_vel, diff_local_ang_vel], dim=-1)
    return obs


def build_hoi_observations(root_pos, root_rot, root_vel, root_ang_vel, dof_pos, dof_vel, body_pos,
                          local_root_obs, root_height_obs, dof_obs_size, target_states, target_contact_buf,
                          contact_buf, object_points, body_rot, body_vel, body_rot_vel, compute_sdf_fn):
    """Build HOI observations from current state.

    Copied directly from Isaac Gym intermimic.py:702
    """
    contact = torch.any(torch.abs(contact_buf) > 0.1, dim=-1).float()
    target_contact = torch.any(torch.abs(target_contact_buf) > 0.1, dim=-1).float().unsqueeze(1)

    tar_pos = target_states[:, 0:3]
    tar_rot = target_states[:, 3:7]
    obj_rot_extend = tar_rot.unsqueeze(1).repeat(1, object_points.shape[1], 1).view(-1, 4)
    object_points_extend = object_points.view(-1, 3)
    obj_points = torch_utils.quat_rotate(obj_rot_extend, object_points_extend).view(tar_rot.shape[0], object_points.shape[1], 3) + tar_pos.unsqueeze(1)
    ig = compute_sdf_fn(body_pos, obj_points).view(-1, 3)
    heading_rot = torch_utils.calc_heading_quat_inv(root_rot)
    heading_rot_extend = heading_rot.unsqueeze(1).repeat(1, body_pos.shape[1], 1).view(-1, 4)
    ig = quat_rotate(heading_rot_extend, ig).view(tar_pos.shape[0], -1)

    # Flatten all body tensors to 2D for concatenation
    body_pos_flat = body_pos.reshape(body_pos.shape[0], -1)
    body_rot_flat = body_rot.reshape(body_rot.shape[0], -1)
    body_vel_flat = body_vel.reshape(body_vel.shape[0], -1)
    body_rot_vel_flat = body_rot_vel.reshape(body_rot_vel.shape[0], -1)
    contact_flat = contact.view(contact.shape[0], -1)

    obs = torch.cat((root_pos, root_rot, dof_pos, dof_vel,
                     body_pos_flat, body_rot_flat,
                     body_vel_flat, body_rot_vel_flat,
                     target_states, ig, contact_flat, target_contact), dim=-1)
    return obs


def compute_ig_obs(curr_obs, ref_obs, key_body_ids, extract_data_component_fn):
    """Compute processed interaction geometry observations.

    Copied directly from Isaac Gym intermimic.py:661
    """
    env_ids_shape = curr_obs.shape[0]
    ig = extract_data_component_fn('ig', obs=curr_obs).view(env_ids_shape, -1, 3)
    # print("ig", ig)
    ig_norm = ig.norm(dim=-1, keepdim=True)
    ig_all = ig / (ig_norm + 1e-6) * (-5 * ig_norm).exp()
    ig = ig_all[:, key_body_ids, :].view(env_ids_shape, -1)
    ig_all = ig_all.view(env_ids_shape, -1)
    ref_ig = extract_data_component_fn('ig', obs=ref_obs)
    # print("ref_ig", ref_ig.view(ref_obs.shape[0], -1, 3))
    ref_ig = ref_ig.view(ref_obs.shape[0], -1, 3)[:, key_body_ids, :]
    ref_ig_norm = ref_ig.norm(dim=-1, keepdim=True)
    ref_ig = ref_ig / (ref_ig_norm + 1e-6) * (-5 * ref_ig_norm).exp()
    ref_ig = ref_ig.view(env_ids_shape, -1)
    return ig_all, ig, ref_ig
