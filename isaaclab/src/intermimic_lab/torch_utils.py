"""Torch utility functions for InterMimic - IsaacLab version."""

import torch
from typing import Tuple
from isaaclab.utils.math import quat_mul as quat_mul_isaaclab, quat_conjugate
from isaaclab.utils.math import quat_rotate as quat_rotate_isaaclab

# Re-export for compatibility
quat_rotate = quat_rotate_isaaclab
quat_mul = quat_mul_isaaclab


@torch.jit.script
def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Calculate heading quaternion inverse (yaw component only).

    Args:
        q: Quaternion (wxyz format)

    Returns:
        Inverse heading quaternion
    """
    # Extract yaw component
    # For wxyz format: w, x, y, z
    heading = torch.zeros_like(q)
    heading[..., 0] = q[..., 0]  # w
    heading[..., 3] = q[..., 3]  # z (yaw axis)

    # Normalize - use explicit p=2 argument for JIT compatibility
    heading = heading / (torch.norm(heading, p=2, dim=-1, keepdim=True) + 1e-8)

    # Return conjugate (inverse for unit quaternions)
    return quat_conjugate(heading)


@torch.jit.script
def calc_heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Calculate heading quaternion (yaw component only).

    Args:
        q: Quaternion (wxyz format)

    Returns:
        Heading quaternion
    """
    # Extract yaw component
    heading = torch.zeros_like(q)
    heading[..., 0] = q[..., 0]  # w
    heading[..., 3] = q[..., 3]  # z (yaw axis)

    # Normalize - use explicit p=2 argument for JIT compatibility
    heading = heading / (torch.norm(heading, p=2, dim=-1, keepdim=True) + 1e-8)

    return heading


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Convert quaternion to tangent-normal representation.

    This creates a 6D representation using tangent and normal vectors.

    Args:
        q: Quaternion (wxyz format)

    Returns:
        6D tangent-normal vector

    Note:
        Not JIT-compiled because it calls quat_rotate which contains logging code.
    """
    # Reference tangent vector (x-axis)
    ref_tan = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    # Reference normal vector (z-axis)
    ref_norm = torch.zeros_like(ref_tan)
    ref_norm[..., 2] = 1
    norm = quat_rotate(q, ref_norm)

    # Concatenate
    norm_tan = torch.cat([tan, norm], dim=-1)
    return norm_tan


@torch.jit.script
def quat_mul_norm(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions and normalize the result.

    Args:
        q1: First quaternion
        q2: Second quaternion

    Returns:
        Normalized product quaternion
    """
    result = quat_mul(q1, q2)
    return result / (torch.norm(result, p=2, dim=-1, keepdim=True) + 1e-8)


@torch.jit.script
def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Compute quaternion inverse (conjugate for unit quaternions).

    Args:
        q: Quaternion (wxyz format)

    Returns:
        Inverse quaternion
    """
    return quat_conjugate(q)


@torch.jit.script
def compute_sdf(body_pos: torch.Tensor, obj_points: torch.Tensor) -> torch.Tensor:
    """Compute signed distance field from body positions to object surface points.

    Args:
        body_pos: Body positions (num_envs, num_bodies, 3)
        obj_points: Object surface points (num_envs, num_points, 3)

    Returns:
        SDF vectors for each body (num_envs, num_bodies, 3)
    """
    # Expand dimensions for broadcasting
    # body_pos: (num_envs, num_bodies, 1, 3)
    # obj_points: (num_envs, 1, num_points, 3)
    body_pos_expanded = body_pos.unsqueeze(2)
    obj_points_expanded = obj_points.unsqueeze(1)

    # Compute distances to all object points
    # Shape: (num_envs, num_bodies, num_points, 3)
    dists = body_pos_expanded - obj_points_expanded

    # Find closest point for each body
    # Shape: (num_envs, num_bodies, num_points)
    dist_norms = torch.norm(dists, p=2, dim=-1)

    # Get index of closest point
    closest_idx = dist_norms.argmin(dim=-1)  # (num_envs, num_bodies)

    # Gather the closest point vectors
    # Need to expand closest_idx for gathering
    closest_idx_expanded = closest_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, -1, 3)
    # Shape: (num_envs, num_bodies, 1, 3)

    # Gather along the point dimension (dim=2)
    closest_vecs = torch.gather(dists, 2, closest_idx_expanded).squeeze(2)
    # Shape: (num_envs, num_bodies, 3)

    return closest_vecs
