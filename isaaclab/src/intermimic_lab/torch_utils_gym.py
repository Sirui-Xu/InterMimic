"""Quaternion utilities in Isaac Gym (xyzw) ordering."""

import torch


def quat_mul(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions (xyzw ordering)."""
    x1, y1, z1, w1 = torch.unbind(q, dim=-1)
    x2, y2, z2, w2 = torch.unbind(r, dim=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack((x, y, z, w), dim=-1)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """Quaternion conjugate (xyzw)."""
    result = q.clone()
    result[..., :3] = -result[..., :3]
    return result


def quat_inverse(q: torch.Tensor) -> torch.Tensor:
    """Inverse of a unit quaternion (xyzw)."""
    return quat_conjugate(q)


def quat_mul_norm(q: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """Multiply two quaternions and normalize the result."""
    prod = quat_mul(q, r)
    return prod / (torch.norm(prod, p=2, dim=-1, keepdim=True) + 1e-8)


def quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector(s) v by quaternion q (xyzw)."""
    q_vec = q[..., :3]
    q_w = q[..., 3:].expand_as(q_vec[..., :1])
    t = 2.0 * torch.cross(q_vec, v, dim=-1)
    return v + q_w * t + torch.cross(q_vec, t, dim=-1)


def quat_to_tan_norm(q: torch.Tensor) -> torch.Tensor:
    """Represent quaternion as concatenated tangent and normal vectors."""
    ref_tan = torch.zeros(q.shape[:-1] + (3,), dtype=q.dtype, device=q.device)
    ref_tan[..., 0] = 1.0
    tan = quat_rotate(q, ref_tan)

    ref_norm = torch.zeros_like(ref_tan)
    ref_norm[..., 2] = 1.0
    norm = quat_rotate(q, ref_norm)

    return torch.cat([tan, norm], dim=-1)


def calc_heading(q: torch.Tensor) -> torch.Tensor:
    """Compute yaw/heading angle from quaternion (xyzw)."""
    ref_dir = torch.zeros_like(q[..., :3])
    ref_dir[..., 0] = 1.0
    rot_dir = quat_rotate(q, ref_dir)
    return torch.atan2(rot_dir[..., 1], rot_dir[..., 0])


def _heading_quaternion(heading: torch.Tensor) -> torch.Tensor:
    half_angle = heading * 0.5
    sin_half = torch.sin(half_angle)
    cos_half = torch.cos(half_angle)
    quat = torch.zeros(heading.shape + (4,), dtype=heading.dtype, device=heading.device)
    quat[..., 2] = sin_half  # rotation around z-axis -> axis (0,0,1)
    quat[..., 3] = cos_half
    return quat


def calc_heading_quat(q: torch.Tensor) -> torch.Tensor:
    """Get heading quaternion (xyzw) keeping only yaw."""
    heading = calc_heading(q)
    return _heading_quaternion(heading)


def calc_heading_quat_inv(q: torch.Tensor) -> torch.Tensor:
    """Inverse heading quaternion (xyzw)."""
    heading = calc_heading(q)
    return _heading_quaternion(-heading)
