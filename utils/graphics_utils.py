#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=None, scale=1.0):
    """
    Computes the transformation matrix from world to view coordinates.

    Args:
        R (np.ndarray or torch.Tensor): Rotation matrix (3x3).
        t (np.ndarray or torch.Tensor): Translation vector (3x1 or 1x3).
        translate (np.ndarray or torch.Tensor): Additional translation vector to apply (3x1).  Defaults to [0, 0, 0] if None.
        scale (float or torch.Tensor): Scaling factor. Defaults to 1.0.

    Returns:
        torch.Tensor: Transformation matrix (4x4).
    """

    # Determine input type and set up accordingly
    if isinstance(R, np.ndarray):
        is_numpy = True
        R = np.float64(R)
        t = np.float64(t)
        if translate is None:
            translate = np.array([.0, .0, .0], dtype=np.float64)
        else:
            translate = np.float64(translate)
        scale = np.float64(scale)
    elif isinstance(R, torch.Tensor):
        is_numpy = False
        if translate is None:
            translate = torch.tensor([.0, .0, .0], dtype=R.dtype, device=R.device)
        else:
            translate = torch.tensor(translate, dtype=R.dtype, device=R.device)
    else:
        raise TypeError("R must be either a NumPy array or a PyTorch tensor.")

    # Ensure correct shapes and types
    if is_numpy:
        if t.ndim == 1:
            t = t.reshape(3, 1)  # Make t a column vector
        Rt = np.zeros((4, 4), dtype=R.dtype)
        Rt[:3, :3] = R.transpose()
        Rt[:3, 3] = t.reshape(3)  # Use reshape for assignment
        Rt[3, 3] = 1.0

        C2W = np.linalg.inv(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale
        C2W[:3, 3] = cam_center
        Rt = np.linalg.inv(C2W)
        return Rt.astype(np.float32) #convert to float32 at the end for consistency with original
    else:  # torch.Tensor
        if t.ndim == 1:
            t = t.reshape(3, 1)
        Rt = torch.zeros((4, 4), dtype=R.dtype, device=R.device)
        Rt[:3, :3] = R.transpose(0, 1)
        Rt[:3, 3] = t.reshape(3)
        Rt[3, 3] = 1.0

        C2W = torch.inverse(Rt)
        cam_center = C2W[:3, 3]
        cam_center = (cam_center + translate) * scale

        # Clone C2W to create a new tensor for modification rather than editing in place.
        C2W_modified = C2W.clone()
        C2W_modified[:3, 3] = cam_center

        Rt_ = torch.inverse(C2W_modified)
        return Rt_


def getProjectionMatrix(znear, zfar, fovX, fovY, cx, cy ):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0

    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = cx
    P[1, 2] = cy
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))