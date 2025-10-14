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

from typing import List
from scene.cameras import Camera
import numpy as np
import torch
import random
import math
from utils.general_utils import PILtoTorch
from utils.graphics_utils import fov2focal
from scene.dataset_readers import CameraInfo
from PIL import Image

WARNED = False

def loadCam(args, id, cam_info, resolution_scale, is_testing=False, optimizing=False):
    if not is_testing:
        orig_w, orig_h = cam_info.image.size

        if args.resolution in [1, 2, 4, 8, 16, 32, 64]:
            resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    global WARNED
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        resized_image_rgb = PILtoTorch(cam_info.image, resolution)
        depth = mask = None
        if cam_info.depth is not None:
            depth = torch.tensor(cam_info.depth).unsqueeze(0)
        if cam_info.mask is not None:
            mask = torch.tensor(cam_info.mask).unsqueeze(0)

        gt_image = resized_image_rgb[:3, ...]
        loaded_mask = None

        if resized_image_rgb.shape[1] == 4:
            loaded_mask = resized_image_rgb[3:4, ...]

        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    cx=cam_info.cx, cy=cam_info.cy,
                    image=gt_image, gt_alpha_mask=loaded_mask,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                    depth=depth, mask=mask, optimizing=optimizing)
    else:
        resized_image_rgb = PILtoTorch(cam_info.image, cam_info.image.size)
        gt_image = resized_image_rgb[:3, ...]

        return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                    cx=cam_info.cx, cy=cam_info.cy,
                    image=gt_image, gt_alpha_mask=None,
                    image_name=cam_info.image_name, uid=id, data_device=args.data_device,
                    depth=None, mask=None, optimizing=optimizing)

def cameraList_from_camInfos(cam_infos, resolution_scale, args, is_idu=False, is_testing=False, is_pseudo_cam=False, optimizing=False):
    camera_list = []

    print(f"optimizing: {optimizing}")

    for id, c in enumerate(cam_infos):
        if is_pseudo_cam:
            uid = c.uid
        else:
            uid = id + (1000 if is_idu else 0)
        camera_list.append(loadCam(args, uid, c, resolution_scale, is_testing=is_testing, optimizing=optimizing))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width),
        'cx' : camera.cx,
        'cy' : camera.cy
    }
    return camera_entry

def three_js_perspective_camera_focal_length(fov: float, image_height: int):
    """Returns the focal length of a three.js perspective camera.

    Args:
        fov: the field of view of the camera in degrees.
        image_height: the height of the image in pixels.
    """
    if fov is None:
        print("Warning: fov is None, using default value")
        return 50
    pp_h = image_height / 2.0
    focal_length = pp_h / np.tan(fov * (np.pi / 180.0) / 2.0)
    return focal_length


def look_at_to_c2w(eye, target, up):
    # Convert inputs to numpy arrays if needed
    eye = np.array(eye)
    target = np.array(target)
    up = np.array(up)
    
    # Calculate forward (negative z-axis)
    forward = target - eye
    forward = forward / np.linalg.norm(forward)
    
    # Calculate right vector (x-axis)
    right = np.cross(forward, up)
    right = right / np.linalg.norm(right)
    
    # Calculate corrected up vector (y-axis)
    up = np.cross(right, forward)
    up = up / np.linalg.norm(up)
    
    # Construct rotation matrix
    R = np.array([
        [right[0], up[0], -forward[0]],
        [right[1], up[1], -forward[1]],
        [right[2], up[2], -forward[2]]
    ])
    
    # Construct full 4x4 transformation matrix
    c2w = np.eye(4)
    c2w[:3, :3] = R
    c2w[:3, 3] = eye

    # NOTE: covert from OpenGL to COLMAP coordinate system
    c2w[:3, 1:3] *= -1 

    
    return c2w

def gen_idu_orbit_camera(
        target: List[float], 
        elevation: float, 
        radius: float, 
        num_cams: int=6,
        num_samples: int=4,
        height: int=512, 
        width: int=512,
        fov: float=60.0,
        use_new_id: bool=True,
        num_train_cams: int=None
    ) -> List[CameraInfo]:
    # Convert target to numpy array
    target = np.array(target)
    # random phi offset
    theta_offset = random.uniform(-np.pi/4, np.pi/4)
    theta_offset = 0  # TODO: disable random offset for now
        
    # Calculate up vector
    up = np.array([0, 0, 1])
    
    # Generate camera-to-world matrices
    c2ws = []
    uids = []
    for i in range(num_cams):
        theta = 2 * np.pi * i / num_cams + theta_offset
        phi = np.pi * elevation / 180
        eye = target + np.array([
            radius * np.cos(theta) * np.cos(phi),
            radius * np.sin(theta) * np.cos(phi),
            radius * np.sin(phi)    
        ])
        c2w = look_at_to_c2w(eye, target, up)
        uid = 1000 + i if use_new_id else random.randint(0, num_train_cams-1)
        for _ in range(num_samples):
            uids.append(uid)
            c2ws.append(c2w)
    fov_r = np.deg2rad(fov)
    camInfos = []
    for i, c2w in enumerate(c2ws):
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        if use_new_id:
            assert num_train_cams is None
            uid = 1000 + i 
        else:
            assert num_train_cams is not None
            uid = uids[i]
        image = Image.new("RGB", (width, height), (0, 0, 0))
        camInfo = CameraInfo(
                    uid=uid, R=R, T=T, 
                    FovY=fov_r, FovX=fov_r, 
                    cx=0, cy=0,
                    image=image, image_path=None,
                    image_name=f"e{elevation}_r{radius}_{i:05d}.png", depth=None, mask=None,
                    width=width, height=height
                )
        camInfos.append(camInfo)

    return camInfos