#!/usr/bin/env python
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
import json
import os
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
from gaussian_renderer import GaussianModel

from typing import Any, Dict, Optional, Tuple, List
from scene.dataset_readers import CameraInfo

from PIL import Image
from typing import NamedTuple

import matplotlib.pyplot as plt


writer = None

coord_transform = np.array([
    [1, 0, 0, 0],
    [0, 0, 1, 0],
    [0, -1, 0, 0],
    [0, 0, 0, 1]
])

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

@torch.no_grad()
def get_path_from_json(camera_path: Dict[str, Any]) -> List[CameraInfo]:
    """Takes a camera path dictionary and returns a trajectory as a Camera instance.

    Args:
        camera_path: A dictionary of the camera path information coming from the viewer.

    Returns:
        A Cameras instance with the camera path.
    """

    image_height = camera_path["render_height"]
    image_width = camera_path["render_width"]
    radius = camera_path["_radius"]



    print(f"Image size: {image_width}x{image_height}")

    if "camera_type" not in camera_path:
        camera_type = "PERSPECTIVE"

    cam_infos = []
    print(f"Reading {len(camera_path['camera_path'])} cameras")
    need_transform = True if "keyframes" in camera_path else False
    for idx, camera in enumerate(camera_path["camera_path"]):
        # pose
        if need_transform and False:
            print("Transforming camera pose")
            c2w = coord_transform @ np.array(camera["camera_to_world"]).reshape((4, 4))
        else:
            c2w = np.array(camera["camera_to_world"]).reshape((4, 4))

        # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
        c2w[:3, 1:3] *= -1
        w2c = np.linalg.inv(c2w)
        R = np.transpose(w2c[:3,:3])
        T = w2c[:3, 3]
        cx = 0
        cy = 0
        # field of view
        fov = camera["fov"]
        focal_length = three_js_perspective_camera_focal_length(fov, image_height)
        FovX = focal2fov(focal_length, image_width)
        FovY = focal2fov(focal_length, image_height)
        # Degree to radians
        # fov = np.deg2rad(fov)
        # FovX = fov
        # FovY = fov

        # pseudo image using PIL
        image = Image.new("1", (image_width, image_height), (0))

        cam_infos.append(
            CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                        cx=cx, cy=cy,
                        image=image,
                        image_path="",
                        image_name="",
                        width=image_width,
                        height=image_height,
                        depth=None,
                        mask=None)
        )
    return cam_infos, radius

def colorize_depth_torch(depth_tensor, mask=None, normalize=True, cmap='Spectral'):
    """
    Colorize depth map using matplotlib colormap, implemented for PyTorch tensors.
    Args:
        depth_tensor: Input depth tensor [B, H, W] or [H, W]
        mask: Optional mask tensor [B, H, W] or [H, W]
        normalize: Whether to normalize the depth values
        cmap: Matplotlib colormap name
    Returns:
        Colored depth tensor [B, 3, H, W] or [3, H, W]
    """

    # Process each item in batch
    # Convert to numpy for matplotlib colormap
    depth = depth_tensor[0].detach().cpu().numpy()

    # Handle invalid depths
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        mask_b = mask[0].detach().cpu().numpy()
        depth = np.where((depth > 0) & mask_b, depth, np.nan)

    # Convert to disparity (inverse depth)
    disp = 1. / depth

    # Normalize disparity
    if normalize:
        min_disp = np.nanquantile(disp, 0.01)
        max_disp = np.nanquantile(disp, 0.99)
        # Avoid division by zero for constant depth
        disp_range = max_disp - min_disp
        if disp_range > 1e-6:
            disp = (disp - min_disp) / disp_range
        else:
            disp = np.zeros_like(disp)

    # Apply colormap
    colored = plt.get_cmap(cmap)(1.0 - disp)
    colored = np.nan_to_num(colored, 0)
    colored = (colored.clip(0, 1) * 255).astype(np.uint8)[:, :, :3]

    # Convert back to torch tensor and rearrange dimensions
    colored = torch.from_numpy(colored).float() / 255.0
    colored = colored.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]

    return colored.to(depth_tensor.device)

def detect_sh_degree_from_ply(ply_path: str) -> int:
    """Detect the spherical harmonics degree from the PLY file."""
    from plyfile import PlyData

    plydata = PlyData.read(ply_path)
    field_names = [p.name for p in plydata.elements[0].properties]

    # Count f_rest fields to determine SH degree
    extra_f_names = [name for name in field_names if name.startswith("f_rest_")]
    num_rest_coeffs = len(extra_f_names)

    # For SH degree d: total coeffs = 3 * (d+1)^2, rest coeffs = total - 3 (DC)
    # So: num_rest_coeffs = 3 * (d+1)^2 - 3 = 3 * ((d+1)^2 - 1)
    # Solving: (d+1)^2 = (num_rest_coeffs / 3) + 1
    if num_rest_coeffs == 0:
        sh_degree = 0
    else:
        sh_degree = int(np.sqrt((num_rest_coeffs / 3) + 1)) - 1

    print(f"Detected SH degree: {sh_degree} (from {num_rest_coeffs} rest coefficients)")
    return sh_degree

def load_ply_gaussians(ply_path: str, sh_degree: int = None, appearance_enabled: bool = False,
                      appearance_n_fourier_freqs: int = 4, appearance_embedding_dim: int = 32) -> GaussianModel:
    """Load Gaussians from a PLY file.

    Args:
        ply_path: Path to the PLY file
        sh_degree: Spherical harmonics degree (auto-detected if None)
        appearance_enabled: Whether appearance modeling is enabled
        appearance_n_fourier_freqs: Number of Fourier frequencies for appearance
        appearance_embedding_dim: Appearance embedding dimension

    Returns:
        GaussianModel: Loaded Gaussian model
    """
    from plyfile import PlyData

    # Auto-detect SH degree if not provided
    if sh_degree is None:
        sh_degree = detect_sh_degree_from_ply(ply_path)

    # First, check if the PLY file has filter_3D field (Mip-Splatting specific)
    plydata = PlyData.read(ply_path)
    field_names = [p.name for p in plydata.elements[0].properties]
    has_filter_3d = 'filter_3D' in field_names

    gaussians = GaussianModel(sh_degree, appearance_enabled, appearance_n_fourier_freqs, appearance_embedding_dim)

    if has_filter_3d:
        # Use the original Mip-Splatting load_ply method
        print(f"Loading Mip-Splatting PLY with filter_3D field (SH degree: {sh_degree})")
        gaussians.load_ply(ply_path)
    else:
        # Load as standard 3D Gaussian Splatting PLY without filter_3D
        print(f"Loading standard 3D Gaussian Splatting PLY (no filter_3D field, SH degree: {sh_degree})")
        load_standard_ply(gaussians, ply_path)

    return gaussians

def load_standard_ply(gaussians: GaussianModel, path: str):
    """Load standard 3D Gaussian Splatting PLY file without filter_3D field."""
    from plyfile import PlyData
    import numpy as np

    plydata = PlyData.read(path)

    xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                    np.asarray(plydata.elements[0]["y"]),
                    np.asarray(plydata.elements[0]["z"])),  axis=1)
    opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

    # Set default filter_3D values (no anti-aliasing filter)
    filter_3D = np.ones((xyz.shape[0], 1))  # Default to 1.0 (no filtering)

    features_dc = np.zeros((xyz.shape[0], 3, 1))
    features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
    features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
    features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

    extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
    extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
    assert len(extra_f_names)==3*(gaussians.max_sh_degree + 1) ** 2 - 3
    features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
    for idx, attr_name in enumerate(extra_f_names):
        features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
    features_extra = features_extra.reshape((features_extra.shape[0], 3, (gaussians.max_sh_degree + 1) ** 2 - 1))

    scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
    scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
    scales = np.zeros((xyz.shape[0], len(scale_names)))
    for idx, attr_name in enumerate(scale_names):
        scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

    rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
    rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
    rots = np.zeros((xyz.shape[0], len(rot_names)))
    for idx, attr_name in enumerate(rot_names):
        rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

    # Convert to tensors and assign to gaussians
    gaussians._xyz = torch.nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._features_dc = torch.nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._features_rest = torch.nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
    gaussians._opacity = torch.nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._scaling = torch.nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
    gaussians._rotation = torch.nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

    # Set filter_3D as a direct attribute (not parameter) - this is how Mip-Splatting accesses it
    gaussians.filter_3D = torch.tensor(filter_3D, dtype=torch.float, device="cuda")

    gaussians.active_sh_degree = gaussians.max_sh_degree

def render_set_from_ply(ply_path, camera_path_name, views, pipeline, background, kernel_size, scale_factor, depth, sh_degree=None, cameras=None):
    imgs = []

    # Load Gaussians from PLY
    print(f"Loading Gaussians from PLY: {ply_path}")
    gaussians = load_ply_gaussians(ply_path, sh_degree=sh_degree)

    # Compute 3D filter if cameras are provided (this is key for proper Mip-Splatting rendering)
    if cameras is not None:
        print("Computing 3D filter from camera parameters...")
        gaussians.compute_3D_filter(cameras)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        if not depth:
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=True)["render"]
        else:
            rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=True)["render_depth"]
            rendering = torch.nan_to_num(rendering, nan=0.0, posinf=0.0, neginf=0.0)
            rendering = colorize_depth_torch(rendering)
        img = rendering.cpu().numpy().transpose(1, 2, 0)
        imgs.append(img)
    return imgs

class PipelineParams:
    """Minimal pipeline parameters for rendering."""
    def __init__(self):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.debug = False

class MinimalArgs:
    """Minimal args object for camera loading."""
    def __init__(self, data_device="cuda"):
        self.data_device = data_device

@torch.no_grad()
def render_video_from_ply(ply_path: str, camera_path: str, output_path: str = None,
                         depth: bool = False, save_images: bool = False, num_frames: int = 0,
                         kernel_size: float = 0.1, white_background: bool = False, sh_degree: int = None):
    """Render video from PLY file and camera path.

    Args:
        ply_path: Path to the PLY file
        camera_path: Path to the camera trajectory JSON file
        output_path: Output directory path (if None, uses PLY directory)
        depth: Whether to render depth maps
        save_images: Whether to save individual frames
        num_frames: Number of frames to render (0 = all frames)
        kernel_size: Kernel size for anti-aliasing
        white_background: Whether to use white background
    """

    # Setup pipeline
    pipeline = PipelineParams()

    # Background color
    bg_color = [1, 1, 1] if white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    # Read camera path json
    with open(camera_path, 'r') as file:
        camera_path_data = json.load(file)

    cams, radius = get_path_from_json(camera_path_data)

    # If num_frames is specified and less than total frames, select evenly spaced frames
    if num_frames > 0 and num_frames < len(cams):
        indices = np.linspace(0, len(cams) - 1, num_frames, dtype=int)
        selected_cams = [cams[i] for i in indices]
        print(f"Rendering {num_frames} evenly spaced frames out of {len(cams)} total frames")
        print(f"Note: Video duration will be shorter ({num_frames / camera_path_data['fps']:.2f}s vs {len(cams) / camera_path_data['fps']:.2f}s)")
        cams = selected_cams

    # Setup output paths
    if output_path is None:
        output_path = os.path.dirname(ply_path)

    camera_path_name = os.path.splitext(os.path.basename(camera_path))[0]
    ply_name = os.path.splitext(os.path.basename(ply_path))[0]

    video_dir = os.path.join(output_path, 'video_from_ply')
    video_path = os.path.join(video_dir, f"{ply_name}_{camera_path_name}{'_depth' if depth else ''}.mp4")

    print(f"Output video: {video_path}")
    os.makedirs(os.path.dirname(video_path), exist_ok=True)

    # Convert camera infos to camera objects
    minimal_args = MinimalArgs(data_device="cuda")
    cam_infos = cameraList_from_camInfos(cams, 1.0, minimal_args, is_testing=True)

    # Render frames
    imgs = render_set_from_ply(ply_path, camera_path_name, cam_infos, pipeline, background, kernel_size, 1.0, depth, sh_degree, cameras=cam_infos)

    # Save individual frames if requested
    if save_images:
        frames_path = os.path.join(video_dir, f"{ply_name}_{camera_path_name}{'_depth' if depth else ''}_frames")
        os.makedirs(frames_path, exist_ok=True)
        print("Saving frames")
        for idx, img in tqdm(enumerate(imgs)):
            img_path = os.path.join(frames_path, '{0:05d}'.format(idx) + ".png")
            Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(img_path)

    # Create video
    with media.VideoWriter(
        path=video_path,
        shape=(cams[0].height, cams[0].width),
        fps=camera_path_data["fps"],
    ) as writer:
        for img in imgs:
            writer.add_image(img)

    print(f"Video saved to: {video_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Render video from PLY file")
    parser.add_argument("--ply_path", type=str, required=True, help="Path to the PLY file")
    parser.add_argument("--camera_path", type=str, required=True, help="Path to camera trajectory JSON")
    parser.add_argument("--output_path", type=str, default=None, help="Output directory (default: PLY directory)")
    parser.add_argument("--depth", action="store_true", help="Render depth maps")
    parser.add_argument("--save_images", action="store_true", help="Save individual frames")
    parser.add_argument("--num_frames", type=int, default=0, help="Number of frames to render (0 = all)")
    parser.add_argument("--kernel_size", type=float, default=0.1, help="Kernel size for anti-aliasing")
    parser.add_argument("--white_background", action="store_true", help="Use white background")
    parser.add_argument("--sh_degree", type=int, default=None, help="Spherical harmonics degree (auto-detected if not specified)")
    parser.add_argument("--quiet", action="store_true")

    args = parser.parse_args()

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Render video
    render_video_from_ply(
        ply_path=args.ply_path,
        camera_path=args.camera_path,
        output_path=args.output_path,
        depth=args.depth,
        save_images=args.save_images,
        num_frames=args.num_frames,
        kernel_size=args.kernel_size,
        white_background=args.white_background,
        sh_degree=args.sh_degree
    )

    # Example usage:
    # python render_video_from_ply.py --ply_path fused/model.ply --camera_path camera_path/r300_e60_fov60.json
    # python render_video_from_ply.py --ply_path fused/model.ply --camera_path camera_path/r300_e60_fov60.json --depth --save_images
    # python render_video_from_ply.py --ply_path fused/model.ply --camera_path camera_path/r300_e60_fov60.json --num_frames 120 --white_background