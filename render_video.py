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
from scene import Scene
import os
import numpy as np
import mediapy as media
import matplotlib.pyplot as plt
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
from utils.general_utils import safe_state
from utils.system_utils import searchForMaxIteration
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
from utils.camera_utils import cameraList_from_camInfos
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
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

def render_set(model_path, camera_path_name, iteration, views, gaussians, pipeline, background, kernel_size, scale_factor, depth):
    imgs = []
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

@torch.no_grad()
def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, camera_path, load_from_checkpoints: bool = False, depth: bool = False, save_images: bool = False, num_frames: int = 0):
    global writer
    
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.appearance_enabled, dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
        if load_from_checkpoints:
            checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
            print(f"Loading model from checkpoint {checkpoint_path}")
            (model_params, first_iter) = torch.load(checkpoint_path)
            gaussians.load_from_checkpoints(model_params)
        
        print(gaussians._xyz.shape)
        # print Gaussian scale statistics
        gs_scale = gaussians.get_scaling.max(dim=1).values
        gs_scale_np = gs_scale.detach().cpu().numpy()
        print("Min: ", gs_scale.min().item())
        print("Max: ", gs_scale.max().item())
        print("Mean: ", gs_scale.mean().item())
        print("Std: ", gs_scale.std().item())
        print("Median: ", gs_scale.median().item())
        print("Q99: ", gs_scale.kthvalue(int(0.99 * gs_scale.shape[0]), dim=0).values.item())
        plt.figure(figsize=(10, 6))
        plt.hist(gs_scale_np, bins=480, range=(0, 30), edgecolor='black')

        # Customize the plot
        plt.title('Histogram with 240 bins')
        plt.xlabel('Values')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)

        camera_path_name = camera_path.split('/')[-1].split('.')[0]
        path = os.path.join(dataset.model_path, 'hist', "ours_{}".format(iteration))
        hist_path = os.path.join(path, f"{camera_path_name}{'_depth' if depth else ''}.png")
        os.makedirs(os.path.dirname(hist_path), exist_ok=True)
        plt.savefig(hist_path, dpi=600, bbox_inches='tight')
        print(f"Histogram saved to {hist_path}")


        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        scale_factor = dataset.resolution
        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        # Read camera path json
        with open(camera_path, 'r') as file:
            camera_path_data = json.load(file)

        cams, radius = get_path_from_json(camera_path_data)

        # Prune the Gaussian outside camera (setting their opacity to 0)
        # gaussians.prune_by_radius(120)
        
        # If num_frames is specified and less than total frames, select evenly spaced frames
        if num_frames > 0 and num_frames < len(cams):
            indices = np.linspace(0, len(cams) - 1, num_frames, dtype=int)
            selected_cams = [cams[i] for i in indices]
            print(f"Rendering {num_frames} evenly spaced frames out of {len(cams)} total frames")
            print(f"Note: Video duration will be shorter ({num_frames / camera_path_data['fps']:.2f}s vs {len(cams) / camera_path_data['fps']:.2f}s)")
            cams = selected_cams
        
        path = os.path.join(dataset.model_path, 'video', "ours_{}".format(iteration))
        video_path = os.path.join(path, f"{camera_path_name}{'_depth' if depth else ''}.mp4")
        # create parent folder
        print(video_path)
        os.makedirs(os.path.dirname(video_path), exist_ok=True)
        
        cam_infos = cameraList_from_camInfos(cams, 1, dataset, is_testing=True)
        imgs = render_set(dataset.model_path, camera_path_name, scene.loaded_iter, cam_infos, gaussians, pipeline, background, kernel_size, scale_factor=scale_factor, depth=depth)

        if save_images:
            frames_path = os.path.join(path, f"{camera_path_name}{'_depth' if depth else ''}_frames")
            os.makedirs(frames_path, exist_ok=True)
            print("Saving frames")
            for idx, img in tqdm(enumerate(imgs)):
                img_path = os.path.join(frames_path, '{0:05d}'.format(idx) + ".png")
                Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(img_path)

        with media.VideoWriter(
            path=video_path,
            shape=(cams[0].height, cams[0].width),
            fps=camera_path_data["fps"],
        ) as writer:
            for img in imgs:
                writer.add_image(img)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_from_checkpoints", action="store_true")
    parser.add_argument("--camera_path", type=str)
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--save_images", action="store_true")
    parser.add_argument("--num_frames", type=int, default=0, 
                       help="Number of frames to render and save (0 = all frames)")
    
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), 
                args.camera_path, args.load_from_checkpoints, args.depth, 
                args.save_images, args.num_frames)
