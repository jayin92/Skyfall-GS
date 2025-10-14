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
from diff_gauss import GaussianRasterizationSettings, GaussianRasterizer
# from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, kernel_size: float, scaling_modifier = 1.0, override_color = None, subpixel_offset=None, testing=False, appearance_embedding=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    if subpixel_offset is None:
        subpixel_offset = torch.zeros((int(viewpoint_camera.image_height), int(viewpoint_camera.image_width), 2), dtype=torch.float32, device="cuda")
        
    raster_settings = GaussianRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        kernel_size=kernel_size,
        subpixel_offset=subpixel_offset,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    means3D = pc.get_xyz
    means2D = screenspace_points
    opacity = pc.get_opacity_with_3D_filter

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling_with_3D_filter
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None

    embedding = None
    if pc.appearance_enabled:
        if not testing:
            if appearance_embedding is not None:
                embedding = appearance_embedding
            else:
                try:
                    embedding = pc.appearance_embeddings[viewpoint_camera.uid]
                    # print("Use embedding from camera", viewpoint_camera.uid)
                    # embedding = pc.appearance_embeddings[5]
                except:
                    print(pc.appearance_embeddings.shape)
                    print('Embedding not found for camera', viewpoint_camera.uid, 'use mean embedding instead')
                    with torch.no_grad():  # important to avoid no needed gradients
                        embedding = torch.mean(pc.appearance_embeddings, dim=0)
                        # embedding = pc.appearance_embeddings[0]
        else:
            # using the mean of embedding as test embedding
            with torch.no_grad():  # important to avoid no needed gradients
                if appearance_embedding is not None:
                    embedding = appearance_embedding
                else:
                    embedding = torch.mean(pc.appearance_embeddings, dim=0)
                    uid = min(6, len(pc.appearance_embeddings)-1)
                    # print(f"Using embedding from uid {uid}")
                    embedding = pc.appearance_embeddings[uid]
    if pc.appearance_enabled and embedding is not None:
        embedding_expanded = embedding[None].repeat(len(means2D), 1)

        assert pc.appearance_mlp is not None
        assert pc._embeddings is not None
        colors_toned = pc.appearance_mlp(pc._embeddings, embedding_expanded, pc.get_features).clamp_max(1.0)

        shdim = (pc.max_sh_degree + 1) ** 2
        colors_toned = colors_toned.view(-1, shdim, 3).transpose(1, 2).contiguous().clamp_max(1.0)
        dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
        dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
        colors_toned = eval_sh(pc.active_sh_degree, colors_toned, dir_pp_normalized)
        colors_toned = torch.clamp_min(colors_toned + 0.5, 0.0)
        colors_precomp = colors_toned
    elif override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (pc.get_xyz - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, rendered_depth, rendered_norm, rendered_alpha, radii, extra = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity.float(),
        scales = scales.float(),
        rotations = rotations,
        cov3Ds_precomp = cov3D_precomp) # For diff_gauss
    # rendered_image, radii = rasterizer(
    #     means3D = means3D,
    #     means2D = means2D,
    #     shs = shs,
    #     colors_precomp = colors_precomp,
    #     opacities = opacity.float(),
    #     scales = scales.float(),
    #     rotations = rotations,
    #     cov3D_precomp = cov3D_precomp) # For mip-splatting orignal

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    # return {"render": rendered_image,
    #         "viewspace_points": screenspace_points,
    #         "visibility_filter" : radii > 0,
    #         "radii": radii}
    return {"render": rendered_image,
            "render_depth": rendered_depth,
            "render_norm": rendered_norm,
            "render_alpha": rendered_alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "extra": extra}
