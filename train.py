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

import os
import numpy as np
import torch
import random
import matplotlib.pyplot as plt
from random import randint
from utils.general_utils import get_expon_lr_func
from utils.loss_utils import l1_loss, ssim
from torchmetrics.functional.regression import pearson_corrcoef
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, IDUParams

from utils.camera_utils import gen_idu_orbit_camera, cameraList_from_camInfos
from scene.dataset_readers import CameraInfo

from PIL import Image
from submodules.MoGe.idu_depth import MoGeIDU

# pip install diffusers==0.30.1 huggingface-hub==0.33.4 transformers==4.46.3 tokenizers==0.20.3 (default)
from submodules.FlowEdit.idu_refine import FlowEditRefineIDU 

# fused SSIM, for faster training

from fused_ssim import fused_ssim

# from utils.gpu_utils import GPUManager

import lpips
import math

from torchvision.transforms.functional import to_pil_image

try:
    from tensorboardX import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

os.makedirs("./depth_tmp", exist_ok=True)
moge_standalone = MoGeIDU(
    "./depth_tmp",
    "cuda:0",
    60.0
)

@torch.no_grad()
def create_offset_gt(image, offset):
    height, width = image.shape[1:]
    meshgrid = np.meshgrid(range(width), range(height), indexing='xy')
    id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
    id_coords = torch.from_numpy(id_coords).cuda()
    
    id_coords = id_coords.permute(1, 2, 0) + offset
    # Avoid division by zero for single-pixel dimensions
    id_coords[..., 0] /= max(width - 1, 1)
    id_coords[..., 1] /= max(height - 1, 1)
    id_coords = id_coords * 2 - 1
    
    image = torch.nn.functional.grid_sample(image[None], id_coords[None], align_corners=True, padding_mode="border")[0]
    return image

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt, num_train_cameras=len(scene.getTrainCameras()))
    if checkpoint:
        print("Restoring model from checkpoint")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
        # set correct xyz lr scheduler
        opt.position_lr_max_steps = opt.iterations
        opt.densify_until_iter = opt.iterations
        opt.densify_from_iter = 0
        gaussians.xyz_scheduler_args = get_expon_lr_func(lr_init=opt.position_lr_init * gaussians.spatial_lr_scale,
                                                        lr_final=opt.position_lr_final * gaussians.spatial_lr_scale,
                                                        lr_delay_mult=opt.position_lr_delay_mult,
                                                        max_steps=opt.position_lr_max_steps)
        print("Restored model from checkpoint at iteration {}".format(first_iter))


    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    allCameras = trainCameras + testCameras

    num_train_cams = len(trainCameras)
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

    viewpoint_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")


        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        
        # Pick a random high resolution camera
        if random.random() < 0.3 and dataset.sample_more_highres:
            viewpoint_cam = trainCameras[highresolution_index[randint(0, len(highresolution_index)-1)]]
            
        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)

        Ll1 = l1_loss(image, gt_image)
        if opt.use_lpips_loss:
            lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0,  gt_image.unsqueeze(0)*2.0-1.0).mean()
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
        else:
            ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
            loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)

        depth_loss = 0.0
        if opt.lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth[nan_inf_mask] = 0.0
            gt_depth[nan_inf_mask] = 0.0
            depth_loss += depth_loss_func(gt_depth, depth)

            loss += opt.lambda_depth * depth_loss
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use entropy loss to encourage binary values
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            loss += opt.lambda_opacity * opacity_loss


        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0 and iteration > opt.start_sample_pseudo and iteration < opt.end_sample_pseudo:
            if not pseudo_stack:
                # sample elevation from 80 to 45
                elevation = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (80 - 45) + 45
                # For Satellite
                radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (300 - 250) + 250
                # For GES
                # radius = (opt.end_sample_pseudo - iteration) / (opt.end_sample_pseudo - opt.start_sample_pseudo) * (100 - 50) + 50
                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, elevation, radius, target_std=opt.target_std)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            moge_depth = moge_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(moge_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = min((iteration - opt.start_sample_pseudo) / 500., 1)
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (opt.lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size))

            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    # size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)

                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opt.lambda_opacity = 0.01
                    opacity_cooldown_iter = 500
                    print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")



            if iteration % 100 == 0 and iteration > opt.densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras) # + pseudoCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

@torch.no_grad()
def render_idu_set(views, gaussians, pipeline, background, kernel_size, idu_random_ap=False):
    imgs = []
    for view in tqdm(views, desc="IDU Rendering progress"):
        rendering = render(view, gaussians, pipeline, background, kernel_size=kernel_size, testing=(not idu_random_ap))["render"]
        img = rendering.cpu().numpy().transpose(1, 2, 0)
        imgs.append(img)
    return imgs

@torch.no_grad()
def generate_idu_training_set(
    dataset : ModelParams,
    checkpoint_path : str,
    pipeline : PipelineParams,
    targets, elevation, radius, idu_num_cams, idu_num_samples_per_view, height=512, width=512, fov_x=60.0,
    num_steps: int=50, strength=0.1, guidance_scale=1, eta=0.5,
    use_flow_edit: bool=False, flow_edit_n_min: int=0, flow_edit_n_max: int=15, flow_edit_n_max_end: int=15, flow_edit_n_avg: int=1, model_type: str="FLUX",
    use_difix3d: bool=False, difix3d_model: str="nvidia/difix", difix3d_steps: int=1, 
    use_dreamscene: bool=False, use_sd21: bool=True,
    difix3d_guidance: float=0.0, difix3d_timesteps: list=None, difix3d_use_reference: bool=False,
    difix3d_prompt: str="remove degradation",
    refine=True, idu_no_curriculum=False, idu_random_ap=False
):

    gaussians = GaussianModel(dataset.sh_degree, dataset.appearance_enabled, dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
    print(f"Loading model from checkpoint {checkpoint_path}")
    (model_params, first_iter) = torch.load(checkpoint_path)
    gaussians.load_from_checkpoints(model_params)
    base_dir = os.path.dirname(checkpoint_path)
    print(base_dir)
    scene = Scene(dataset, gaussians, load_iteration=first_iter, shuffle=False, ply_path=base_dir)

    
    # print(gaussians._xyz.shape)
    # # print Gaussian scale statistics
    # gs_scale = gaussians.get_scaling.max(dim=1).values
    # print("Min: ", gs_scale.min().item())
    # print("Max: ", gs_scale.max().item())
    # print("Mean: ", gs_scale.mean().item())
    # print("Std: ", gs_scale.std().item())
    # print("Median: ", gs_scale.median().item())
    # print("Q99: ", gs_scale.kthvalue(int(0.99 * gs_scale.shape[0]), dim=0).values.item())
    
    bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
    kernel_size = dataset.kernel_size

    idu_cam_infos = []
    if isinstance(elevation, list) and isinstance(radius, list):
        assert len(elevation) == len(radius)
        assert idu_no_curriculum, "When using multiple elevations and radii, idu_no_curriculum must be set to True"
        for ele, rad in zip(elevation, radius):
            for target in targets:
                idu_cam_infos += gen_idu_orbit_camera(
                    target,
                    ele,
                    rad,
                    idu_num_cams,
                    idu_num_samples_per_view,
                    height,
                    width,
                    fov_x,
                )
        num_cams = len(idu_cam_infos)
        idu_cam_infos = random.sample(idu_cam_infos, num_cams // len(elevation))
        print("Warning! Sampling a subset of cameras for each elevation/radius pair")
    else:
        for target in targets:
            idu_cam_infos += gen_idu_orbit_camera(
                target,
                elevation,
                radius,
                idu_num_cams,
                idu_num_samples_per_view,
                height,
                width,
                fov_x,
                use_new_id=(not idu_random_ap),
                num_train_cams=(len(scene.getTrainCameras()) if idu_random_ap else None)
            )
    print(f"Generated {len(idu_cam_infos)} IDU cameras")

    cam_lists = cameraList_from_camInfos(idu_cam_infos, 1, dataset, is_pseudo_cam=idu_random_ap)
    imgs = render_idu_set(cam_lists, gaussians, pipeline, background, kernel_size, idu_random_ap)

    # render folder, used to store the unprocessed images
    frames_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render")
    os.makedirs(frames_path, exist_ok=True)
    for idx, img in enumerate(imgs):
        img_path = os.path.join(frames_path, '{0:05d}'.format(idx) + ".png")
        Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)).save(img_path)
    
    # Load 
    refine_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render_refine")
    refine_pipe = None
    
    final_imgs = []
    if refine:
        if use_flow_edit:
            refine_pipe = FlowEditRefineIDU(
                save_path = refine_path,
                device="cuda:0",
                model_type=model_type
            )
            final_imgs = refine_pipe.run(
                imgs,
                n_min=flow_edit_n_min,
                n_max=flow_edit_n_max,
                n_max_end=flow_edit_n_max_end,
                n_avg=flow_edit_n_avg
            )
        elif use_difix3d:
            raise NotImplementedError("Difix3D refine is not yet implemented. Please use FlowEdit refine instead.")
        elif use_dreamscene:
            raise NotImplementedError("DreamScene refine is not yet implemented. Please use FlowEdit refine instead.")
        else:
            raise NotImplementedError("DiffusionSat refine is deprecated")
        if refine_pipe:
            del refine_pipe
        torch.cuda.empty_cache()
    else:   
        for img in imgs:
            # from torch tensor to PIL
            final_imgs.append(Image.fromarray((img * 255 + 0.5).clip(0, 255).astype(np.uint8)))


    depth_path = os.path.join(dataset.model_path, "idu", f"e{elevation}_r{radius}", "render_depth")
    os.makedirs(depth_path, exist_ok=True)
    moge = MoGeIDU(
        depth_path,
        device = "cuda:0",
        fov_x=fov_x
    )
    depths = moge.run(final_imgs)


    final_idu_cam_infos = []
    # Save to cam_infos
    for idx, cam_info in enumerate(idu_cam_infos):
        final_cam_info = CameraInfo(
            uid=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FovY=cam_info.FovY, FovX=cam_info.FovX, 
            cx=0, cy=0,
            image=final_imgs[idx], image_path=cam_info.image_path,
            image_name=cam_info.image_name, 
            depth=depths[idx], mask=None,
            width=cam_info.width, height=cam_info.height
        )
        final_idu_cam_infos.append(final_cam_info)

    final_cam_lists = cameraList_from_camInfos(final_idu_cam_infos, 1, dataset, is_idu=True, is_pseudo_cam=idu_random_ap)
        
    del moge
    del gaussians
    torch.cuda.empty_cache()

    return final_cam_lists

@torch.no_grad()
def generate_pseudo_cams(
    dataset : ModelParams,
    num_cams: int,
    num_train_cams: int,
    elevation: float=80.0,
    radius: float=300.0,
    target_std: float=64.0
):
    idu_cam_infos = []
    for _ in range(num_cams):
        mean = torch.tensor([0., 0.])
        std = torch.tensor([target_std, target_std])
        xy = torch.normal(mean, std)
        z = torch.tensor([0])
        target = torch.cat((xy, z)).tolist()
        gen_cams = gen_idu_orbit_camera(
            target,
            elevation=elevation,
            radius=radius,
            num_cams=12,
            num_samples=1,
            height=1024,
            width=1024,
            fov=60.0,
            use_new_id=False,
            num_train_cams=num_train_cams
        )
        gen_cam = random.choice(gen_cams)
        idu_cam_infos.append(gen_cam)

    print(f"Generated {len(idu_cam_infos)} pseudo cameras with e={elevation:.2f} r={radius:.2f}")

    final_idu_cam_infos = []
    # Save to cam_infos
    for idx, cam_info in enumerate(idu_cam_infos):
        final_cam_info = CameraInfo(
            uid=cam_info.uid, R=cam_info.R, T=cam_info.T, 
            FovY=cam_info.FovY, FovX=cam_info.FovX, 
            cx=0, cy=0,
            image=Image.new("1", (cam_info.width, cam_info.height), (0)), image_path=cam_info.image_path,
            image_name=cam_info.image_name, 
            depth=None, mask=None,
            width=cam_info.width, height=cam_info.height
        )
        final_idu_cam_infos.append(final_cam_info)

    final_cam_lists = cameraList_from_camInfos(final_idu_cam_infos, 1, dataset, is_pseudo_cam=True)
    

    return final_cam_lists

def training_idu_episode(
        dataset, opt, pipe, 
        checkpoint_path,
        targets, elevation, radius, fov,
        idu_num_cams, idu_num_samples_per_view
    ):
    # NOTE: generate pose -> render frame -> refined using DiffusionSat -> use MoGe to predict monocular depth
    if opt.use_lpips_loss:
        lpips_loss_fn = lpips.LPIPS(net=opt.lpips_net)
        for param in lpips_loss_fn.parameters():
            param.requires_grad = False
        lpips_loss_fn.cuda()
        print("Initialized LPIPS loss")
    # Generate IDU training set
    if not opt.idu_no_curriculum:
        assert isinstance(elevation, float) and isinstance(radius, float)
    else:
        assert isinstance(elevation, list) and isinstance(radius, list), "Elevation and radius should be list when no_curriculum is True"
    
    # Validate refinement method selection
    if opt.idu_use_flow_edit and opt.idu_use_difix3d:
        raise ValueError("Cannot use both FlowEdit and Difix3D simultaneously. Please choose one refinement method.")
    
    if opt.idu_refine and not opt.idu_use_flow_edit and not opt.idu_use_difix3d and not opt.idu_use_dreamscene:
        print("Warning: Refinement is enabled but no refinement method is selected. Defaulting to FlowEdit.")
        opt.idu_use_flow_edit = True

    idu_cam_list = generate_idu_training_set(
        dataset,
        checkpoint_path,
        pipe,
        targets, elevation, radius, idu_num_cams, idu_num_samples_per_view, height=opt.idu_render_size, width=opt.idu_render_size, fov_x=fov, # GES: fov_x = 20.0, satellite: 60.0
        num_steps=opt.idu_ddim_step, strength=opt.idu_ddim_strength,
        guidance_scale=opt.idu_ddim_guidance_scale, eta=opt.idu_ddim_eta,
        use_flow_edit=opt.idu_use_flow_edit, flow_edit_n_min=opt.idu_flow_edit_n_min, flow_edit_n_max=opt.idu_flow_edit_n_max, flow_edit_n_max_end=opt.idu_flow_edit_n_max_end, flow_edit_n_avg=opt.idu_flow_edit_n_avg,
        model_type=opt.idu_model_type,
        use_difix3d=opt.idu_use_difix3d, difix3d_model=opt.idu_difix3d_model, difix3d_steps=opt.idu_difix3d_steps,
        difix3d_guidance=opt.idu_difix3d_guidance, difix3d_timesteps=opt.idu_difix3d_timesteps, 
        difix3d_use_reference=opt.idu_difix3d_use_reference, difix3d_prompt=opt.idu_difix3d_prompt,
        use_dreamscene=opt.idu_use_dreamscene, use_sd21=opt.idu_use_sd21,
        refine=opt.idu_refine, idu_no_curriculum=opt.idu_no_curriculum, idu_random_ap=opt.idu_random_ap
    )

    # load Gaussians and scene
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(
        dataset.sh_degree,
        dataset.appearance_enabled,
        dataset.appearance_n_fourier_freqs,
        dataset.appearance_embedding_dim
    )
    scene = Scene(dataset, gaussians)
    # set IDU cameras
    scene.train_idu_cameras[1.0] = idu_cam_list
    gaussians.training_setup(
        opt,
        num_train_cameras=len(scene.getTrainCameras()),
        from_scratch=False  
        # NOTE: set appearacne lr to zero and set the xyz lr scheduler
    )
    if checkpoint_path:
        print(f"Restoring model from checkpoint {checkpoint_path}")
        # original implementation
        (model_params, first_iter) = torch.load(checkpoint_path)
        gaussians.restore(model_params, opt, iterative_datasets_update=True)
        print("Restored model from checkpoint at iteration {}".format(first_iter))
        opt.iterations = first_iter + opt.idu_episode_iterations  # TODO: make this a parameter
        idu_densify_until_iter = first_iter + opt.idu_densify_until_iter
        assert idu_densify_until_iter < opt.iterations
        print(f"Set iterations to {opt.iterations}, densify until {idu_densify_until_iter}")
    else:
        raise ValueError("Checkpoint is required for iterative datasets update")

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    trainCameras = scene.getTrainCameras().copy()
    testCameras = scene.getTestCameras().copy()
    trainIDUCameras = scene.getTrainIDUCameras().copy()
    allCameras = trainCameras + trainIDUCameras + testCameras

    num_train_cams = len(trainCameras)
    
    # highresolution index
    highresolution_index = []
    for index, camera in enumerate(trainCameras):
        if camera.image_width >= 800:
            highresolution_index.append(index)

    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)

    viewpoint_train_stack = None
    viewpoint_train_idu_stack = None
    pseudo_stack = None
    ema_loss_for_log = 0.0
    ema_depth_loss_for_log = 0.0
    ema_opacity_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    testing_iterations = [iter for iter in range(first_iter, opt.iterations + 2, opt.idu_testing_interval)][1:] # skip first iter
    if opt.iterations not in testing_iterations:
        testing_iterations.append(opt.iterations)
    checkpoint_iterations = [opt.iterations]


    checkpoint_path = None

    opacity_cooldown_iter = None
    origin_lambda_opacity = opt.lambda_opacity

    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        if opacity_cooldown_iter is not None:
            if opacity_cooldown_iter > 0:
                opacity_cooldown_iter -= 1
            else:
                opacity_cooldown_iter = None
                opt.lambda_opacity = origin_lambda_opacity
                print(f"Restore lambda opacity to {opt.lambda_opacity}")

        iter_start.record()

        gaussians.update_learning_rate(iteration - first_iter)  # NOTE: modified for IDU

        # Every 1000 its we increase the levels of SH up to a maximum degree
        # if iteration % 1000 == 0:
        #     gaussians.oneupSHdegree()

        # Pick a random Camera
        idu_viewpoint = None

        if iteration + opt.idu_iter_full_train <= opt.iterations and random.random() < opt.idu_train_ratio:
            idu_viewpoint = True
            if not viewpoint_train_idu_stack:
                viewpoint_train_idu_stack = scene.getTrainIDUCameras().copy()
            viewpoint_cam = viewpoint_train_idu_stack.pop(randint(0, len(viewpoint_train_idu_stack)-1))
            lambda_depth = opt.lambda_depth
        else:
            idu_viewpoint = False
            if not viewpoint_train_stack:
                viewpoint_train_stack = scene.getTrainCameras().copy()
            viewpoint_cam = viewpoint_train_stack.pop(randint(0, len(viewpoint_train_stack)-1))
            lambda_depth = 0
        

        #TODO ignore border pixels
        if dataset.ray_jitter:
            subpixel_offset = torch.rand((int(viewpoint_cam.image_height), int(viewpoint_cam.image_width), 2), dtype=torch.float32, device="cuda") - 0.5
            # subpixel_offset *= 0.0
        else:
            subpixel_offset = None

        render_pkg = render(
            viewpoint_cam, 
            gaussians, 
            pipe, 
            background, 
            kernel_size=dataset.kernel_size, 
            subpixel_offset=subpixel_offset,
            testing=(idu_viewpoint and not opt.idu_random_ap)
            # If running iterative datasets update, render image using mean of training embedding
        )
        image, depth, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_depth"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]

        # Loss
        mask = viewpoint_cam.original_mask.cuda()
        gt_image = mask * viewpoint_cam.original_image.cuda()
        gt_depth = mask * viewpoint_cam.original_depth.cuda()

        image = mask * image
        depth = mask * depth
        
        # sample gt_image with subpixel offset
        loss = None
        if dataset.resample_gt_image:
            gt_image = create_offset_gt(gt_image, subpixel_offset)
        if opt.idu_refine or not idu_viewpoint:
            Ll1 = l1_loss(image, gt_image)
            if opt.use_lpips_loss:
                lpips_value = lpips_loss_fn(image.unsqueeze(0)*2.0-1.0,  gt_image.unsqueeze(0)*2.0-1.0).mean()
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * lpips_value
            else:
                ssim_value = fused_ssim(image.unsqueeze(0), gt_image.unsqueeze(0))
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim_value)
        else:
            Ll1 = torch.tensor(0.0)


        depth_loss = 0.0
        if lambda_depth > 0:
            gt_depth = gt_depth.reshape(-1, 1)
            depth = depth.reshape(-1, 1)
            nan_inf_mask = torch.isnan(depth) | torch.isinf(depth) | torch.isnan(gt_depth) | torch.isinf(gt_depth)
            depth = depth[~nan_inf_mask]
            gt_depth = gt_depth[~nan_inf_mask]
            depth_loss += depth_loss_func(gt_depth, depth)
            if torch.isnan(depth_loss).sum() == 0:
                if loss:
                    loss += lambda_depth * depth_loss
                else:
                    loss = lambda_depth * depth_loss
            else:
                depth_loss = 0.0

            # loss += lambda_depth * depth_loss
        if opt.lambda_pseudo_depth > 0 and iteration % opt.sample_pseudo_interval == 0:
            if not pseudo_stack:
                # sample elevation from 80 to 45
                elevation = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (85 - 45) + 45
                # radius = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (300 - 250) + 250
                radius = (first_iter + opt.idu_episode_iterations - iteration) / opt.idu_episode_iterations * (150 - 75) + 75  # For GES

                pseudo_stack = generate_pseudo_cams(dataset, opt.num_pseudo_cams, num_train_cams, elevation, radius)
            
            pseudo_cam = pseudo_stack.pop(randint(0, len(pseudo_stack) - 1))
            render_pkg = render(
                pseudo_cam, 
                gaussians, 
                pipe, 
                background, 
                kernel_size=dataset.kernel_size, 
                subpixel_offset=subpixel_offset
            )
            render_image, render_depth = render_pkg["render"], render_pkg["render_depth"]
            
            render_image_pil = to_pil_image(render_image)
            moge_depth = moge_standalone.run([render_image_pil], pbar=False)[0]
            gt_depth = torch.tensor(moge_depth).to(render_depth.device)

            gt_depth = gt_depth.reshape(-1, 1)
            render_depth = render_depth.reshape(-1, 1)
            depth_loss_pseudo = depth_loss_func(gt_depth, render_depth)

            if torch.isnan(depth_loss_pseudo).sum() == 0:
                loss_scale = 1.0
                loss += loss_scale * opt.lambda_pseudo_depth * depth_loss_pseudo
                depth_loss += depth_loss_pseudo
        
        opacity_loss = 0.0
        if opt.lambda_opacity > 0:
            # Get each gaussians' opacity and use entropy loss to encourage binary values
            opacity = gaussians.get_opacity.clamp(1.0e-3, 1.0 - 1.0e-3)
            opacity_loss = torch.mean(-opacity * torch.log(opacity + 1e-6))
            if loss:
                loss += opt.lambda_opacity * opacity_loss
            else:
                loss = opt.lambda_opacity * opacity_loss

        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            if (lambda_depth > 0 or opt.lambda_pseudo_depth > 0) and not isinstance(depth_loss, float):
                if math.isnan(ema_depth_loss_for_log):
                    ema_depth_loss_for_log = depth_loss.item()
                else:
                    ema_depth_loss_for_log = 0.4 * depth_loss.item() + 0.6 * ema_depth_loss_for_log
            else:
                ema_depth_loss_for_log = 0
            if opt.lambda_opacity > 0:
                ema_opacity_loss_for_log = 0.4 * opacity_loss.item() + 0.6 * ema_opacity_loss_for_log
            else:
                ema_opacity_loss_for_log = 0.6 * ema_opacity_loss_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}", 
                    "Depth Loss": f"{ema_depth_loss_for_log:.{7}f}",
                    "Opacity Loss": f"{ema_opacity_loss_for_log:.{7}f}",
                    "# of GS": f"{gaussians.get_xyz.shape[0]}"
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(
                tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background, dataset.kernel_size),
                iterative_datasets_update=True
            )

            # Densification
            if iteration < idu_densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    print("densification!")
                    size_threshold = opt.size_threshold
                    # size_threshold = None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold)
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)

                if (iteration % opt.opacity_reset_interval == 0 and iteration < opt.iterations - 100) or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
                    opt.lambda_opacity = 0.0
                    opacity_cooldown_iter = opt.idu_opacity_cooling_iterations
                    print(f"Turn off opacity regularization for {opacity_cooldown_iter} iterations")

            if iteration % 100 == 0 and iteration > idu_densify_until_iter:
                if iteration < opt.iterations - 100:
                    # don't update in the end of training
                    gaussians.compute_3D_filter(cameras=trainCameras + trainIDUCameras)
        
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                checkpoint_path = scene.model_path + "/chkpnt" + str(iteration) + ".pth"
                torch.save((gaussians.capture(), iteration), checkpoint_path)
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                
    return checkpoint_path

def training_idu(dataset, opt, pipe, init_checkpoint_path):
    start_checkpoint_path = init_checkpoint_path
    opt.opacity_reset_interval = opt.idu_opacity_reset_interval
    opt.idu_testing_interval = opt.idu_episode_iterations // 4
    opt.idu_position_lr_max_steps = opt.idu_episode_iterations
    # extract idu params
    idu_params: IDUParams = opt.idu_params[opt.datasets_type]
    opt.idu_radius_list = idu_params.radius_list
    opt.idu_elevation_list = idu_params.elevation_list
    opt.idu_fov = idu_params.fov
    print("===== IDU Params =====")
    print(f"Datasets Type: {opt.datasets_type}")
    print(f"Radius List: {opt.idu_radius_list}")
    print(f"Elevation List: {opt.idu_elevation_list}")
    print(f"FOV: {opt.idu_fov}")
    print("======================")
    # generate targets
    x = np.linspace(-opt.idu_grid_width/2, opt.idu_grid_width/2, opt.idu_grid_size+2)
    y = np.linspace(-opt.idu_grid_height/2, opt.idu_grid_height/2, opt.idu_grid_size+2)
    # remove border
    x = x[1:-1]
    y = y[1:-1]
    xx, yy = np.meshgrid(x, y)
    targets = np.stack([xx, yy, np.zeros_like(xx)], axis=-1).reshape(-1, 3).tolist()
    assert len(targets) == opt.idu_grid_size * opt.idu_grid_size
    if not opt.idu_no_curriculum:
        
        for radius, elevation in zip(opt.idu_radius_list, opt.idu_elevation_list):
            print(f"Training IDU episode with elevation {elevation} and radius {radius}")
            print(f"# of IDU targets: {len(targets)}")
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=elevation, radius=radius, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view
            )
    else:
        print("===== Disable IDU curriculum learning =====")
        assert opt.idu_episode_iterations == 10000, "IDU episode iterations should be 10000"
        assert opt.idu_densify_until_iter == 9000, "IDU episode iterations should be 9000"
        for _ in range(5):
            start_checkpoint_path = training_idu_episode(
                dataset, opt, pipe, 
                checkpoint_path=start_checkpoint_path,
                targets=targets, elevation=opt.idu_elevation_list, radius=opt.idu_radius_list, fov=opt.idu_fov,
                idu_num_cams=opt.idu_num_cams,
                idu_num_samples_per_view=opt.idu_num_samples_per_view
            )
        


def depth_loss_func(gt_depth, depth):
    # gt_depth = torch.nan_to_num(gt_depth, nan=0.0, posinf=0.0, neginf=0.0)
    # depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    return (1 - pearson_corrcoef(gt_depth, depth)).mean()
    # return min(
    #     1 - pearson_corrcoef(gt_depth, depth),
    #     1 - pearson_corrcoef(1 / (gt_depth + 200.), depth)
    # )

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

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
    
    if mask is None:
        depth = np.where(depth > 0, depth, np.nan)
    else:
        mask_b = mask[0].detach().cpu().numpy()
        depth = np.where((depth > 0) & mask_b, depth, np.nan)
    
    # Convert to disparity (inverse depth)
    disp = 1 / depth
    
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

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs, iterative_datasets_update=False):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = [{'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()[::4]}]

        if iterative_datasets_update:
            validation_configs.append({'name': 'train_idu', 'cameras' : scene.getTrainIDUCameras()[::3]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    render_pkg = renderFunc(viewpoint, scene.gaussians, *renderArgs, testing=(config['name'] == 'test'))
                    image = torch.clamp(render_pkg["render"], 0.0, 1.0)
                    depth = render_pkg["render_depth"]
                    gt_depth = viewpoint.original_depth.to("cuda")
                    mask = viewpoint.original_mask.cuda()
                    depth = mask * depth
                    gt_depth = mask * gt_depth
                    depth_vis = torch.nan_to_num(depth, nan=0, posinf=0, neginf=0)
                    # Colorize depth
                    colored_depth = colorize_depth_torch(
                        depth_vis,
                    )
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    colored_gt_depth = colorize_depth_torch(
                        mask * gt_depth,
                    )
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images(
                            config['name'] + f"_view_{viewpoint.image_name}/depth_colored",
                            colored_depth[None],  # Add batch dimension
                            global_step=iteration,
                            dataformats='NCHW'
                        )
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                            tb_writer.add_images(config['name'] + "_view_{}/depth".format(viewpoint.image_name), colored_gt_depth[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])       
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[2000, 3050, 7_000, 10000, 15000, 20000, 21000, 22000, 23000, 30_000, 60100, 61000, 62000, 65000, 67500, 70000, 70000, 70100, 71000, 72000, 75000, 77500, 80000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--iterative_datasets_update", action="store_true")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    if not args.iterative_datasets_update:
        training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)
    else:
    # Start running iterative datasets update
        training_idu(lp.extract(args), op.extract(args), pp.extract(args), args.start_checkpoint)
    # All done
    print("\nTraining complete.")
