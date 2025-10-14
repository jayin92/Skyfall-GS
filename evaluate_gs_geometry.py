import os
import torch
import numpy as np
import json
import rasterio
import cv2
from pathlib import Path
from tqdm import tqdm
import argparse
from scipy import ndimage
from sklearn.metrics import mean_absolute_error
import csv
import warnings
import shutil
import utm
import pyproj
import torchvision
import open3d as o3d
import matplotlib.pyplot as plt
import matplotlib.colors as colors


# Import Gaussian Splatting components
from scene import Scene
from gaussian_renderer import render, GaussianModel
from utils.general_utils import safe_state
from argparse import ArgumentParser
from utils.system_utils import searchForMaxIteration
from arguments import ModelParams, PipelineParams, get_combined_args
from osgeo import gdal
gdal.UseExceptions()
import datetime
# Import plyflatten for DSM generation
try:
    from plyflatten import plyflatten
    PLYFLATTEN_AVAILABLE = True
except ImportError:
    print("Warning: plyflatten not available, falling back to manual DSM generation")
    PLYFLATTEN_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings("ignore")


def load_camera_parameters(transforms_json_path):
    """Load camera parameters from transforms.json file."""
    with open(transforms_json_path, 'r') as f:
        transforms = json.load(f)
    
    cameras = []
    for frame in transforms['frames']:
        camera_info = {
            'image_path': frame['file_path'],
            'transform_matrix': np.array(frame['transform_matrix']),
            'fl_x': frame['fl_x'],
            'fl_y': frame['fl_y'],
            'cx': frame['cx'],
            'cy': frame['cy']
        }
        cameras.append(camera_info)
    
    return cameras


def load_enu_origin(enu_origin_path):
    """Load ENU observer origin coordinates (lat, lon, alt)."""
    with open(enu_origin_path, 'r') as f:
        origin = json.load(f)
    return origin  # [latitude, longitude, altitude]


def enu_to_utm_coordinates(points_enu, enu_origin):
    """Convert ENU coordinates to UTM coordinates using ENU observer origin.
    
    Args:
        points_enu: (N, 3) array of points in ENU coordinates
        enu_origin: [lat, lon, alt] of ENU observer origin
    
    Returns:
        points_utm: (N, 3) array of points in UTM coordinates
    """
    if points_enu.shape[0] == 0:
        return points_enu
    
    # Extract origin coordinates
    origin_lat, origin_lon, origin_alt = enu_origin
    # print(f"Using ENU origin: lat={origin_lat}, lon={origin_lon}, alt={origin_alt}")
    
    # Convert origin from lat/lon to UTM
    origin_utm_x, origin_utm_y, utm_zone_number, utm_zone_letter = utm.from_latlon(origin_lat, origin_lon)
    origin_utm_z = origin_alt  # Altitude remains the same
    
    print(f"ENU origin: lat={origin_lat:.6f}, lon={origin_lon:.6f}, alt={origin_alt:.3f}")
    print(f"UTM origin: x={origin_utm_x:.3f}, y={origin_utm_y:.3f}, zone={utm_zone_number}{utm_zone_letter}")
    
    # Transform ENU points to UTM
    points_utm = np.zeros_like(points_enu)
    
    # ENU to UTM transformation:
    # UTM_x = UTM_origin_x + ENU_east
    # UTM_y = UTM_origin_y + ENU_north
    # UTM_z = UTM_origin_z + ENU_up
    
    points_utm[:, 0] = origin_utm_x + points_enu[:, 0]  # East -> UTM X
    points_utm[:, 1] = origin_utm_y + points_enu[:, 1]  # North -> UTM Y
    points_utm[:, 2] = origin_utm_z + points_enu[:, 2]  # Up -> UTM Z
    
    # Debug: Print some transformation statistics (only for first call)
    if hasattr(enu_to_utm_coordinates, '_debug_printed') is False:
        print(f"ENU point sample (first 3 points):")
        for i in range(min(3, points_enu.shape[0])):
            print(f" ENU: [{points_enu[i, 0]:.3f}, {points_enu[i, 1]:.3f}, {points_enu[i, 2]:.3f}]")
            print(f" UTM: [{points_utm[i, 0]:.3f}, {points_utm[i, 1]:.3f}, {points_utm[i, 2]:.3f}]")
        enu_to_utm_coordinates._debug_printed = True
    
    return points_utm


def render_depth_from_camera(camera, gaussians, pipeline, background):
    """Render depth from Gaussian Splatting model using existing camera."""
    with torch.no_grad():
        render_pkg = render(camera, gaussians, pipeline, background, testing=True)
        depth = render_pkg["render_depth"]
        # Apply camera mask and handle invalid values
        if hasattr(camera, 'original_mask') and camera.original_mask is not None:
            depth = camera.original_mask * torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        else:
            depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
        return depth


def depth_to_point_cloud(depth_map, camera, enu_origin=None):
    """
    Convert a depth map to a 3D point cloud in UTM coordinates.
    
    This function assumes the camera pose (camera.R, camera.T) is already
    in the ENU world coordinate system.
    
    Args:
        depth_map (torch.Tensor): Rendered depth map (H, W).
        camera (object): A camera object with the following attributes:
            - R (np.ndarray): (3, 3) Rotation matrix (world-to-camera).
            - T (np.ndarray): (3,) Translation vector (world-to-camera).
            - focal_x (float): Focal length in x.
            - focal_y (float): Focal length in y.
            - cx (float, optional): Principal point x.
            - cy (float, optional): Principal point y.
        enu_origin (list, optional): [lat, lon, alt] for ENU to UTM conversion.
    
    Returns:
        np.ndarray: (N, 3) array of 3D points in UTM coordinates.
    """
    # 1. Prepare inputs
    if hasattr(depth_map, 'cpu'):
        depth_map = depth_map.cpu().numpy()
    
    if depth_map.ndim == 3:
        depth_map = depth_map.squeeze()
    
    height, width = depth_map.shape
    
    # Filter out invalid depth values
    valid_mask = depth_map > 0
    if not np.any(valid_mask):
        return np.empty((0, 3))
    
    # 2. Create pixel coordinate grid
    u, v = np.meshgrid(np.arange(width), np.arange(height))
    
    # 3. Use robust camera intrinsics (use camera's cx/cy if they exist)
    # cx = width / 2
    # cy = height / 2
    cx = camera.cx / 2 * width + width / 2
    cy = camera.cy / 2 * height + height / 2
    print(cx, cy, camera.focal_x, camera.focal_y)
    
    # 4. Vectorized back-projection to camera coordinate system
    # We only need to compute for valid pixels
    z_cam = depth_map[valid_mask]
    u_valid = u[valid_mask]
    v_valid = v[valid_mask]
    
    x_cam = (u_valid - cx) * z_cam / camera.focal_x
    y_cam = (v_valid - cy) * z_cam / camera.focal_y
    
    # Stack into a (N, 3) array of points in camera space
    points_cam = np.stack((x_cam, y_cam, z_cam), axis=-1)
    
    # 5. Vectorized transformation from camera space to ENU world space
    # The formula to go from camera to world is: P_world = R_transpose @ (P_camera - T)
    # This is equivalent to P_world = P_camera @ R + C where C is the camera center
    R_world_to_cam = camera.R
    T_world_to_cam = camera.T
    
    # Points in world = (Points in cam @ Cam-to-World_Rotation) + Camera_Center_in_world
    # R_cam_to_world is R_world_to_cam.T
    # Camera_Center is -R_world_to_cam.T @ T_world_to_cam
    R_cam_to_world = R_world_to_cam.T
    
    # Camera center in world coordinates
    camera_center_enu = -R_cam_to_world @ T_world_to_cam
    
    # Apply the correct transformation
    points_enu = points_cam @ R_cam_to_world + camera_center_enu
    
    # 6. Convert from ENU to UTM coordinates
    if enu_origin is not None:
        print("Converting points from ENU to UTM coordinates...")
        # This function must also be correct
        points_utm = enu_to_utm_coordinates(points_enu, enu_origin)
    else:
        # If no ENU origin, the "ENU" points are the final output
        points_utm = points_enu
    
    return points_utm


def create_dsm_plyflatten_satnerf_style(point_cloud, gt_metadata_path, dsm_resolution=None, radius=1):
    """Create DSM using plyflatten method following SatNeRF's exact approach.
    
    Args:
        point_cloud: (N, 3) array of 3D points in UTM coordinates [easts, norths, alts]
        gt_metadata_path: Path to GT DSM metadata file (.txt)
        dsm_resolution: DSM resolution (if None, uses GT metadata)
        radius: Radius for plyflatten interpolation
    
    Returns:
        dsm: 2D array representing the DSM
    """
    if not PLYFLATTEN_AVAILABLE or point_cloud.shape[0] == 0:
        return create_dsm_manual_satnerf_style(point_cloud, gt_metadata_path, dsm_resolution)
    
    print(f"Point cloud range: x=[{point_cloud[:, 0].min():.3f}, {point_cloud[:, 0].max():.3f}] y=[{point_cloud[:, 1].min():.3f}, {point_cloud[:, 1].max():.3f}]")
    
    # Read GT region of interest (following SatNeRF exactly)
    gt_roi_metadata = np.loadtxt(gt_metadata_path)
    xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
    xsize, ysize = gt_roi_metadata[2], gt_roi_metadata[2]  # SatNeRF uses fixed size
    resolution = gt_roi_metadata[3] if dsm_resolution is None else dsm_resolution
    yoff += ysize * resolution  # SatNeRF's "weird but seems necessary ?" line
    
    print(f"SatNeRF-style DSM parameters:")
    print(f" xoff={xoff:.3f}, yoff={yoff:.3f}")
    print(f" xsize={xsize}, ysize={ysize}")
    print(f" resolution={resolution}")
    
    # Use plyflatten exactly like SatNeRF
    # try:
    dsm = plyflatten(
        point_cloud,  # [easts, norths, alts] format
        xoff,  # No float() casting - keep as numpy types
        yoff,
        resolution,
        int(xsize),
        int(ysize),
        radius=int(radius),
        sigma=np.float32("inf")
    )
    
    print(f"plyflatten succeeded, DSM shape: {dsm.shape}")
    return dsm
    
    # except Exception as e:
    #     print(f"plyflatten failed: {e}, falling back to manual method")
    #     import traceback
    #     traceback.print_exc()
    #     return create_dsm_manual_satnerf_style(point_cloud, gt_metadata_path, dsm_resolution)


def create_dsm_manual_satnerf_style(point_cloud, gt_metadata_path, dsm_resolution=None):
    """Manual DSM creation following SatNeRF's coordinate system."""
    if point_cloud.shape[0] == 0:
        # Read metadata to get DSM size
        gt_roi_metadata = np.loadtxt(gt_metadata_path)
        xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        return np.full((ysize, xsize), np.nan)
    
    # Read GT region of interest
    gt_roi_metadata = np.loadtxt(gt_metadata_path)
    xoff, yoff = gt_roi_metadata[0], gt_roi_metadata[1]
    xsize, ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
    resolution = gt_roi_metadata[3] if dsm_resolution is None else dsm_resolution
    yoff += ysize * resolution  # SatNeRF's adjustment
    
    dsm = np.full((ysize, xsize), np.nan)
    
    # Project points to grid (following SatNeRF's coordinate system)
    easts = point_cloud[:, 0]
    norths = point_cloud[:, 1]
    alts = point_cloud[:, 2]
    
    # Convert to grid indices
    grid_x = ((easts - xoff) / resolution).astype(int)
    grid_y = ((yoff - norths) / resolution).astype(int)  # Note: yoff - norths (SatNeRF style)
    
    # Filter valid indices
    valid_indices = (
        (grid_x >= 0) & (grid_x < xsize) &
        (grid_y >= 0) & (grid_y < ysize)
    )
    
    if np.any(valid_indices):
        grid_x = grid_x[valid_indices]
        grid_y = grid_y[valid_indices]
        grid_z = alts[valid_indices]
        
        # Use maximum height for overlapping pixels
        for gx, gy, gz in zip(grid_x, grid_y, grid_z):
            if np.isnan(dsm[gy, gx]) or gz > dsm[gy, gx]:
                dsm[gy, gx] = gz
    
    return dsm


def create_dsm_manual(point_cloud, dsm_bounds, dsm_resolution):
    """Create DSM using manual rasterization (fallback method)."""
    if point_cloud.shape[0] == 0:
        xmin, xmax, ymin, ymax = dsm_bounds
        dsm_width = int(np.ceil((xmax - xmin) / dsm_resolution))
        dsm_height = int(np.ceil((ymax - ymin) / dsm_resolution))
        return np.full((dsm_height, dsm_width), np.nan)
    
    xmin, xmax, ymin, ymax = dsm_bounds
    dsm_width = int(np.ceil((xmax - xmin) / dsm_resolution))
    dsm_height = int(np.ceil((ymax - ymin) / dsm_resolution))
    
    dsm = np.full((dsm_height, dsm_width), np.nan)
    
    # Project points to grid
    world_x = point_cloud[:, 0]
    world_y = point_cloud[:, 1]
    world_z = point_cloud[:, 2]
    
    # Convert to grid indices
    grid_x = ((world_x - xmin) / dsm_resolution).astype(int)
    grid_y = ((ymax - world_y) / dsm_resolution).astype(int)
    
    # Filter valid indices
    valid_indices = (
        (grid_x >= 0) & (grid_x < dsm_width) &
        (grid_y >= 0) & (grid_y < dsm_height)
    )
    
    if np.any(valid_indices):
        grid_x = grid_x[valid_indices]
        grid_y = grid_y[valid_indices]
        grid_z = world_z[valid_indices]
        
        # Use maximum height for overlapping pixels
        for gx, gy, gz in zip(grid_x, grid_y, grid_z):
            if np.isnan(dsm[gy, gx]) or gz > dsm[gy, gx]:
                dsm[gy, gx] = gz
    
    return dsm


def load_gt_dsm(gt_dsm_path, metadata_path):
    """Load ground truth DSM and its metadata."""
    # Load DSM
    with rasterio.open(gt_dsm_path, 'r') as dataset:
        gt_dsm = dataset.read(1)
        profile = dataset.profile
    
    # Load metadata
    metadata = np.loadtxt(metadata_path)
    xoff, yoff = metadata[0], metadata[1]
    xsize, ysize = int(metadata[2]), int(metadata[2])
    resolution = metadata[3]
    
    # Calculate bounds
    ulx, uly = xoff, yoff + ysize * resolution
    lrx, lry = xoff + xsize * resolution, yoff
    bounds = [ulx, lrx, lry, uly]  # [xmin, xmax, ymin, ymax]
    
    return gt_dsm, bounds, resolution, profile


def register_dsms_dsmr(pred_dsm_path, gt_dsm_path, gt_metadata_path, gt_mask_path=None):
    """Register predicted DSM to ground truth DSM using DSMR shift alignment (like SatNeRF).
    
    Args:
        pred_dsm_path: Path to predicted DSM file
        gt_dsm_path: Path to ground truth DSM file  
        gt_metadata_path: Path to GT DSM metadata (.txt file)
        gt_mask_path: Optional path to water mask
        
    Returns:
        registered_dsm: Registered DSM array
        transform: Registration transform parameters
    """

    
    # Create temporary files with unique identifiers
    unique_identifier = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pred_dsm_crop_path = f"tmp_crop_dsm_to_delete_{unique_identifier}.tif"
    pred_rdsm_path = f"tmp_crop_rdsm_to_delete_{unique_identifier}.tif"
    
    try:
        # Read DSM metadata (following SatNeRF's approach)
        dsm_metadata = np.loadtxt(gt_metadata_path)
        xoff, yoff = dsm_metadata[0], dsm_metadata[1]
        xsize, ysize = int(dsm_metadata[2]), int(dsm_metadata[2])
        resolution = dsm_metadata[3]
        
        # Define projwin for gdal translate (following SatNeRF exactly)
        ulx, uly, lrx, lry = xoff, yoff + ysize * resolution, xoff + xsize * resolution, yoff
        
        # Check predicted DSM bounds before cropping
        ds = gdal.Open(pred_dsm_path)
        if ds is None:
            raise ValueError(f"Cannot open predicted DSM: {pred_dsm_path}")
        
        # Get predicted DSM info
        geotransform = ds.GetGeoTransform()
        pred_width = ds.RasterXSize
        pred_height = ds.RasterYSize
        
        # Calculate predicted DSM bounds
        pred_ulx = geotransform[0]
        pred_uly = geotransform[3]
        pred_lrx = pred_ulx + pred_width * geotransform[1]
        pred_lry = pred_uly + pred_height * geotransform[5]
        
        print(f"Predicted DSM bounds: ULX={pred_ulx:.3f}, ULY={pred_uly:.3f}, LRX={pred_lrx:.3f}, LRY={pred_lry:.3f}")
        print(f"GT DSM crop bounds: ULX={ulx:.3f}, ULY={uly:.3f}, LRX={lrx:.3f}, LRY={lry:.3f}")
        
        # Check if GT bounds are within predicted DSM bounds
        if (ulx < pred_ulx or uly > pred_uly or lrx > pred_lrx or lry < pred_lry):
            print("Warning: GT bounds extend beyond predicted DSM bounds")
            # Adjust crop bounds to intersection
            crop_ulx = max(ulx, pred_ulx)
            crop_uly = min(uly, pred_uly)
            crop_lrx = min(lrx, pred_lrx)
            crop_lry = max(lry, pred_lry)
            
            # Ensure positive dimensions
            if crop_lrx <= crop_ulx or crop_lry >= crop_uly:
                raise ValueError("No overlap between predicted DSM and GT DSM bounds")
            
            print(f"Adjusted crop bounds: ULX={crop_ulx:.3f}, ULY={crop_uly:.3f}, LRX={crop_lrx:.3f}, LRY={crop_lry:.3f}")
            crop_bounds = [crop_ulx, crop_uly, crop_lrx, crop_lry]
        else:
            crop_bounds = [ulx, uly, lrx, lry]
        
        # Perform the crop
        print(f"Cropping predicted DSM...")
        ds_cropped = gdal.Translate(pred_dsm_crop_path, ds, projWin=crop_bounds)
        if ds_cropped is None:
            raise ValueError("GDAL Translate failed - check coordinate systems and bounds")
        
        ds = None
        ds_cropped = None
        
        # Apply water mask if provided (following SatNeRF)
        if gt_mask_path is not None and os.path.exists(gt_mask_path):
            print("Applying water mask...")
            with rasterio.open(gt_mask_path, "r") as f:
                mask = f.read()[0, :, :]
                water_mask = mask.copy()
                water_mask[mask != 9] = 0
                water_mask[mask == 9] = 1
            
            with rasterio.open(pred_dsm_crop_path, "r") as f:
                profile = f.profile
                pred_dsm = f.read()[0, :, :]
            
            with rasterio.open(pred_dsm_crop_path, 'w', **profile) as dst:
                pred_dsm[water_mask.astype(bool)] = np.nan
                dst.write(pred_dsm, 1)
        
        # Try DSMR registration (following SatNeRF's approach)
        fix_xy = False
        try:
            import dsmr
            print("Using DSMR for 3D registration (X, Y, Z alignment)...")
        except ImportError:
            print("Warning: dsmr not found! DSM registration will only use the Z dimension")
            fix_xy = True
        
        if fix_xy:
            # Fallback: simple vertical offset (like our original method)
            print("Using simple vertical offset registration...")
            with rasterio.open(gt_dsm_path, "r") as f:
                gt_dsm = f.read()[0, :, :]
            with rasterio.open(pred_dsm_crop_path, "r") as f:
                profile = f.profile
                pred_dsm = f.read()[0, :, :]
            
            # Calculate vertical offset
            valid_pred = ~np.isnan(pred_dsm)
            valid_gt = ~np.isnan(gt_dsm)
            valid_both = valid_pred & valid_gt
            
            if np.sum(valid_both) > 0:
                offset = np.nanmean(gt_dsm[valid_both] - pred_dsm[valid_both])
                pred_rdsm = pred_dsm + offset
                transform = (0, 0, offset)  # (dx, dy, dz)
            else:
                pred_rdsm = pred_dsm
                transform = (0, 0, 0)
            
            # Save registered DSM
            with rasterio.open(pred_rdsm_path, 'w', **profile) as dst:
                dst.write(pred_rdsm, 1)
                
        else:
            # Use DSMR for full 3D registration
            print("Computing 3D shift using DSMR...")
            transform = dsmr.compute_shift(gt_dsm_path, pred_dsm_crop_path, scaling=False)
            print(f"DSMR transform: dx={transform[0]:.3f}, dy={transform[1]:.3f}, dz={transform[2]:.3f}")
            
            # Apply the computed shift
            dsmr.apply_shift(pred_dsm_crop_path, pred_rdsm_path, *transform)
            
            # Load registered DSM
            with rasterio.open(pred_rdsm_path, "r") as f:
                pred_rdsm = f.read()[0, :, :]
        
        return pred_rdsm, transform
        
    finally:
        pass
        # Clean up temporary files
        # for temp_file in [pred_dsm_crop_path, pred_rdsm_path]:
        #     if os.path.exists(temp_file):
        #         os.remove(temp_file)

def register_dsms_simple(pred_dsm, gt_dsm):
    """Simple DSM registration using vertical offset only (fallback method)."""
    # Find valid pixels in both DSMs
    if pred_dsm.ndim == 3 and pred_dsm.shape[2] == 1:
        pred_dsm = pred_dsm.squeeze(axis=2)
    valid_pred = ~np.isnan(pred_dsm)
    valid_gt = ~np.isnan(gt_dsm)
    valid_both = valid_pred & valid_gt
    
    if np.sum(valid_both) == 0:
        return pred_dsm, 0.0
    
    # Calculate vertical offset
    pred_valid = pred_dsm[valid_both]
    gt_valid = gt_dsm[valid_both]
    
    offset = np.nanmean(gt_valid - pred_valid)
    registered_dsm = pred_dsm + offset
    
    return registered_dsm, offset


def compute_dsm_metrics(pred_dsm, gt_dsm, mask=None):
    """Compute DSM evaluation metrics."""
    # Apply mask if provided
    if mask is not None:
        pred_dsm = pred_dsm.copy()
        gt_dsm = gt_dsm.copy()
        pred_dsm[~mask] = np.nan
        gt_dsm[~mask] = np.nan
    
    # Find valid pixels
    valid_pred = ~np.isnan(pred_dsm)
    valid_gt = ~np.isnan(gt_dsm)
    valid_both = valid_pred & valid_gt
    
    if np.sum(valid_both) == 0:
        return {
            'mae': np.nan,
            'rmse': np.nan,
            'valid_pixels': 0,
            'completeness': 0.0
        }
    
    pred_valid = pred_dsm[valid_both].flatten()
    gt_valid = gt_dsm[valid_both].flatten()
    
    # Compute metrics
    mae = mean_absolute_error(gt_valid, pred_valid)
    rmse = np.sqrt(np.mean((pred_valid - gt_valid) ** 2))
    completeness = np.sum(valid_both) / np.sum(valid_gt) if np.sum(valid_gt) > 0 else 0.0
    
    return {
        'mae': mae,
        'rmse': rmse,
        'valid_pixels': np.sum(valid_both),
        'completeness': completeness
    }


def load_water_mask(mask_path, aoi_id):
    """Load water mask from classification file."""
    # Determine correct mask file
    if aoi_id in ["JAX_004", "JAX_260"]:
        mask_file = f"{aoi_id}_CLS_v2.tif"
    else:
        mask_file = f"{aoi_id}_CLS.tif"
    
    full_mask_path = os.path.join(mask_path, mask_file)
    
    if not os.path.exists(full_mask_path):
        return None
    
    with rasterio.open(full_mask_path, 'r') as dataset:
        mask = dataset.read(1)
    
    # Water pixels have value 9, create binary mask (True for non-water)
    water_mask = mask != 9
    
    return water_mask


def evaluate_scene(dataset, pipeline, scene_name, satellite_data_path, gt_data_path, camera_path=None, iteration=-1, load_from_checkpoints=False):
    """Evaluate geometry accuracy for a single scene."""
    print(f"\nEvaluating scene: {scene_name}")
    
    # Override dataset source path for this scene
    dataset.source_path = os.path.join(satellite_data_path, scene_name, "outputs_skew")
    
    # Load ENU origin coordinates
    # Look for ENU origin in the original data directory structure
    base_scene_path = os.path.join(satellite_data_path, scene_name)
    enu_origin_path = os.path.join(base_scene_path, "outputs_srtm", "enu_observer_latlonalt.json")
    
    # Fallback paths
    if not os.path.exists(enu_origin_path):
        enu_origin_path = os.path.join(base_scene_path, "outputs_skew", "enu_observer_latlonalt.json")
        if not os.path.exists(enu_origin_path):
            enu_origin_path = os.path.join(dataset.source_path, "enu_observer_latlonalt.json")
    
    enu_origin = None
    if os.path.exists(enu_origin_path):
        enu_origin = load_enu_origin(enu_origin_path)
        print(f"Loaded ENU origin from {enu_origin_path}")
        print(f"ENU observer origin: lat={enu_origin[0]:.6f}, lon={enu_origin[1]:.6f}, alt={enu_origin[2]:.3f}")
    else:
        print(f"Warning: ENU origin not found at {enu_origin_path}")
        print("Assuming points are already in UTM coordinates")
    
    # Find iteration if not specified
    if iteration == -1:
        iteration = searchForMaxIteration(os.path.join(dataset.model_path, "point_cloud"))
    
    # Load Gaussian model
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.appearance_enabled, dataset.appearance_n_fourier_freqs, dataset.appearance_embedding_dim)
    
        # Load from checkpoint if specified
        if load_from_checkpoints:
            checkpoint_path = os.path.join(dataset.model_path, f"chkpnt{iteration}.pth")
            print(f"Loading model from checkpoint {checkpoint_path}")
            if os.path.exists(checkpoint_path):
                (model_params, first_iter) = torch.load(checkpoint_path)
                gaussians.load_from_checkpoints(model_params)
            else:
                print(f"Checkpoint not found: {checkpoint_path}")
                return None
        
        # Print Gaussian statistics
        print(f"Gaussians shape: {gaussians._xyz.shape}")
        gs_scale = gaussians.get_scaling.max(dim=1).values
        print(f"Scale - Min: {gs_scale.min().item():.6f}, Max: {gs_scale.max().item():.6f}, Mean: {gs_scale.mean().item():.6f}")
        
        # Load scene
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        
        # Setup background and pipeline
        scale_factor = dataset.resolution
        bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        kernel_size = dataset.kernel_size

        # gaussians.prune_by_radius(500)
        
        # Get cameras - use camera path if provided, otherwise use training cameras
        if camera_path is not None and os.path.exists(camera_path):
            print(f"Loading camera trajectory from: {camera_path}")
            # Import the function from render_video.py
            from render_video import get_path_from_json
            from utils.camera_utils import cameraList_from_camInfos
            
            # Read camera path json (following render_video.py pattern)
            with open(camera_path, 'r') as file:
                camera_path_data = json.load(file)
            
            # Get camera info list from JSON
            cam_infos, radius = get_path_from_json(camera_path_data)
            
            # Convert to camera objects
            test_cameras = cameraList_from_camInfos(cam_infos, 1, dataset, is_testing=True)
            
            print(f"Loaded {len(test_cameras)} cameras from trajectory file")
        else:
            # Fallback to training cameras
            print("Using training cameras (no camera path provided)")
            test_cameras = scene.getTrainCameras()
            if len(test_cameras) == 0:
                test_cameras = scene.getTestCameras()
        
        if len(test_cameras) == 0:
            print(f"No cameras found for {scene_name}")
            return None
        
        # Load ground truth DSM
        aoi_id = scene_name
        gt_dsm_path = os.path.join(gt_data_path, "Track3-Truth", f"{aoi_id}_DSM.tif")
        gt_metadata_path = os.path.join(gt_data_path, "Track3-Truth", f"{aoi_id}_DSM.txt")
        
        if not os.path.exists(gt_dsm_path) or not os.path.exists(gt_metadata_path):
            print(f"Ground truth DSM not found for {scene_name}")
            return None
        
        gt_dsm, dsm_bounds, dsm_resolution, dsm_profile = load_gt_dsm(gt_dsm_path, gt_metadata_path)
        
        # Debug: Print GT DSM info
        print(f"Ground truth DSM info:")
        print(f" Shape: {gt_dsm.shape}")
        print(f" Valid pixels: {np.sum(~np.isnan(gt_dsm))}/{gt_dsm.size}")
        print(f" Height range: [{np.nanmin(gt_dsm):.3f}, {np.nanmax(gt_dsm):.3f}]")
        print(f" Resolution: {dsm_resolution}m")
        
        # Load water mask
        water_mask = load_water_mask(os.path.join(gt_data_path, "Track3-Truth"), aoi_id)
        
        # Render depth from multiple views and create point clouds
        all_point_clouds = []
        valid_renders = 0
        
        # Use all cameras if from camera path, or limit to 5 if from training cameras
        if camera_path is not None:
            holdout = len(test_cameras) // 24
            cameras_to_use = test_cameras[::holdout]  # Use all cameras from trajectory
            print(f"Using all {len(cameras_to_use)} cameras from trajectory")
        else:
            max_cameras = min(5, len(test_cameras))  # Limit training cameras
            cameras_to_use = test_cameras[:max_cameras]
            print(f"Using {len(cameras_to_use)} training cameras")
        
        for camera in tqdm(cameras_to_use, desc=f"Rendering {scene_name}"):
            try:
                # Render depth
                render_pkg = render(camera, gaussians, pipeline, background, kernel_size=kernel_size, testing=True)
                depth = render_pkg["render_depth"]
                rgb = render_pkg["render"]
                
                # Apply mask and handle invalid values
                depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
                # Apply near far clipping, < min depth -> 0, > max depth -> 0
                # depth = torch.where(depth < 0, torch.tensor(0.0, device=depth.device), depth)
                # depth = torch.where(depth > 1000, torch.tensor(0.0, device=depth.device), depth)
                # center crop 512 x 512
                # if depth.shape[0] > 512 or depth.shape[1] > 512:
                #     depth = torchvision.transforms.functional.center_crop(depth, (512, 512))
                #     rgb = torchvision.transforms.functional.center_crop(rgb, (512, 512))
                from render_video import colorize_depth_torch
                depth_vis = colorize_depth_torch(depth)

                torchvision.utils.save_image(depth_vis, f"{dataset.model_path}/depth_{camera.image_name}.png")
                torchvision.utils.save_image(rgb, f"{dataset.model_path}/rgb_{camera.image_name}.png")
                
                if torch.any(depth > 0):
                    # Convert depth to 3D point cloud in UTM coordinates
                    points_utm = depth_to_point_cloud(depth, camera, enu_origin)
                    
                    if points_utm.shape[0] > 0:
                        all_point_clouds.append(points_utm)
                        valid_renders += 1
                        print(f" Camera {valid_renders}: {points_utm.shape[0]} points")
            
            except Exception as e:
                print(f"Error rendering camera: {e}")
                continue
    
    if len(all_point_clouds) == 0:
        print(f"No valid renders for {scene_name}")
        return None
    
    # Merge all point clouds
    merged_point_cloud = np.vstack(all_point_clouds)
    print(f"Total merged points: {merged_point_cloud.shape[0]}")

    # save ply file
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(merged_point_cloud)
    ply_path = os.path.join(dataset.model_path, f"{scene_name}_geometry_acc_point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, pcd)
    print(f"Saved merged point cloud to {ply_path}")
    
    # Debug: Print point cloud statistics
    print(f"Point cloud bounds:")
    print(f" X: [{merged_point_cloud[:, 0].min():.3f}, {merged_point_cloud[:, 0].max():.3f}]")
    print(f" Y: [{merged_point_cloud[:, 1].min():.3f}, {merged_point_cloud[:, 1].max():.3f}]")
    print(f" Z: [{merged_point_cloud[:, 2].min():.3f}, {merged_point_cloud[:, 2].max():.3f}]")
    
    print(f"Ground truth DSM bounds: [{dsm_bounds[0]:.3f}, {dsm_bounds[1]:.3f}, {dsm_bounds[2]:.3f}, {dsm_bounds[3]:.3f}]")
    
    # Create DSM using plyflatten (SatNeRF style)
    pred_dsm = create_dsm_plyflatten_satnerf_style(merged_point_cloud, gt_metadata_path, dsm_resolution, radius=1.0)
    
    # Save predicted DSM to temporary file for DSMR registration
    import tempfile
    import datetime
    unique_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    temp_pred_dsm_path = f"temp_pred_dsm_{scene_name}_{unique_id}.tif"
    
    try:
        # Create DSM profile following SatNeRF's exact approach (lines 324-337)
        # Read GT DSM metadata to get proper parameters
        gt_roi_metadata = np.loadtxt(gt_metadata_path)
        gt_xoff, gt_yoff = gt_roi_metadata[0], gt_roi_metadata[1]
        gt_xsize, gt_ysize = int(gt_roi_metadata[2]), int(gt_roi_metadata[2])
        gt_resolution = gt_roi_metadata[3]
        
        # SatNeRF's coordinate adjustment
        gt_yoff += gt_ysize * gt_resolution
        
        # Get UTM zone info for CRS (following SatNeRF approach)
        # We need some sample coordinates to determine UTM zone
        sample_points = merged_point_cloud[:10]  # Use first 10 points
        if len(sample_points) > 0:
            # Convert sample point from UTM back to lat/lon to get UTM zone
            # For now, assume we're in UTM zone 17R (Florida) like JAX scenes
            import utm
            from plyflatten.utils import rasterio_crs, crs_proj
            
            # Use known UTM zone for JAX scenes (could be improved by reverse lookup)
            utm_zone = "17R"  # JAX scenes are in UTM zone 17R
            crs_proj_utm = rasterio_crs(crs_proj(utm_zone, crs_type="UTM"))
        else:
            # Fallback CRS
            crs_proj_utm = None
        
        # Create profile exactly like SatNeRF (lines 326-334)
        profile = {}
        profile["dtype"] = pred_dsm.dtype
        profile["height"] = pred_dsm.shape[0]
        profile["width"] = pred_dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        if crs_proj_utm is not None:
            profile["crs"] = crs_proj_utm
        
        # Create affine transform exactly like SatNeRF (line 334)
        import affine
        profile["transform"] = affine.Affine(gt_resolution, 0.0, gt_xoff, 0.0, -gt_resolution, gt_yoff)
        
        print(f"Created SatNeRF-style profile:")
        print(f"  dtype: {profile['dtype']}")
        print(f"  shape: ({profile['height']}, {profile['width']})")
        print(f"  resolution: {gt_resolution}m")
        print(f"  transform: {profile['transform']}")
        
        # Save predicted DSM as GeoTIFF (following SatNeRF line 335-336)
        with rasterio.open(temp_pred_dsm_path, 'w', **profile) as pred_file:
            if pred_dsm.ndim == 3:
                pred_file.write(pred_dsm[:, :, 0], 1)
            else:
                pred_file.write(pred_dsm, 1)
        
        print(f"Saved temporary predicted DSM: {temp_pred_dsm_path}")
        
        # Get water mask path for DSMR registration
        if aoi_id in ["JAX_004", "JAX_260"]:
            water_mask_path  = os.path.join(gt_data_path, "Track3-Truth", f"{aoi_id}_CLS_v2.tif")
        else:
            water_mask_path = os.path.join(gt_data_path, "Track3-Truth", f"{aoi_id}_CLS.tif")
        
        if not os.path.exists(water_mask_path):
            water_mask_path = None
        
        # Register DSMs using DSMR (like SatNeRF)
        print("Registering DSMs using DSMR-based alignment...")
        pred_dsm_registered, transform = register_dsms_dsmr(
            temp_pred_dsm_path, 
            gt_dsm_path, 
            gt_metadata_path, 
            water_mask_path
        )
        
        # Extract registration info
        if len(transform) >= 3:
            dx, dy, dz = transform[0], transform[1], transform[2]
            print(f"DSMR registration: dx={dx:.3f}m, dy={dy:.3f}m, dz={dz:.3f}m")
        else:
            dx, dy, dz = 0, 0, transform if isinstance(transform, (int, float)) else 0
            
    except Exception as e:
        print(f"DSMR registration failed: {e}")
        print("Falling back to simple vertical offset registration...")
        pred_dsm_registered, dz = register_dsms_simple(pred_dsm, gt_dsm)
        dx, dy = 0, 0
        
    finally:
        pass
        # Clean up temporary file
        # if os.path.exists(temp_pred_dsm_path):
        #     os.remove(temp_pred_dsm_path)
        #     print(f"Cleaned up temporary file: {temp_pred_dsm_path}")
    
    # Compute metrics
    metrics = compute_dsm_metrics(pred_dsm_registered, gt_dsm, mask=water_mask)
    metrics['dx_offset'] = dx
    metrics['dy_offset'] = dy  
    metrics['dz_offset'] = dz
    metrics['valid_renders'] = valid_renders
    metrics['total_points'] = merged_point_cloud.shape[0]
    
    print(f"Results for {scene_name}:")
    print(f" MAE: {metrics['mae']:.3f} m")
    print(f" RMSE: {metrics['rmse']:.3f} m")
    print(f" Completeness: {metrics['completeness']:.3f}")
    print(f" Registration offset: dx={metrics['dx_offset']:.3f}m, dy={metrics['dy_offset']:.3f}m, dz={metrics['dz_offset']:.3f}m")
    print(f" Valid renders: {metrics['valid_renders']}")
    print(f" Total points: {metrics['total_points']}")
    
    return {
        'scene': scene_name,
        **metrics
    }


def main():
    # Set up command line argument parser (following render_video.py pattern)
    parser = ArgumentParser(description="Evaluate Gaussian Splatting geometry accuracy")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--load_from_checkpoints", action="store_true")
    # parser.add_argument("--model_root_path", type=str, required=True, help="Path to the Gaussian Splatting model")
    # parser.add_argument("--postfix", type=str, default="", help="Postfix for model path")
    parser.add_argument("--satellite_data_path", type=str,
                        default="/project/jayinnn/SatelliteSfM/data/DFC2019_processed",
                        help="Path to satellite data with camera parameters")
    parser.add_argument("--gt_data_path", type=str,
                        default="/project/jayinnn/mip-splatting/DFC2019",
                        help="Path to ground truth DSM data")
    parser.add_argument("--scenes", type=str, nargs='+',
                        default=["JAX_004", "JAX_068", "JAX_214", "JAX_260"],
                        help="List of scenes to evaluate")
    parser.add_argument("--camera_path", type=str, default="camera_path/r400_e90_fov60.json",
                        help="Path to camera trajectory JSON file (like render_video.py)")
    parser.add_argument("--output_file", type=str, default="gs_geometry_results.csv",
                        help="Output CSV file for results")
    
    args = get_combined_args(parser)
    
    # Handle camera_path argument (add default if missing)
    if not hasattr(args, 'camera_path'):
        args.camera_path = None
    
    # Initialize system state (RNG) - following render_video.py pattern
    safe_state(args.quiet)
    
    # Ensure CUDA is available
    if not torch.cuda.is_available():
        print("CUDA not available. This script requires GPU support.")
        return
    
    results = []
    
    for scene in args.scenes:
        # args.model_path = os.path.join(args.model_root_path, scene+args.postfix)
        print("Evaluating geometry for " + args.model_path)
        # print(args.model_path)
        try:
            result = evaluate_scene(
                model.extract(args),
                pipeline.extract(args),
                scene,
                args.satellite_data_path,
                args.gt_data_path,
                args.camera_path,
                args.iteration,
                args.load_from_checkpoints
            )
            
            if result is not None:
                results.append(result)
        
        except Exception as e:
            print(f"Error evaluating scene {scene}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save results
    if results:
        fieldnames = ['scene', 'mae', 'rmse', 'completeness', 'valid_pixels',
                      'dx_offset', 'dy_offset', 'dz_offset', 'valid_renders', 'total_points']
        
        with open(args.output_file, 'w', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to {args.output_file}")
        
        # Print summary statistics
        print("\nSummary Statistics:")
        if len(results) > 1:
            avg_mae = np.mean([r['mae'] for r in results if not np.isnan(r['mae'])])
            avg_rmse = np.mean([r['rmse'] for r in results if not np.isnan(r['rmse'])])
            avg_completeness = np.mean([r['completeness'] for r in results])
            
            print(f"Average MAE: {avg_mae:.3f} m")
            print(f"Average RMSE: {avg_rmse:.3f} m")
            print(f"Average Completeness: {avg_completeness:.3f}")
    else:
        print("No successful evaluations completed.")


if __name__ == "__main__":
    main()