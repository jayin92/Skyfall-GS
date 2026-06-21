"""
Evaluation script for integrated models.
Calculates CLIP-FID, CMMD, PSNR, SSIM, and LPIPS between GT images and method outputs.

Usage:
python eval.py \
    --data_dir results_eval/data_eval_JAX \
    --temp_dir temp_frames_JAX \
    --methods mip-splatting sat-nerf eogs corgs ours_stage1 ours_stage2 \
    --output_file metrics_results_JAX.csv \
    --frame_rate 30 \
    --resolution 1024 \
    --batch_size 64 

python eval.py \
    --data_dir results_eval/data_eval_NYC \
    --temp_dir temp_frames_NYC \
    --methods citydreamer gaussiancity corgs ours_stage1 ours_stage2 \
    --output_file metrics_results_NYC.csv \
    --frame_rate 24 \
    --no_resize \
    --batch_size 64
"""

import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import argparse
from cleanfid import fid
import shutil
import csv
import warnings
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from cmmd_pytorch.main import compute_cmmd
import pyiqa

# Suppress warnings
warnings.filterwarnings("ignore")

def patchify(image, patch_size, stride=None, min_patches=None):
    """Split an image into overlapping patches of the specified size.

    Args:
        image: Input image as numpy array (height, width, channels)
        patch_size: Tuple of (patch_height, patch_width) or single integer for square patches
        stride: Step size between patches (defaults to 1 for maximum overlap)
        min_patches: Tuple of (min_h_patches, min_w_patches) to ensure minimum coverage

    Returns:
        List of image patches as numpy arrays
    """
    height, width = image.shape[:2]

    # Handle case where patch_size is a single integer (square patches)
    if isinstance(patch_size, int):
        patch_height = patch_width = patch_size
    else:
        patch_height, patch_width = patch_size

    # Calculate stride based on min_patches if provided
    if min_patches is not None:
        min_h_patches, min_w_patches = min_patches

        # Calculate required stride to achieve minimum number of patches
        h_stride = max(1, (height - patch_height) // max(min_h_patches - 1, 1))
        w_stride = max(1, (width - patch_width) // max(min_w_patches - 1, 1))

        # Use the calculated stride
        stride = min(h_stride, w_stride)
    elif stride is None:
        stride = 1

    # Calculate number of patches in each dimension
    num_patches_h = max(1, (height - patch_height) // stride + 1)
    num_patches_w = max(1, (width - patch_width) // stride + 1)

    # For edge cases when the image is smaller than the requested number of patches
    if min_patches is not None:
        min_h_patches, min_w_patches = min_patches
        if num_patches_h < min_h_patches or num_patches_w < min_w_patches:
            # Recalculate strides to force the minimum number of patches
            h_gaps = max(min_h_patches - 1, 1)
            w_gaps = max(min_w_patches - 1, 1)

            h_stride = (height - patch_height) / h_gaps if h_gaps > 0 else 0
            w_stride = (width - patch_width) / w_gaps if w_gaps > 0 else 0

            # Get patch start positions
            h_positions = [int(i * h_stride) for i in range(min_h_patches)]
            w_positions = [int(i * w_stride) for i in range(min_w_patches)]

            # Extract patches at calculated positions
            patches = []
            for y_start in h_positions:
                for x_start in w_positions:
                    y_end = min(y_start + patch_height, height)
                    x_end = min(x_start + patch_width, width)

                    # Adjust start positions to ensure patch size is maintained
                    y_start = max(0, y_end - patch_height)
                    x_start = max(0, x_end - patch_width)

                    patch = image[y_start:y_end, x_start:x_end]

                    # Only add if patch is the right size
                    if patch.shape[0] == patch_height and patch.shape[1] == patch_width:
                        patches.append(patch)

            return patches

    # Initialize list to store patches
    patches = []

    # Extract each patch
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Calculate patch coordinates
            y_start = i * stride
            y_end = y_start + patch_height
            x_start = j * stride
            x_end = x_start + patch_width

            # Make sure we don't go out of bounds
            if y_end <= height and x_end <= width:
                # Extract the patch
                patch = image[y_start:y_end, x_start:x_end]
                patches.append(patch)

    return patches

def extract_frames(video_path, output_dir, frame_rate=1, prefix="", resolution=None):
    """Extract frames from a video file"""
    os.makedirs(output_dir, exist_ok=True)

    # Open the video file
    video = cv2.VideoCapture(str(video_path))

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"FPS: {fps}, Total Frames: {total_frames}")

    indices = np.linspace(0, total_frames - 1, frame_rate, endpoint=False, dtype=int)
    
    frame_count = 0
    for idx in indices:
        video.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = video.read()
        if ret:
            if resolution is not None:
                frame = cv2.resize(frame, (resolution, resolution))
            cv2.imwrite(os.path.join(output_dir, f"{prefix}frame_{frame_count:05d}.png"), frame)
            frame_count += 1

    # Release the video file
    video.release()

    return frame_count

def extract_uniform_frames(image_folder, num_frames=24):
    """Extract uniformly distributed frames from an image folder."""
    # Get all image files and sort them
    image_files = sorted([f for f in os.listdir(image_folder) if f.endswith(('.jpg', '.png', '.jpeg'))])
    total_frames = len(image_files)

    assert total_frames == num_frames, f"Expected {num_frames} frames, but found {total_frames} frames in {image_folder}"

    return [os.path.join(image_folder, f) for f in image_files]

def extract_frames_from_video(video_path, num_frames=24):
    """Extract uniformly distributed frames from a video."""
    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video path: {str(video_path)}, Total frames in video: {total_frames}")

    if total_frames <= num_frames:
        # Extract all frames
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()
        return frames

    # Calculate frame indices for uniform sampling
    indices = np.linspace(0, total_frames - 1, num_frames, endpoint=False, dtype=int)

    frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            frames.append(frame)

    cap.release()
    return frames

def process_single_image(args):
    """Process a single image and save its patches"""
    img_path, output_dir, patch_size, min_patches = args

    img_name = os.path.basename(img_path)
    frame = cv2.imread(img_path)

    if frame is None:
        return 0

    patches = patchify(frame, patch_size=patch_size, min_patches=min_patches)

    saved_count = 0
    for patch_idx, patch in enumerate(patches):
        frame_path = os.path.join(output_dir, f"{img_name}_{patch_idx:02d}.png")
        success = cv2.imwrite(frame_path, patch)
        if success:
            saved_count += 1

    return saved_count

def patchify_parallel(input_dir, output_dir, patch_size=512, min_patches=(9, 16), max_workers=None):
    """Parallelize the patchify operation for all images in a directory"""
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files
    img_files = [f for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    img_paths = [os.path.join(input_dir, f) for f in img_files]

    # Prepare arguments for parallel processing
    args_list = [(img_path, output_dir, patch_size, min_patches) for img_path in img_paths]

    if max_workers is None:
        max_workers = min(mp.cpu_count(), len(img_files))

    total_patches_saved = 0

    # Process images in parallel
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all jobs
        future_to_img = {executor.submit(process_single_image, args): args[0] for args in args_list}

        # Collect results with progress bar
        for future in tqdm(as_completed(future_to_img), total=len(img_files), desc=f"Patchifying {os.path.basename(input_dir)}"):
            try:
                patches_saved = future.result()
                total_patches_saved += patches_saved
            except Exception as e:
                img_path = future_to_img[future]
                print(f"Error processing {img_path}: {e}")

    print(f"Saved {total_patches_saved} patches to {output_dir}")
    return total_patches_saved

def preprocess_image_for_iqa(img_bgr, target_size=None):
    """Convert BGR image to RGB tensor for PyTorch IQA."""
    # Convert BGR to RGB
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Resize if target size specified
    if target_size is not None:
        img_rgb = cv2.resize(img_rgb, (target_size[1], target_size[0]))

    # Convert to PIL Image and then to tensor
    img_pil = Image.fromarray(img_rgb)

    # Convert to tensor and normalize to [0,1]
    transform = transforms.Compose([transforms.ToTensor()])
    img_tensor = transform(img_pil).unsqueeze(0)  # Add batch dimension
    return img_tensor

class IntegratedIQACalculator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device

        # Initialize PyTorch IQA metrics (only the ones we need)
        self.metrics = {}

        # Reference-based metrics
        try:
            self.metrics['psnr'] = pyiqa.create_metric('psnr', device=device)
            print("✓ PSNR metric loaded")
        except Exception as e:
            print(f"✗ Failed to load PSNR: {e}")

        try:
            self.metrics['ssim'] = pyiqa.create_metric('ssim', device=device)
            print("✓ SSIM metric loaded")
        except Exception as e:
            print(f"✗ Failed to load SSIM: {e}")

        try:
            self.metrics['lpips'] = pyiqa.create_metric('lpips', device=device)
            print("✓ LPIPS metric loaded")
        except Exception as e:
            print(f"✗ Failed to load LPIPS: {e}")

    def calculate_reference_metrics(self, img1, img2):
        """Calculate reference-based metrics (PSNR, SSIM, LPIPS)."""
        results = {}

        # Ensure images have the same size
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))

        # Convert to tensors
        img1_tensor = preprocess_image_for_iqa(img1).to(self.device)
        img2_tensor = preprocess_image_for_iqa(img2).to(self.device)

        # Calculate reference-based metrics
        for metric_name in ['psnr', 'ssim', 'lpips']:
            if metric_name in self.metrics:
                try:
                    with torch.no_grad():
                        score = self.metrics[metric_name](img1_tensor, img2_tensor)
                    results[metric_name] = score.item()
                except Exception as e:
                    print(f"Error calculating {metric_name}: {e}")
                    results[metric_name] = None
            else:
                results[metric_name] = None

        return results

def calculate_clip_fid_cmmd(gt_dir, test_dir, use_patchify=True, max_workers=None, batch_size=64):
    """Calculate CLIP-FID and CMMD with optional patchification"""

    # Patchify the images if requested
    if use_patchify:
        new_gt_dir = gt_dir + "_patchified"
        new_test_dir = test_dir + "_patchified"

        if not os.path.exists(new_gt_dir):
            print("Parallel patchifying GT images...")
            patchify_parallel(gt_dir, new_gt_dir, patch_size=512, min_patches=(9, 16), max_workers=max_workers)

        if not os.path.exists(new_test_dir):
            print("Parallel patchifying test images...")
            patchify_parallel(test_dir, new_test_dir, patch_size=512, min_patches=(9, 16), max_workers=max_workers)

        gt_dir = new_gt_dir
        test_dir = new_test_dir

    # Calculate CLIP-FID
    try:
        print("Calculating CLIP-FID...")
        clip_fid_score = fid.compute_fid(gt_dir, test_dir, mode="clean", model_name="clip_vit_b_32")
    except Exception as e:
        print(f"Error calculating CLIP-FID: {e}")
        clip_fid_score = None

    # Calculate CMMD
    try:
        print("Calculating CMMD...")
        cmmd_score = compute_cmmd(gt_dir, test_dir, batch_size=batch_size)
    except Exception as e:
        print(f"Error calculating CMMD: {e}")
        cmmd_score = None

    return clip_fid_score, cmmd_score

def evaluate_scene_method_integrated(gt_images, method_frames, iqa_calc, scene_name="", method_name=""):
    """Evaluate IQA metrics between GT images and method frames."""
    all_metrics = {'psnr': [], 'ssim': [], 'lpips': []}

    min_frames = min(len(gt_images), len(method_frames))

    for i in tqdm(range(min_frames), desc=f"Computing IQA metrics for {method_name}"):
        # Load GT image
        if isinstance(gt_images[i], str):
            gt_img = cv2.imread(gt_images[i])
        else:
            gt_img = gt_images[i]

        method_img = method_frames[i]

        if gt_img is None or method_img is None:
            continue

        # Resize method image to match GT size if necessary
        if gt_img.shape[:2] != method_img.shape[:2]:
            method_img = cv2.resize(method_img, (gt_img.shape[1], gt_img.shape[0]))

        # Calculate reference-based metrics (GT vs Method)
        ref_metrics = iqa_calc.calculate_reference_metrics(gt_img, method_img)
        for metric_name, value in ref_metrics.items():
            if value is not None:
                all_metrics[metric_name].append(value)

    # Calculate averages and standard deviations
    results = {}
    for metric_name, values in all_metrics.items():
        if values:
            results[metric_name] = np.mean(values)
            results[f'{metric_name}_std'] = np.std(values)
        else:
            results[metric_name] = None
            results[f'{metric_name}_std'] = None

    results['num_frames'] = min_frames

    return results

def main():
    parser = argparse.ArgumentParser(description="Integrated evaluation script for CLIP-FID, CMMD, PSNR, SSIM, LPIPS")
    parser.add_argument("--data_dir", type=str, default="data_eval", help="Path to the data directory")
    parser.add_argument("--output_file", type=str, default="metrics_results.csv", help="Output CSV file name")
    parser.add_argument("--temp_dir", type=str, default="temp_frames", help="Path to the temporary directory for frames")
    parser.add_argument("--frame_rate", type=int, default=24, help="Number of frames to extract")
    parser.add_argument("--resolution", type=int, default=1024, help="Resolution of the frames")
    parser.add_argument("--no_resize", action="store_true", default=False, help="Do not resize the frames")
    parser.add_argument("--no_patchify", action="store_true", default=False, help="Do not use patchify for CLIP-FID/CMMD")
    parser.add_argument("--methods", nargs='+', default=['citydreamer', 'gaussiancity', 'ours_stage1', 'ours_stage2', 'difix'], help="List of methods to evaluate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for CMMD")
    parser.add_argument("--device", type=str, default='cuda', help="Device to use for computation (cuda/cpu)")

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    temp_dir = Path(args.temp_dir)

    os.makedirs(temp_dir, exist_ok=True)
    if os.path.exists(temp_dir):
        shutil.rmtree(temp_dir)
    os.makedirs(temp_dir, exist_ok=True)

    # Initialize IQA calculator
    device = args.device if torch.cuda.is_available() or args.device == 'cpu' else 'cpu'
    print(f"Using device: {device}")
    iqa_calc = IntegratedIQACalculator(device=device)

    # Find all scene directories
    scene_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
    scene_dirs = sorted(scene_dirs)

    results = []
    methods = args.methods

    for scene_dir in scene_dirs:
        scene_name = scene_dir.name
        print(f"\n" + "="*50)
        print(f"Processing scene: {scene_name}")
        print("="*50)

        # Create combined directories for each method
        gt_combined_dir = temp_dir / f"{scene_name}_GT_combined"
        os.makedirs(gt_combined_dir, exist_ok=True)

        method_combined_dirs = {}
        for method in methods:
            method_combined_dir = temp_dir / f"{scene_name}_{method}_combined"
            os.makedirs(method_combined_dir, exist_ok=True)
            method_combined_dirs[method] = method_combined_dir

        # Get GT videos and extract frames into the combined directory
        print("Extracting frames from all GT videos...")
        gt_dir = scene_dir / "GT"
        total_gt_frames = 0
        gt_videos = sorted(list(gt_dir.glob("*.mp4")))
        resolution = args.resolution if not args.no_resize else None

        if len(gt_videos) == 0:
            # Copy the images from the input directory to the combined directory
            print("No GT videos found, copying images from input directory...")
            total_frames = len(list(gt_dir.glob("*.jpg")))
            frame_interval = int(total_frames / args.frame_rate)
            if frame_interval <= 0:
                frame_interval = 1
            print(f"Total frames: {total_frames}, Frame interval: {frame_interval}")
            cnt_frames = 0
            for img in sorted(list(gt_dir.glob("*.jpg")))[:-1:frame_interval]:  # exclude the last image
                shutil.copy(img, gt_combined_dir)
                cnt_frames += 1
            print(f"Copied {cnt_frames} frames from images")
            total_gt_frames = cnt_frames
        else:
            for gt_video in gt_videos:
                gt_video_name = gt_video.stem
                print(f"Processing GT video: {gt_video_name}")
                frame_prefix = f"{gt_video_name}_"
                gt_frame_count = extract_frames(gt_video, gt_combined_dir, args.frame_rate, prefix=frame_prefix, resolution=resolution)
                total_gt_frames += gt_frame_count

        print(f"Extracted a total of {total_gt_frames} frames from all GT videos")

        # Get GT images for IQA evaluation
        gt_images_for_iqa = extract_uniform_frames(str(gt_combined_dir), total_gt_frames)

        # Process each method
        for method in methods:
            print(f"\nProcessing method: {method}")
            method_dir = scene_dir / method
            method_videos = sorted(list(method_dir.glob("*.mp4")))
            method_combined_dir = method_combined_dirs[method]
            total_method_frames = 0

            # Extract frames from all videos of this method
            for method_video in method_videos:
                method_video_name = method_video.stem
                print(f"Processing {method} video: {method_video_name}")
                frame_prefix = f"{method_video_name}_"
                method_frame_count = extract_frames(method_video, method_combined_dir, args.frame_rate, prefix=frame_prefix, resolution=resolution)
                total_method_frames += method_frame_count

            print(f"Extracted a total of {total_method_frames} frames from all {method} videos")

            # Get method frames for IQA evaluation
            method_images_for_iqa = extract_uniform_frames(str(method_combined_dir), total_method_frames)
            method_frames_for_iqa = [cv2.imread(img_path) for img_path in method_images_for_iqa]

            # Calculate IQA metrics (PSNR, SSIM, LPIPS)
            print("Calculating IQA metrics...")
            iqa_metrics = evaluate_scene_method_integrated(gt_images_for_iqa, method_frames_for_iqa, iqa_calc, scene_name, method)

            # Calculate CLIP-FID and CMMD
            print("Calculating CLIP-FID and CMMD...")
            clip_fid_score, cmmd_score = calculate_clip_fid_cmmd(
                str(gt_combined_dir), str(method_combined_dir),
                use_patchify=not args.no_patchify, batch_size=args.batch_size
            )

            # Record results
            result = {
                "scene": scene_name,
                "method": method,
                "psnr": iqa_metrics['psnr'],
                "ssim": iqa_metrics['ssim'],
                "lpips": iqa_metrics['lpips'],
                "psnr_std": iqa_metrics['psnr_std'],
                "ssim_std": iqa_metrics['ssim_std'],
                "lpips_std": iqa_metrics['lpips_std'],
                "clip_fid": clip_fid_score,
                "cmmd": cmmd_score,
                "num_frames_evaluated": iqa_metrics['num_frames']
            }

            results.append(result)

            print(f"Results for {method}:")
            for metric in ['psnr', 'ssim', 'lpips', 'clip_fid', 'cmmd']:
                value = result[metric]
                if value is not None:
                    print(f"  {metric.upper()}: {value:.4f}")
                else:
                    print(f"  {metric.upper()}: N/A")

        # Clean up the combined directories
        shutil.rmtree(gt_combined_dir)
        for method_dir in method_combined_dirs.values():
            shutil.rmtree(method_dir)

    # Save results to a CSV file
    if results:
        fieldnames = ["scene", "method", "psnr", "ssim", "lpips", "psnr_std", "ssim_std", "lpips_std", "clip_fid", "cmmd", "num_frames_evaluated"]

        with open(args.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)

        print(f"\nResults saved to {args.output_file}")

        # Print summary statistics
        print("\nSummary Statistics:")
        for method in methods:
            method_results = [r for r in results if r["method"] == method]
            if method_results:
                print(f"\n{method}:")
                for metric in ['psnr', 'ssim', 'lpips', 'clip_fid', 'cmmd']:
                    values = [r[metric] for r in method_results if r[metric] is not None]
                    if values:
                        avg_val = np.mean(values)
                        std_val = np.std(values)
                        print(f"  Average {metric.upper()}: {avg_val:.4f} ± {std_val:.4f}")
                    else:
                        print(f"  Average {metric.upper()}: N/A")
    else:
        print("No results to save.")

    # Clean up main temporary directory
    shutil.rmtree(temp_dir)

if __name__ == "__main__":
    main()