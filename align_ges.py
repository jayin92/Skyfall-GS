#!/usr/bin/env python3

import os
import subprocess
import json
import numpy as np
import cv2
import argparse
from tqdm import tqdm
from PIL import Image
import shutil
import glob
from concurrent.futures import ThreadPoolExecutor

def extract_reference_frames(video_path, num_frames=1, output_dir="reference_frames"):
    """Extract multiple reference frames from ground truth video at evenly spaced intervals."""
    os.makedirs(output_dir, exist_ok=True)
    video_name = os.path.basename(video_path).split('.')[0]
    
    # Get video duration and frame count
    cmd = f"ffprobe -v error -select_streams v:0 -count_packets -show_entries stream=nb_read_packets -of csv=p=0 {video_path}"
    try:
        frame_count = int(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
    except:
        # Fallback method
        cmd = f"ffprobe -v error -select_streams v:0 -show_entries stream=nb_frames -of csv=p=0 {video_path}"
        try:
            frame_count = int(subprocess.check_output(cmd, shell=True).decode('utf-8').strip())
        except:
            print("Warning: Could not determine frame count, assuming 240 frames")
            frame_count = 240
    
    print(f"Video has {frame_count} frames")
    
    # Calculate frame indices to extract
    if num_frames == 1:
        frame_indices = [0]  # Just the first frame
    else:
        # Evenly spaced frames across the video
        frame_indices = np.linspace(0, frame_count - 1, num_frames, dtype=int)
    
    reference_frames = []
    
    # Extract each frame
    for idx, frame_idx in enumerate(frame_indices):
        output_path = os.path.join(output_dir, f"{video_name}_frame_{frame_idx:03d}.png")
        if not os.path.exists(output_path):
            cmd = f"ffmpeg -i {video_path} -vf \"select=eq(n\\,{frame_idx})\" -vframes 1 {output_path}"
            subprocess.run(cmd, shell=True, check=True)
        reference_frames.append(output_path)
        print(f"Extracted frame {frame_idx} to {output_path}")
    
    return reference_frames

def generate_camera_path(original_path, target_alt, temp_folder="temp_calibration"):
    """Generate a new camera path JSON with the updated target altitude using gen_render_path.py."""
    os.makedirs(temp_folder, exist_ok=True)
    
    # Read parameters from original camera path
    with open(original_path, 'r') as f:
        original_data = json.load(f)
    
    # Get original parameters
    original_target = original_data["_target"]
    radius = original_data["_radius"]
    elevation = original_data["_elevation"]
    fov = original_data["camera_path"][0]["fov"]
    height = original_data["render_height"]
    width = original_data["render_width"]
    fps = original_data["fps"]
    num_frame = len(original_data["camera_path"])
    
    # Create new target with updated altitude
    new_target = f"{original_target[0]},{original_target[1]},{target_alt}"
    
    # Generate a unique output filename
    temp_path = os.path.join(temp_folder, f"alt_{target_alt:.2f}.json")
    
    # Run gen_render_path.py to create new camera path
    cmd = [
        "python", "gen_render_path.py",
        "--fov", str(fov),
        "--target", new_target,
        "--elevation", str(elevation),
        "--radius", str(radius),
        "--num_frame", str(num_frame),
        "--fps", str(fps),
        "--height", str(height),
        "--width", str(width),
        "--output_folder", temp_folder
    ]
    
    # Run the command
    subprocess.run(cmd, stdout=subprocess.DEVNULL)
    
    # Find the most recently created JSON file in the temp folder
    json_files = [f for f in os.listdir(temp_folder) if f.endswith('.json')]
    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {temp_folder} after running gen_render_path.py")
    
    latest_file = max([os.path.join(temp_folder, f) for f in json_files], key=os.path.getctime)
    
    # Rename to our expected path
    if latest_file != temp_path:
        shutil.move(latest_file, temp_path)
    
    return temp_path

def render_frames(scene_id, camera_path, target_alt, model_prefix, model_suffix, 
                  iterations, num_frames, temp_folder="temp_calibration", gpu_id=0):
    """Render frames from a camera path with specific target altitude using the --num_frames parameter."""
    # Create new camera path
    temp_camera_path = generate_camera_path(camera_path, target_alt, temp_folder)
    
    # Set up output directory
    output_dir = os.path.join(temp_folder, f"alt_{target_alt:.2f}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct model path
    model_path = f"{model_prefix}{scene_id}{model_suffix}"
    
    # Build command to render only the specified number of frames
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_id} python render_video.py " \
          f"-m {model_path} " \
          f"--camera_path {temp_camera_path} " \
          f"--load_from_checkpoints " \
          f"--iteration {iterations} " \
          f"--save_images " \
          f"--num_frames {num_frames}"
    
    # Run the command
    try:
        subprocess.run(cmd, shell=True, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        # Find the rendered frames
        video_name = os.path.basename(temp_camera_path).split('.')[0]
        frames_dir = os.path.join(model_path, "video", f"ours_{iterations}", f"{video_name}_frames")
        
        if not os.path.exists(frames_dir):
            print(f"Warning: Frames directory {frames_dir} not found")
            return []
        
        # List all frames - now there should only be the number of frames we requested
        all_frames = sorted(glob.glob(os.path.join(frames_dir, "*.png")))
        
        if len(all_frames) == 0:
            print(f"Warning: No frames found in {frames_dir}")
            return []
        
        # Copy the frames to our temp directory for easier comparison
        copied_frames = []
        for i, frame in enumerate(all_frames):
            dest_path = os.path.join(output_dir, f"frame_{i:03d}.png")
            shutil.copy(frame, dest_path)
            copied_frames.append(dest_path)
        
        return copied_frames
    
    except subprocess.CalledProcessError:
        print(f"Error rendering frames with altitude {target_alt:.2f}")
        print(f"Command: {cmd}")
        return []

def compare_structure(rendered_path, reference_path):
    """Compare blurry images by matching blur levels first."""
    rendered_img = cv2.imread(rendered_path)
    reference_img = cv2.imread(reference_path)
    
    # Resize if needed
    if rendered_img.shape != reference_img.shape:
        rendered_img = cv2.resize(rendered_img, (reference_img.shape[1], reference_img.shape[0]))
    
    # Convert to grayscale
    rendered_gray = cv2.cvtColor(rendered_img, cv2.COLOR_BGR2GRAY)
    reference_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    
    # Apply blur to reference to match rendered image quality
    blurred_reference = cv2.GaussianBlur(reference_gray, (5, 5), 0)
    
    # Apply histogram equalization 
    rendered_gray = cv2.equalizeHist(rendered_gray)
    blurred_reference = cv2.equalizeHist(blurred_reference)
    
    # Use Sobel operators which might work better on blurry images
    rendered_sobel_x = cv2.Sobel(rendered_gray, cv2.CV_64F, 1, 0, ksize=3)
    rendered_sobel_y = cv2.Sobel(rendered_gray, cv2.CV_64F, 0, 1, ksize=3)
    rendered_sobel = cv2.magnitude(rendered_sobel_x, rendered_sobel_y)
    
    reference_sobel_x = cv2.Sobel(blurred_reference, cv2.CV_64F, 1, 0, ksize=3)
    reference_sobel_y = cv2.Sobel(blurred_reference, cv2.CV_64F, 0, 1, ksize=3)
    reference_sobel = cv2.magnitude(reference_sobel_x, reference_sobel_y)
    
    # Normalize
    rendered_sobel = cv2.normalize(rendered_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    reference_sobel = cv2.normalize(reference_sobel, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Save debug images
    debug_dir = os.path.join(os.path.dirname(rendered_path), "debug")
    os.makedirs(debug_dir, exist_ok=True)
    cv2.imwrite(os.path.join(debug_dir, "rendered_sobel.png"), rendered_sobel)
    cv2.imwrite(os.path.join(debug_dir, "reference_sobel.png"), reference_sobel)
    
    # Use both template matching and mean squared error
    result = cv2.matchTemplate(rendered_sobel, reference_sobel, cv2.TM_CCOEFF_NORMED)
    similarity = np.max(result)
    
    # Calculate MSE between edge maps
    mse = np.mean((rendered_sobel.astype(float) - reference_sobel.astype(float)) ** 2)
    mse_score = 1 / (1 + mse)  # Convert to a similarity score (0-1)
    
    # Combined score
    combined_score = 0.7 * similarity + 0.3 * mse_score
    
    return combined_score

def compare_all_frames(rendered_frames, reference_frames):
    """Compare multiple rendered frames with reference frames and return average score."""
    total_score = 0.0
    num_comparisons = 0
    
    # Use at most the available number of frames
    num_frames = min(len(rendered_frames), len(reference_frames))
    
    if num_frames == 0:
        return 0.0
    
    for i in range(num_frames):
        score = compare_structure(rendered_frames[i], reference_frames[i])
        total_score += score
        num_comparisons += 1
        print(f"  Frame {i}: Score {score:.4f}")
    
    if num_comparisons == 0:
        return 0.0
    
    avg_score = total_score / num_comparisons
    print(f"  Average score across {num_comparisons} frames: {avg_score:.4f}")
    return avg_score

def binary_search_altitude(scene_id, camera_path, reference_frames, model_prefix, model_suffix, 
                          iterations, min_alt, max_alt, num_frames, tolerance=0.5, gpu_id=0):
    """Find optimal target altitude using binary search and structural comparison across multiple frames."""
    best_alt = min_alt
    best_score = 0.0
    
    print(f"Starting binary search for scene {scene_id} with altitude range: {min_alt} to {max_alt}")
    print(f"Comparing {num_frames} frames per altitude test")
    
    iteration = 1
    while max_alt - min_alt > tolerance:
        print(f"Iteration {iteration}: Testing altitude range [{min_alt:.2f}, {max_alt:.2f}]")
        
        # Test three points: middle and quarters
        mid_alt = (min_alt + max_alt) / 2
        left_alt = (min_alt + mid_alt) / 2
        right_alt = (mid_alt + max_alt) / 2
        
        # Render frames at these altitudes
        print(f"Rendering frames for altitude {left_alt:.2f}m")
        left_frames = render_frames(scene_id, camera_path, left_alt, 
                                  model_prefix, model_suffix, iterations, num_frames, gpu_id=gpu_id)
        
        print(f"Rendering frames for altitude {mid_alt:.2f}m")
        mid_frames = render_frames(scene_id, camera_path, mid_alt, 
                                 model_prefix, model_suffix, iterations, num_frames, gpu_id=gpu_id)
        
        print(f"Rendering frames for altitude {right_alt:.2f}m")
        right_frames = render_frames(scene_id, camera_path, right_alt, 
                                   model_prefix, model_suffix, iterations, num_frames, gpu_id=gpu_id)
        
        # Compare with reference frames
        print(f"Comparing frames for altitude {left_alt:.2f}m")
        left_score = compare_all_frames(left_frames, reference_frames) if left_frames else 0
        
        print(f"Comparing frames for altitude {mid_alt:.2f}m")
        mid_score = compare_all_frames(mid_frames, reference_frames) if mid_frames else 0
        
        print(f"Comparing frames for altitude {right_alt:.2f}m")
        right_score = compare_all_frames(right_frames, reference_frames) if right_frames else 0
        
        print(f"Scores: {left_alt:.2f}m ({left_score:.4f}), {mid_alt:.2f}m ({mid_score:.4f}), {right_alt:.2f}m ({right_score:.4f})")
        
        # Update search range based on scores
        if left_score > mid_score and left_score > right_score:
            max_alt = mid_alt
            if left_score > best_score:
                best_score = left_score
                best_alt = left_alt
        elif right_score > mid_score and right_score > left_score:
            min_alt = mid_alt
            if right_score > best_score:
                best_score = right_score
                best_alt = right_alt
        else:
            # Mid is best or equal
            min_alt = left_alt
            max_alt = right_alt
            if mid_score > best_score:
                best_score = mid_score
                best_alt = mid_alt
        
        iteration += 1
    
    # Final verification with a finer grid around the best altitude
    print(f"Refining around best altitude {best_alt:.2f}m")
    fine_range = 2 * tolerance
    step = fine_range / 3  # Test fewer points in refinement to save time
    test_alts = [best_alt - fine_range/2 + i*step for i in range(4)]
    
    best_fine_alt = best_alt
    best_fine_score = best_score
    
    for alt in test_alts:
        print(f"Rendering frames for fine-tuning altitude {alt:.2f}m")
        frames = render_frames(scene_id, camera_path, alt, 
                             model_prefix, model_suffix, iterations, num_frames, gpu_id=gpu_id)
        if frames:
            print(f"Comparing frames for fine-tuning altitude {alt:.2f}m")
            score = compare_all_frames(frames, reference_frames)
            print(f"Fine test: {alt:.2f}m -> {score:.4f}")
            if score > best_fine_score:
                best_fine_score = score
                best_fine_alt = alt
    
    if best_fine_score > best_score:
        return best_fine_alt, best_fine_score
    else:
        return best_alt, best_score

def create_batch_script(scene_id, optimal_alt):
    """Create a batch script to regenerate all camera paths with the optimized altitude."""
    script_content = f"""#!/bin/bash
# Generated script to create camera paths with optimized target altitude ({optimal_alt:.2f}m) for scene {scene_id}

# Create output directory
mkdir -p camera_path_{scene_id}_optimized

# Generate high, medium, and low elevation camera paths
python gen_render_path.py --fov 20 --target 0,0,{optimal_alt:.2f} --elevation 78 --radius 612 --num_frame 240 --fps 24 --height 1024 --width 1024 --output_folder camera_path_{scene_id}_optimized
python gen_render_path.py --fov 20 --target 0,0,{optimal_alt:.2f} --elevation 50 --radius 488 --num_frame 240 --fps 24 --height 1024 --width 1024 --output_folder camera_path_{scene_id}_optimized
python gen_render_path.py --fov 20 --target 0,0,{optimal_alt:.2f} --elevation 27 --radius 352 --num_frame 240 --fps 24 --height 1024 --width 1024 --output_folder camera_path_{scene_id}_optimized

echo "All camera paths generated with optimized target altitude of {optimal_alt:.2f}m"
"""
    
    script_path = f"generate_optimal_paths_{scene_id}.sh"
    with open(script_path, "w") as f:
        f.write(script_content)
    
    os.chmod(script_path, 0o755)  # Make the script executable
    print(f"Created batch script at {script_path}")

def main():
    parser = argparse.ArgumentParser(description='Find optimal target altitude by comparing with ground truth.')
    parser.add_argument('--scene-id', required=True, help='Scene ID (e.g., 004)')
    parser.add_argument('--gt-video', required=True, help='Path to ground truth video')
    parser.add_argument('--camera-path', required=True, help='Path to the original camera path JSON')
    parser.add_argument('--min-alt', type=float, default=0, help='Minimum altitude to try')
    parser.add_argument('--max-alt', type=float, default=100, help='Maximum altitude to try')
    parser.add_argument('--num-frames', type=int, default=1, 
                        help='Number of frames to compare (1 = first frame only, 4 = evenly spaced frames)')
    parser.add_argument('--tolerance', type=float, default=0.5, help='Convergence tolerance')
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID to use')
    parser.add_argument('--model-prefix', type=str, default='satellite_final/JAX_', 
                        help='Prefix for model path')
    parser.add_argument('--model-suffix', type=str, default='_pseudo_depth_sh_1_flux_v1', 
                        help='Suffix for model path')
    parser.add_argument('--iterations', type=int, default=80000, 
                        help='Number of iterations to use for rendering')
    
    args = parser.parse_args()
    
    # Extract reference frames from ground truth video
    reference_frames = extract_reference_frames(args.gt_video, args.num_frames)
    print(f"Extracted {len(reference_frames)} reference frames")
    
    # Run optimization
    optimal_alt, score = binary_search_altitude(
        args.scene_id,
        args.camera_path,
        reference_frames,
        args.model_prefix,
        args.model_suffix,
        args.iterations,
        args.min_alt,
        args.max_alt,
        args.num_frames,
        args.tolerance,
        args.gpu
    )
    
    print(f"\nResults for {args.scene_id}:")
    print(f"Optimal target altitude: {optimal_alt:.2f}m")
    print(f"Similarity score: {score:.4f}")
    
    # Save result to file
    result_file = f"optimal_altitude_{args.scene_id}.txt"
    with open(result_file, "w") as f:
        f.write(f"Scene ID: {args.scene_id}\n")
        f.write(f"Optimal target altitude: {optimal_alt:.2f}m\n")
        f.write(f"Similarity score: {score:.4f}\n")
        f.write(f"Ground truth video: {args.gt_video}\n")
        f.write(f"Camera path: {args.camera_path}\n")
        f.write(f"Compared {args.num_frames} frames\n")
    
    print(f"Results saved to {result_file}")
    
    # Create an optimized camera path for future use
    optimized_folder = "optimized_camera_paths"
    os.makedirs(optimized_folder, exist_ok=True)
    optimized_path = generate_camera_path(args.camera_path, optimal_alt, optimized_folder)
    print(f"Optimized camera path saved to {optimized_path}")
    
    # Create a batch script to generate all camera paths with this optimized altitude
    create_batch_script(args.scene_id, optimal_alt)

if __name__ == "__main__":
    main()