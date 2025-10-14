import cv2
import os
import argparse
import subprocess
import glob
import shutil
from pathlib import Path

def merge_frames(left, right, output, lowres=False, target_size=(2048, 2048)):
    """Merge two frames into one"""
    os.makedirs(os.path.dirname(output), exist_ok=True)
    frame1 = cv2.imread(left)
    frame2 = cv2.imread(right)
    
    if frame1 is None or frame2 is None:
        print(f"Error: Could not read images {left} or {right}")
        return None

    # Resize frames to target size if they're not already that size
    if frame1.shape[0] != target_size[1] or frame1.shape[1] != target_size[0]:
        frame1 = cv2.resize(frame1, target_size)
    if frame2.shape[0] != target_size[1] or frame2.shape[1] != target_size[0]:
        frame2 = cv2.resize(frame2, target_size)
    
    res = frame1.copy()
    
    # For 2048x2048 resolution, adjust the merging region
    # Taking the bottom quarter of frame2 and placing it in the bottom-left quarter of frame1
    height, width = target_size[1], target_size[0]
    
    # Define merge region (adjust these values based on your specific needs)
    start_h = int(height * 0.75)  # Start at 75% height (1536 for 2048x2048)
    end_h = height                # End at 100% height (2048)
    start_w = 0                   # Start at 0% width
    end_w = int(width * 0.5)      # End at 50% width (1024 for 2048x2048)
    
    res[start_h:end_h, start_w:end_w] = frame2[start_h:end_h, start_w:end_w]

    if lowres:
        res = cv2.resize(res, (int(res.shape[1]/2), int(res.shape[0]/2)))
    
    cv2.imwrite(output, res)
    return output

def create_video_from_frames(frames_dir, output_video, fps=30):
    """Create a video from frames using ffmpeg"""
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_video), exist_ok=True)
    
    # Construct the ffmpeg command
    cmd = [
        'ffmpeg',
        '-y',  # Overwrite output file if it exists
        '-framerate', str(fps),
        '-pattern_type', 'glob',
        '-i', frames_dir,
        '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p',
        output_video
    ]
    
    # Run the ffmpeg command
    try:
        subprocess.run(cmd, check=True)
        print(f"Video created successfully: {output_video}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating video: {e}")
        return False

def process_image_folders(left_folder, right_folder, output_path, create_video=False, fps=30, lowres=False, size="2048x2048"):
    """Process image folders by merging corresponding images and optionally creating a video"""
    
    # Parse size parameter
    if "x" in size:
        width, height = map(int, size.split("x"))
        target_size = (width, height)
    else:
        target_size = (2048, 2048)  # Default
    
    # Get all jpeg files from both folders
    left_images = sorted(glob.glob(os.path.join(left_folder, "*.jpeg")))
    right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpeg")))

    # Also try .jpg extension if no .jpeg files found
    if not left_images:
        left_images = sorted(glob.glob(os.path.join(left_folder, "*.jpg")))
    if not right_images:
        right_images = sorted(glob.glob(os.path.join(right_folder, "*.jpg")))

    print(f"Found {len(left_images)} left images and {len(right_images)} right images")
    print(f"Target size: {target_size}")
    
    if not left_images or not right_images:
        print("Error: No images found in one or both folders")
        return []
    
    # Ensure we have the same number of images
    min_images = min(len(left_images), len(right_images))
    left_images = left_images[:min_images]
    right_images = right_images[:min_images]
    
    # Create output directory for merged frames
    merged_frames_dir = os.path.join(output_path, "merged_frames")
    os.makedirs(merged_frames_dir, exist_ok=True)
    
    # Merge frames
    merged_frames = []
    for i, (left, right) in enumerate(zip(left_images, right_images)):
        output_frame = os.path.join(merged_frames_dir, f"frame_{i:04d}.jpg")
        result = merge_frames(left, right, output_frame, lowres, target_size)
        if result:
            merged_frames.append(result)
        
        # Progress indicator
        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{min_images} frames")
    
    print(f"Successfully merged {len(merged_frames)} frames")
    
    # Create video if requested
    if create_video:
        if not output_path.endswith((".mp4", ".avi", ".mov")):
            output_path += ".mp4"
        success = create_video_from_frames(os.path.join(merged_frames_dir, "*.jpg"), output_path, fps)
        
        # Clean up temporary files if video was created successfully
        if success:
            shutil.rmtree(merged_frames_dir)
            print(f"Cleaned up temporary frames directory")
    
    return merged_frames

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Merge image frames from two folders")
    parser.add_argument("--left", help="path to the left image folder with jpeg files", required=True)
    parser.add_argument("--right", help="path to the right image folder with jpeg files", required=True)
    parser.add_argument("--output", help="path to the output (folder or video)", required=True)
    parser.add_argument("--video", help="create video from merged frames", action="store_true", default=False)
    parser.add_argument("--fps", help="frames per second for output video", type=int, default=30)
    parser.add_argument("--lowres", action="store_true", help="low resolution output", default=False)
    parser.add_argument("--size", help="target size for images (e.g., 2048x2048)", default="2048x2048")

    args = parser.parse_args()
    
    # Check if inputs are folders
    if os.path.isdir(args.left) and os.path.isdir(args.right):
        process_image_folders(args.left, args.right, args.output, args.video, args.fps, args.lowres, args.size)
    else:
        print("Error: Both inputs must be directories containing JPEG images")