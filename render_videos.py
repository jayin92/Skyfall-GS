#!/usr/bin/env python3

import subprocess
import argparse
import threading
from queue import Queue
import time
import os
import glob
import json

def discover_camera_paths():
    """Scan for camera path folders and their JSON files."""
    tasks = []
    
    # Look for camera path folders matching the pattern camera_path_*
    camera_folders = [
        "camera_path_004",
        "camera_path_068",
        "camera_path_214",
        "camera_path_260",
    ]
    # camera_folders = [
    #     "camera_path_164",
    #     "camera_path_168",
    #     "camera_path_175",
    #     "camera_path_264",
    # ]
    # Don't match _NYC
    # camera_folders = [folder for folder in camera_folders if "_NYC" not in folder]
    
    if not camera_folders:
        print("Warning: No camera_path folders found!")
        # Also check for the generic camera_path folder
        if os.path.exists("camera_path"):
            camera_folders = ["camera_path"]
    
    print(f"Found camera path folders: {camera_folders}")
    
    # Scan each folder for JSON files
    for folder in camera_folders:
        scene_id = folder.split("_")[-1] if "_" in folder else "default"
        json_files = glob.glob(os.path.join(folder, "*.json"))
        
        if not json_files:
            print(f"Warning: No JSON files found in {folder}")
            continue
            
        print(f"Found {len(json_files)} camera paths in {folder}")
        
        # Add each JSON file as a rendering task
        for json_file in json_files:
            # Extract r and e values from filename if possible
            filename = os.path.basename(json_file)
            radius = elevation = None
            
            # Try to parse the standard naming format r{r}_e{e}_fov{fov}.json
            if filename.startswith('r') and '_e' in filename:
                try:
                    r_part = filename.split('_')[0][1:]  # Extract digits after 'r'
                    e_part = filename.split('_')[1][1:]  # Extract digits after 'e'
                    radius = int(r_part)
                    elevation = int(e_part)
                except (IndexError, ValueError):
                    pass
            
            # If we couldn't parse values from filename, try to read from JSON
            if radius is None or elevation is None:
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)
                        radius = data.get("_radius", 0)
                        elevation = data.get("_elevation", 0)
                except (json.JSONDecodeError, KeyError, FileNotFoundError):
                    # If all else fails, use default values
                    radius = 0
                    elevation = 0
            
            tasks.append((scene_id, folder, json_file, radius, elevation))
    
    return tasks

def worker(task_queue, gpu_id, total_tasks, model_prefix, model_suffix, iterations):
    """Worker function that processes tasks for a specific GPU."""
    completed_tasks = 0
    
    while not task_queue.empty():
        task = task_queue.get()
        completed_tasks += 1
        
        scene_id, folder, json_path, radius, elevation = task
        
        # Construct model path using prefix and suffix
        model_path = f"{model_prefix}{scene_id}{model_suffix}"
        
        # Create rendering commands
        cmd1 = f"CUDA_VISIBLE_DEVICES={gpu_id} python render_video.py -m {model_path} --camera_path {json_path} --load_from_checkpoints --iteration {iterations}"
        # cmd2 = f"CUDA_VISIBLE_DEVICES={gpu_id} python render_video.py -m {model_path} --camera_path {json_path} --load_from_checkpoints --iteration {iterations} --depth"
        cmd2 = ""
        
        print(f"[GPU {gpu_id}] Starting task {completed_tasks}/{total_tasks}: scene {scene_id}, json {os.path.basename(json_path)}")
        print(f"[GPU {gpu_id}] Using model: {model_path}")
        start_time = time.time()
        
        try:
            subprocess.run(cmd1, shell=True, check=True)
            subprocess.run(cmd2, shell=True, check=True)
            end_time = time.time()
            print(f"[GPU {gpu_id}] Completed task {completed_tasks}/{total_tasks}: scene {scene_id}, json {os.path.basename(json_path)} in {end_time - start_time:.2f}s")
        except subprocess.CalledProcessError as e:
            print(f"[GPU {gpu_id}] Error in task {completed_tasks}/{total_tasks}. Return code: {e.returncode}")
        
        task_queue.task_done()

def main():
    parser = argparse.ArgumentParser(description='Render all camera paths found in camera_path folders')
    parser.add_argument('--gpus', type=str, default='0', help='Comma-separated list of GPU IDs to use')
    parser.add_argument('--model-prefix', type=str, default='satellite_final/JAX_', 
                       help='Prefix for model path (default: satellite_final/JAX_)')
    parser.add_argument('--model-suffix', type=str, default='_pseudo_depth_sh_1_flux_v1', 
                       help='Suffix for model path (default: _pseudo_depth_sh_1_flux_v1)')
    parser.add_argument('--iterations', type=int, default=80000, 
                       help='Number of iterations to use for rendering (default: 80000)')
    args = parser.parse_args()
    
    # Parse GPU IDs
    gpu_ids = args.gpus.split(',')
    print(f"Using GPUs: {', '.join(gpu_ids)}")
    print(f"Model format: {args.model_prefix}SCENE_ID{args.model_suffix}")
    print(f"Using {args.iterations} iterations")
    
    # Discover all camera paths
    all_tasks = discover_camera_paths()
    
    if not all_tasks:
        print("No camera paths found to render. Exiting.")
        return
    
    print(f"Discovered {len(all_tasks)} total rendering tasks")
    
    # Distribute tasks among GPUs
    gpu_tasks = {gpu_id: [] for gpu_id in gpu_ids}
    for idx, task in enumerate(all_tasks):
        gpu_id = gpu_ids[idx % len(gpu_ids)]
        gpu_tasks[gpu_id].append(task)
    
    # Create a task queue for each GPU
    gpu_queues = {gpu_id: Queue() for gpu_id in gpu_ids}
    
    # Add tasks to queues
    for gpu_id, tasks in gpu_tasks.items():
        for task in tasks:
            gpu_queues[gpu_id].put(task)
        print(f"Assigned {len(tasks)} tasks to GPU {gpu_id}")
    
    # Create and start worker threads for each GPU
    workers = []
    for gpu_id, tasks in gpu_tasks.items():
        thread = threading.Thread(
            target=worker, 
            args=(
                gpu_queues[gpu_id], 
                gpu_id, 
                len(tasks),
                args.model_prefix,
                args.model_suffix,
                args.iterations
            )
        )
        thread.daemon = True
        thread.start()
        workers.append(thread)
    
    # Wait for all queues to be processed
    for gpu_id in gpu_ids:
        gpu_queues[gpu_id].join()
    
    print("All render tasks completed successfully!")

if __name__ == "__main__":
    main()