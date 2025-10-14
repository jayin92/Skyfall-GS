# single-scale training and multi-scale testing setting proposed in mip-splatting
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time
import random

scenes = ["JAX_004", "JAX_068", "JAX_214", "JAX_260"]

factors = [1] * len(scenes)


dataset_dir = "./data/datasets_JAX"
output_dir = "./outputs/JAX_naive"

dry_run = False
train = True
fused_only = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    # Base command with environment variables and Python script
    base_cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py"

    # Define arguments as a list for easy commenting and modification
    args = [
        f"-s {dataset_dir}/{scene}/outputs_skew",
        f"-m {output_dir}/{scene}",
        "--eval",
        f"--port {6209+int(gpu)+random.randint(10, 100)}",
        "--kernel_size 0.1",
        "--resolution 1",
        "--sh_degree 1",
        "--appearance_enabled",
        "--lambda_depth 0.0",
        "--lambda_opacity 0.0",
        "--densify_until_iter 21000",
        "--densify_grad_threshold 0.0001",
        "--lambda_pseudo_depth 0.0",
        "--start_sample_pseudo 1000",
        "--end_sample_pseudo 21000",
        "--size_threshold 100000",
        "--opacity_reset_interval 4000",
        "--sample_pseudo_interval 10",
        "--datasets_type jax_v1",
    ]

    # Combine base command with all arguments
    cmd = base_cmd + " " + " ".join(args)

    # Create log file path in the output directory
    log_file = f"{output_dir}/{scene}/log.txt"

    # Add logging with tee to display on terminal and save to file simultaneously
    cmd_with_tee = f"{cmd} 2>&1 | tee {log_file}"

    print(cmd)
    print(f"Logging output to terminal and: {log_file}")
    if not dry_run and train and not fused_only:
        # Create output directory if it doesn't exist
        os.makedirs(f"{output_dir}/{scene}", exist_ok=True)
        os.system(cmd_with_tee)

    # Render command
    render_base_cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py"
    render_args = [
        f"-m {output_dir}/{scene}",
        "--load_from_checkpoints",
        # Add more render args here if needed
    ]
    cmd = render_base_cmd + " " + " ".join(render_args)
    print(cmd)
    if not dry_run and not fused_only:
        os.system(cmd)

    # Metrics command
    metrics_base_cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py"
    metrics_args = [
        f"-m {output_dir}/{scene}",
        "--resolution 1",
        # Add more metrics args here if needed
    ]
    cmd = metrics_base_cmd + " " + " ".join(metrics_args)
    print(cmd)
    if not dry_run and not fused_only:
        os.system(cmd)

    # Create fused ply command
    fused_base_cmd = f"python create_fused_ply.py"
    fused_args = [
        f"-m {output_dir}/{scene}",
        f"--output_ply fused/{scene}_naive_toned_iter_30000.ply",
        "--load_from_checkpoints",
        "--iteration 30000",
        # Add more fused ply args here if needed
    ]
    cmd = fused_base_cmd + " " + " ".join(fused_args)
    print(cmd)
    if not dry_run:
        os.system(cmd)

    return True
        
    
def worker(gpu, scene, factor):
    print(f"Starting job on GPU {gpu} with scene {scene}\n")
    train_scene(gpu, scene, factor)
    print(f"Finished job on GPU {gpu} with scene {scene}\n")
    # This worker function starts a job and returns when it's done.
    
    
def dispatch_jobs(jobs, executor):
    future_to_job = {}
    reserved_gpus = set([])  # GPUs that are slated for work but may not be active yet

    while jobs or future_to_job:
        # Get the list of available GPUs, not including those that are reserved.
        all_available_gpus = set(GPUtil.getAvailable(order="first", limit=10, maxMemory=0.15))
        available_gpus = list(all_available_gpus - reserved_gpus)

        # Launch new jobs on available GPUs
        while available_gpus and jobs:
            gpu = available_gpus.pop(0)
            job = jobs.pop(0)
            future = executor.submit(worker, gpu, *job)  # Unpacking job as arguments to worker
            future_to_job[future] = (gpu, job)

            reserved_gpus.add(gpu)  # Reserve this GPU until the job starts processing

        # Check for completed jobs and remove them from the list of running jobs.
        # Also, release the GPUs they were using.
        done_futures = [future for future in future_to_job if future.done()]
        for future in done_futures:
            job = future_to_job.pop(future)  # Remove the job associated with the completed future
            gpu = job[0]  # The GPU is the first element in each job tuple
            reserved_gpus.discard(gpu)  # Release this GPU
            print(f"Job {job} has finished., rellasing GPU {gpu}")
        # (Optional) You might want to introduce a small delay here to prevent this loop from spinning very fast
        # when there are no GPUs available.
        time.sleep(5)
        
    print("All jobs have been processed.")


# Using ThreadPoolExecutor to manage the thread pool
with ThreadPoolExecutor(max_workers=8) as executor:
    dispatch_jobs(jobs, executor)

