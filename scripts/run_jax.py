# single-scale training and multi-scale testing setting proposed in mip-splatting
import os
import GPUtil
from concurrent.futures import ThreadPoolExecutor
import queue
import time

scenes = ["JAX_004", "JAX_068", "JAX_214", "JAX_260"]

factors = [1] * len(scenes)

dataset_dir = "./data/datasets_JAX"
output_dir = "./outputs/JAX"

dry_run = False
train = True
fused_only = False

jobs = list(zip(scenes, factors))

def train_scene(gpu, scene, factor):
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python train.py -s {dataset_dir}/{scene}/outputs_skew -m {output_dir}/{scene} --eval --port {6209+int(gpu)} --kernel_size 0.1 --resolution 1 --sh_degree 1 --appearance_enabled --lambda_depth 0 --lambda_opacity 10 --densify_until_iter 21000 --densify_grad_threshold 0.0001 --lambda_pseudo_depth 0.5 --start_sample_pseudo 1000 --end_sample_pseudo 21000 --size_threshold 20 --scaling_lr 0.001 --rotation_lr 0.001 --opacity_reset_interval 3000 --sample_pseudo_interval 10"
    print(cmd)
    if not dry_run and train and not fused_only:
        os.system(cmd)

    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python render.py -m {output_dir}/{scene} --load_from_checkpoints"
    print(cmd)
    if not dry_run and not fused_only:
        os.system(cmd)
        
    cmd = f"OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES={gpu} python metrics.py -m {output_dir}/{scene} --resolution 1"
    print(cmd)
    if not dry_run and not fused_only:
        os.system(cmd)

    cmd = f"python create_fused_ply.py -m {output_dir}/{scene} --output_ply fused/{scene}_toned_iter_30000.ply --load_from_checkpoints --iteration 30000"
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

