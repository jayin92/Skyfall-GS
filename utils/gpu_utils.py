import torch
import os
import nvidia_smi
from typing import Optional, Tuple

class GPUManager:
    def __init__(self):
        nvidia_smi.nvmlInit()
        self.device_count = nvidia_smi.nvmlDeviceGetCount()
        
    def get_gpu_memory_info(self, device_id: int) -> Tuple[int, int]:
        """Get memory usage info for a specific GPU."""
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        return info.used, info.total
    
    def get_gpu_utilization(self, device_id: int) -> int:
        """Get GPU utilization percentage."""
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(device_id)
        return nvidia_smi.nvmlDeviceGetUtilizationRates(handle).gpu

    def find_best_gpu(self, min_memory_gb: float = 4.0) -> Optional[int]:
        """
        Find the GPU with the most available memory and lowest utilization.
        Args:
            min_memory_gb: Minimum required free memory in GB
        Returns:
            GPU index or None if no suitable GPU found
        """
        best_gpu = None
        best_score = float('-inf')
        min_memory_bytes = min_memory_gb * 1024 * 1024 * 1024

        for i in range(self.device_count):
            try:
                used_mem, total_mem = self.get_gpu_memory_info(i)
                util = self.get_gpu_utilization(i)
                
                free_mem = total_mem - used_mem
                if free_mem < min_memory_bytes:
                    continue
                
                # Score based on free memory and inverse of utilization
                # Higher score is better
                score = free_mem / total_mem * (100 - util)
                
                if score > best_score:
                    best_score = score
                    best_gpu = i
            except Exception as e:
                print(f"Error checking GPU {i}: {e}")
                continue
                
        return best_gpu

    def __del__(self):
        nvidia_smi.nvmlShutdown()

def get_next_available_gpu(min_memory_gb: float = 4.0) -> torch.device:
    """
    Get the next available GPU device.
    Args:
        min_memory_gb: Minimum required free memory in GB
    Returns:
        torch.device: CUDA device for the selected GPU
    Raises:
        RuntimeError: If no suitable GPU is found
    """
    gpu_manager = GPUManager()
    gpu_id = gpu_manager.find_best_gpu(min_memory_gb)
    
    if gpu_id is None:
        raise RuntimeError(f"No GPU with {min_memory_gb}GB free memory available")
        
    return torch.device(f"cuda:{gpu_id}")