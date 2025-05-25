import os
import socket

def print_env_vars(prefix=""):
    """Print important environment variables with rank and node information."""
    hostname = socket.gethostname()
    vllm_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "NOT_SET")
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "NOT_SET")
    rank = os.environ.get("RANK", "NOT_SET")
    local_rank = os.environ.get("LOCAL_RANK", "NOT_SET")
    world_size = os.environ.get("WORLD_SIZE", "NOT_SET")
    
    print(f"{prefix} on {hostname} - RANK: {rank}, LOCAL_RANK: {local_rank}, WORLD_SIZE: {world_size}")
    print(f"{prefix} on {hostname} - VLLM_ATTENTION_BACKEND: {vllm_backend}")
    print(f"{prefix} on {hostname} - CUDA_VISIBLE_DEVICES: {cuda_visible_devices}")
