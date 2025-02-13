import torch

def get_available_devices():
    devices = []
    
    # Check CPU
    devices.append(torch.device("cpu"))
    
    # Check CUDA (NVIDIA GPUs)
    if torch.cuda.is_available():
        num_cuda = torch.cuda.device_count()
        for i in range(num_cuda):
            devices.append(torch.device(f"cuda:{i}"))
    
    # Check MPS (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        devices.append(torch.device("mps"))

    return devices
