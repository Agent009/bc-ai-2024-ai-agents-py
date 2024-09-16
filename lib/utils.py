import torch

def check_cuda():
    print(f"CUDA version that PyTorch is using: {torch.version.cuda}")
    print(f"GPU is available for PyTorch? {torch.cuda.is_available()}")
    print("GPU if available: " + torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")