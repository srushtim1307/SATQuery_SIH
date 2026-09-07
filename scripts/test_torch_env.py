import sys
import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch Version: {torch.__version__}")
cuda_avail = torch.cuda.is_available()
print(f"CUDA Available: {cuda_avail}")
if cuda_avail:
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"Device VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
else:
    print("Running on CPU mode.")
