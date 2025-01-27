import torch
print(torch.__version__)          # Check PyTorch version
print(torch.cuda.is_available())  # Check if CUDA is available
print(torch.cuda.get_device_name(0))  # Display GPU name (if available