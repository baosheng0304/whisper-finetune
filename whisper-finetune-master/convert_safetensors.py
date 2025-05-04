
from safetensors.torch import load_file
import torch

# Path to your safetensors file
safetensors_path = "./models_small/model.safetensors"
# Path to save the converted PyTorch .bin file
pytorch_bin_path = "./models_small/pytorch_model.bin"

# Load the safetensors weights
state_dict = load_file(safetensors_path)

# Save as PyTorch .bin checkpoint
torch.save(state_dict, pytorch_bin_path)
