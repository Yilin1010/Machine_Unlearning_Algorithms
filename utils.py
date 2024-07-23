import torch
import numpy as np

def device_config():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using {device}")
    return device


def set_seed(seed_value=42):
    """Set seed for reproducibility."""
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)  # if you are using GPU
    print(f"Seed set to {seed_value}")

