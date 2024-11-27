import sys
import os
# import parent directory 
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from hybrid.model_zoo import get_gpt_neo_causal, get_mamba_causal
from hybrid.hybrid_model import HybridModel

def count_parameters(model):
    """Count the total and trainable parameters of a PyTorch model."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params