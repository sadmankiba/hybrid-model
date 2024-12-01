import torch

from hybrid.projector import Combiner, Splitter
from utils.params import count_parameters

def test_combiner_param_count():
    combiner = Combiner(10, 20)
    
    # params = 10 * 20 + 20 + 20 * 20 + 20 = 640
    total_params, trainable_params = count_parameters(combiner)
    assert total_params == 642
    assert trainable_params == 642
    
def test_combiner_forward():
    combiner = Combiner(10, 20)
    x1 = torch.randn(5, 10)
    x2 = torch.randn(5, 20)
    output = combiner(x1, x2)
    assert output.shape == (5, 20)
    
    x1_padded = torch.cat([x1, torch.zeros((5, 10))], dim=1)
    torch.isclose(output, 0.5 * x1_padded + 0.5 * x2, atol=1e-6)

def test_splitter_param_count():
    splitter = Splitter(10, 20)
    
    # params = 20 * 10 + 10 + 20 * 20 + 20 = 630
    total_params, trainable_params = count_parameters(splitter)
    assert total_params == 632
    assert trainable_params == 632
    

def test_splitter_forward():
    splitter = Splitter(10, 20)
    x = torch.randn(5, 20)
    out1, out2 = splitter(x)
    assert out1.shape == (5, 10)
    assert out2.shape == (5, 20)
    
    torch.isclose(out1, x[:, :10], atol=1e-6)
    torch.isclose(out2, x, atol=1e-6)