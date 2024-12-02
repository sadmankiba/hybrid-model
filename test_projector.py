import torch

from hybrid.projector import (
    Combiner, Splitter, 
    NullCombiner, NullSplitter,
    ResidualCombiner, ResidualSplitter
)
from utils.params import count_parameters


def test_combiner_param_count():
    combiner = Combiner(10, 20)
    
    # params = 10 * 20 + 20 + 20 * 20 + 20 = 640
    total_params, trainable_params = count_parameters(combiner)
    assert total_params == 642
    assert trainable_params == 642
    
def test_combiner_forward():
    combiner = Combiner(10, 20)
    x1 = torch.randn(5, 8, 10) # (batch_size, seq_len, input_dim1)
    x2 = torch.randn(5, 8, 20)
    output = combiner(x1, x2)
    assert output.shape == (5, 8, 20)
    
    x1_padded = torch.cat([x1, torch.zeros((5, 8, 10))], dim=2)
    torch.isclose(output, 0.5 * x1_padded + 0.5 * x2, atol=1e-6)

def test_splitter_param_count():
    splitter = Splitter(10, 20)
    
    # params = 20 * 10 + 10 + 20 * 20 + 20 = 630
    total_params, trainable_params = count_parameters(splitter)
    assert total_params == 632
    assert trainable_params == 632
    

def test_splitter_forward():
    splitter = Splitter(10, 20)
    x = torch.randn(5, 8, 20)
    out1, out2 = splitter(x)
    assert out1.shape == (5, 8, 10)
    assert out2.shape == (5, 8, 20)
    
    torch.isclose(out1, x[:, :, :10], atol=1e-6)
    torch.isclose(out2, x, atol=1e-6)


def test_null_combiner_forward():
    null_combiner = NullCombiner(10, 20)
    x1 = torch.randn(5, 8, 10) # (batch_size, seq_len, input_dim)
    x2 = torch.randn(5, 8, 20)
    output = null_combiner(x1, x2)
    assert output.shape == (5, 8, 20)
    
    x1_padded = torch.cat([x1, torch.zeros((5, 8, 10))], dim=2)
    expected_output = (x1_padded + x2) / 2
    assert torch.allclose(output, expected_output, atol=1e-6)


def test_null_splitter_forward():
    null_splitter = NullSplitter(10, 20)
    x = torch.randn(5, 8, 20)
    out1, out2 = null_splitter(x)
    assert out1.shape == (5, 8, 10)
    assert out2.shape == (5, 8, 20)
    
    assert torch.allclose(out1, x[:, :, :10], atol=1e-6)
    assert torch.allclose(out2, x, atol=1e-6)
    
        
def test_residual_combiner_forward():
    residual_combiner = ResidualCombiner(10, 20)
    x1 = torch.randn(5, 8, 10) # (batch_size, seq_len, input_dim1)
    x2 = torch.randn(5, 8, 20)
    output = residual_combiner(x1, x2)
    assert output.shape == (5, 8, 20)
    
    x1_proj = residual_combiner.in_proj1(x1)
    x2_proj = residual_combiner.in_proj2(x2)
    x1_padded = torch.cat([x1, torch.zeros((5, 8, 10))], dim=2)
    expected_output = (x1_proj + x1_padded + x2_proj + x2) / 2
    assert torch.allclose(output, expected_output, atol=1e-6)


def test_residual_splitter_forward():
    residual_splitter = ResidualSplitter(10, 20)
    x = torch.randn(5, 8, 20)
    out1, out2 = residual_splitter(x)
    assert out1.shape == (5, 8, 10)
    assert out2.shape == (5, 8, 20)
    
    x1_proj = residual_splitter.out_proj1(x)
    x2_proj = residual_splitter.out_proj2(x)
    expected_out1 = x1_proj + x[:, :, :10]
    expected_out2 = x2_proj + x
    assert torch.allclose(out1, expected_out1, atol=1e-6)
    assert torch.allclose(out2, expected_out2, atol=1e-6)

