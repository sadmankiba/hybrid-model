import torch
import torch.nn as nn

#### Gated Residual Combiner and Splitters (following  Manticore)  ####

class Combiner(nn.Module):
    """
    Combiner two inbound projectors for outputs from two model blocks. 
    The projected outputs are added in a weighted fashion. The combined output 
    is passed to the Splitter or LM head.
    
    The projected output has dimension as the maximum of the two input dimensions.   
    """
    def __init__(self, input_dim1, input_dim2):
        super(Combiner, self).__init__()
        proj_dim = max(input_dim1, input_dim2)
        self.alpha1 = nn.Parameter(torch.ones(1)) # gating parameter for projector1
        self.alpha2 = nn.Parameter(torch.ones(1)) # gating parameter for projector2
        self.in_proj1 = nn.Linear(input_dim1, proj_dim)
        self.in_proj2 = nn.Linear(input_dim2, proj_dim)
    
    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The output tensor from the first model block. Shape (batch_size, seq_len, input_dim1)
            x2 (torch.Tensor): The output tensor from the second model block. Shape (batch_size, seq_len, input_dim2)
        """
        proj_dim = self.in_proj1.weight.shape[0] # weight.shape = (proj_dim, input_dim1)
        
        x1_proj = self.in_proj1(x1)
        x2_proj = self.in_proj2(x2)
        
        if x1.shape[2] < proj_dim:
            padding = torch.zeros((x1.shape[0], x1.shape[1], proj_dim - x1.shape[2]), device=x1.device)
            x1 = torch.cat([x1, padding], dim=2)
            
        if x2.shape[2] < proj_dim:
            padding = torch.zeros((x2.shape[0], x2.shape[1], proj_dim - x2.shape[2]), device=x2.device)
            x2 = torch.cat([x2, padding], dim=2)
        
        x1_proj_out = (1 - self.alpha1) * x1_proj + self.alpha1 * x1
        x2_proj_out = (1 - self.alpha2) * x2_proj + self.alpha2 * x2
        
        alpha_sum = torch.softmax(torch.cat([self.alpha1, self.alpha2]), dim=0)
        alpha1_sf, alpha2_sf = alpha_sum[0], alpha_sum[1]
        combined = alpha1_sf * x1_proj_out + alpha2_sf * x2_proj_out
        return combined


class Splitter(nn.Module):
    """
    Splitter is used to split the output of the intermediate combiner 
    into two parts to be passed to the two model blocks.
    """
    def __init__(self, input_dim1, input_dim2):
        super(Splitter, self).__init__()
        proj_dim = max(input_dim1, input_dim2)
        self.alpha1 = nn.Parameter(torch.ones(1)) # gating parameter for projector1
        self.alpha2 = nn.Parameter(torch.ones(1)) # gating parameter for projector2
        self.out_proj1 = nn.Linear(proj_dim, input_dim1)
        self.out_proj2 = nn.Linear(proj_dim, input_dim2)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output tensor from the combiner. Shape (batch_size, seq_len, proj_dim)
        """
        input_dim1 = self.out_proj1.weight.shape[0] # weight.shape = (input_dim1, proj_dim)
        input_dim2 = self.out_proj2.weight.shape[0]
        proj1_out = (1 - self.alpha1) * self.out_proj1(x) * self.alpha1 * x[:, :, :input_dim1] 
        proj2_out = (1 - self.alpha2) * self.out_proj2(x) * self.alpha2 * x[:, :, :input_dim2]
        return proj1_out, proj2_out
    
#### Null Combiner and Splitters (no projection or residual) ####
    
class NullCombiner(nn.Module):
    """
    Combiner two outputs from two model blocks with equal weight.  
    """
    def __init__(self, input_dim1, input_dim2):
        super(NullCombiner, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The output tensor from the first model block. Shape (batch_size, seq_len, input_dim)
            x2 (torch.Tensor): The output tensor from the second model block. Shape (batch_size, seq_len, input_dim)
            
        """
        assert x1.shape[2] == self.input_dim1, f"Expected input_dim1={self.input_dim1}, got {x1.shape[2]}"
        assert x2.shape[2] == self.input_dim2, f"Expected input_dim2={self.input_dim2}, got {x2.shape[2]}"
        combined_dim = max(x1.shape[2], x2.shape[2])
        
        if x1.shape[2] < combined_dim:
            padding = torch.zeros((x1.shape[0], x1.shape[1], combined_dim - x1.shape[2]), device=x1.device)
            x1 = torch.cat([x1, padding], dim=2)
            
        if x2.shape[2] < combined_dim:
            padding = torch.zeros((x2.shape[0], x2.shape[1], combined_dim - x2.shape[2]), device=x2.device)
            x2 = torch.cat([x2, padding], dim=2)

        return (x1 + x2) / 2


class NullSplitter(nn.Module):
    """
    NullSplitter copies output to the two model blocks without any change.
    """
    def __init__(self, output_dim1, output_dim2):
        super(NullSplitter, self).__init__()
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
    
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output tensor from the combiner. Shape (batch_size, seq_len, proj_dim)
        """
        return x[:, :, :self.output_dim1], x[:, :, :self.output_dim2]