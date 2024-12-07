import torch
import torch.nn as nn

#### Gated Residual Softmaxed Combiner and Splitters (following  Manticore)  ####

class GatedResidualSoftCombiner(nn.Module):
    """
    Combines with gated residual, with gating softmaxed before projection. 
    """
    def __init__(self, input_dim1, input_dim2):
        super(GatedResidualSoftCombiner, self).__init__()
        proj_dim = max(input_dim1, input_dim2)
        self.alpha1 = nn.Parameter(torch.randn(1)) # gating parameter for projector1, in [0, 1]
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
        
        alpha1 = self.alpha1
        alpha2 = 1 - alpha1
        alpha_sf1, alpha_sf2 = torch.softmax(torch.cat([alpha1, alpha2]), dim=0)
        x1_proj_out = (1 - alpha_sf1) * x1_proj + alpha_sf1 * x1
        x2_proj_out = (1 - alpha_sf2) * x2_proj + alpha_sf2 * x2
        
        combined = alpha_sf1 * x1_proj_out + alpha_sf2 * x2_proj_out
        return combined


class GatedResidualSoftSplitter(nn.Module):
    """
    Splits with softmaxed gating before projection.
    """
    def __init__(self, output_dim1, output_dim2):
        super(GatedResidualSoftSplitter, self).__init__()
        proj_dim = max(output_dim1, output_dim2)
        self.alpha1 = nn.Parameter(torch.randn(1)) # gating parameter for projector1
        self.out_proj1 = nn.Linear(proj_dim, output_dim1)
        self.out_proj2 = nn.Linear(proj_dim, output_dim2)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output tensor from the combiner. Shape (batch_size, seq_len, proj_dim)
        """
        output_dim1 = self.out_proj1.weight.shape[0] # weight.shape = (output_dim1, proj_dim)
        output_dim2 = self.out_proj2.weight.shape[0]
        alpha1 = self.alpha1
        alpha2 = 1 - alpha1
        alpha_sf1, alpha_sf2 = torch.softmax(torch.cat([alpha1, alpha2]), dim=0)

        proj1_out = (1 - alpha_sf1) * self.out_proj1(x) + alpha_sf1 * x[:, :, :output_dim1] 
        proj2_out = (1 - alpha_sf2) * self.out_proj2(x) + alpha_sf2 * x[:, :, :output_dim2]
        return proj1_out, proj2_out
    

#### Gated Residual Combiner and Splitters (without softmax before proj) ####

class GatedResidualCombiner(nn.Module):
    """
    Combiner two inbound projectors for outputs from two model blocks. 
    The projected outputs are added in a weighted fashion. The combined output 
    is passed to the Splitter or LM head.
    
    The projected output has dimension as the maximum of the two input dimensions.   
    """
    def __init__(self, input_dim1, input_dim2):
        super(GatedResidualCombiner, self).__init__()
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
        
        alpha1_sf, alpha2_sf = torch.softmax(torch.cat([self.alpha1, self.alpha2]), dim=0)
        combined = alpha1_sf * x1_proj_out + alpha2_sf * x2_proj_out
        return combined


class GatedResidualSplitter(nn.Module):
    """
    Splitter is used to split the output of the intermediate combiner 
    into two parts to be passed to the two model blocks.
    """
    def __init__(self, output_dim1, output_dim2):
        super(GatedResidualSplitter, self).__init__()
        proj_dim = max(output_dim1, output_dim2)
        self.alpha1 = nn.Parameter(torch.ones(1)) # gating parameter for projector1
        self.alpha2 = nn.Parameter(torch.ones(1)) # gating parameter for projector2
        self.out_proj1 = nn.Linear(proj_dim, output_dim1)
        self.out_proj2 = nn.Linear(proj_dim, output_dim2)


    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output tensor from the combiner. Shape (batch_size, seq_len, proj_dim)
        """
        output_dim1 = self.out_proj1.weight.shape[0] # weight.shape = (output_dim1, proj_dim)
        output_dim2 = self.out_proj2.weight.shape[0]
        proj1_out = (1 - self.alpha1) * self.out_proj1(x) + self.alpha1 * x[:, :, :output_dim1] 
        proj2_out = (1 - self.alpha2) * self.out_proj2(x) + self.alpha2 * x[:, :, :output_dim2]
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
    

#### Residual Combiner and Splitters (has projection+residual, but no gating) ####

class ResidualCombiner(nn.Module):
    """
    ResidualCombiner combines residual projected outputs from 
    two model blocks with equal weight.  
    """
    def __init__(self, input_dim1, input_dim2):
        super(ResidualCombiner, self).__init__()
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.proj_dim = max(input_dim1, input_dim2)
        self.in_proj1 = nn.Linear(input_dim1, self.proj_dim)
        self.in_proj2 = nn.Linear(input_dim2, self.proj_dim)

    def forward(self, x1, x2):
        """
        Args:
            x1 (torch.Tensor): The output tensor from the first model block. Shape (batch_size, seq_len, input_dim)
            x2 (torch.Tensor): The output tensor from the second model block. Shape (batch_size, seq_len, input_dim)
            
        """
        # verify input dimensions
        assert x1.shape[2] == self.input_dim1, f"Expected input_dim1={self.input_dim1}, got {x1.shape[2]}"
        assert x2.shape[2] == self.input_dim2, f"Expected input_dim2={self.input_dim2}, got {x2.shape[2]}"
        
        # projected embeddings
        x1_proj = self.in_proj1(x1)
        x2_proj = self.in_proj2(x2)
        
        # pad input to match output dim
        if x1.shape[2] < self.proj_dim:
            padding = torch.zeros((x1.shape[0], x1.shape[1], self.proj_dim - x1.shape[2]), device=x1.device)
            x1 = torch.cat([x1, padding], dim=2)
            
        if x2.shape[2] < self.proj_dim:
            padding = torch.zeros((x2.shape[0], x2.shape[1], self.proj_dim - x2.shape[2]), device=x2.device)
            x2 = torch.cat([x2, padding], dim=2)
            
        # residual connection
        x1_proj_out = x1_proj + x1
        x2_proj_out = x2_proj + x2

        # return combined output
        return (x1_proj_out + x2_proj_out) / 2
    
    
class ResidualSplitter(nn.Module):
    """
    ResidualSplitter outputs residual projected input to 
    the two model blocks.
    """
    def __init__(self, output_dim1, output_dim2):
        super(ResidualSplitter, self).__init__()
        self.output_dim1 = output_dim1
        self.output_dim2 = output_dim2
        self.proj_dim = max(output_dim1, output_dim2)
        self.out_proj1 = nn.Linear(self.proj_dim, output_dim1)
        self.out_proj2 = nn.Linear(self.proj_dim, output_dim2)
    
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): The output tensor from the combiner. Shape (batch_size, seq_len, proj_dim)
        """
        assert x.shape[2] == self.proj_dim, f"Expected proj_dim={self.proj_dim}, got {x.shape[2]}"
        
        # calculate projected outputs
        x1_proj = self.out_proj1(x)
        x2_proj = self.out_proj2(x)
        
        # add residual with truncated input
        x1_proj_out = x1_proj + x[:, :, :self.output_dim1]
        x2_proj_out = x2_proj + x[:, :, :self.output_dim2]
        
        return x1_proj_out, x2_proj_out