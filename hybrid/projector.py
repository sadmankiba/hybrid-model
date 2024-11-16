import torch

# Combiner and Splitter follow the Manticore model architecture

class Combiner(torch.nn.Module):
    """
    Combiner two inbound projectors for outputs from two model blocks. 
    The projected outputs are added in a weighted fashion. The combined output 
    is passed to the Splitter or LM head.
    
    The projected output has dimension as the maximum of the two input dimensions.   
    """
    def __init__(self, input_dim1, input_dim2):
        super(Combiner, self).__init__()
        proj_dim = max(input_dim1, input_dim2)
        self.in_proj1 = torch.nn.Linear(input_dim1, proj_dim)
        self.in_proj2 = torch.nn.Linear(input_dim2, proj_dim)
    
    def forward(self, x1, x2):
        x1_proj = self.in_proj1(x1)
        x2_proj = self.in_proj2(x2)
        combined = x1_proj + x2_proj
        return combined

class Splitter(torch.nn.Module):
    """
    Splitter is used to split the output of the intermediate combiner 
    into two parts to be passed to the two model blocks.
    """
    def __init__(self, input_dim1, input_dim2):
        super(Splitter, self).__init__()
        proj_dim = max(input_dim1, input_dim2)
        self.out_proj1 = torch.nn.Linear(proj_dim, input_dim1)
        self.out_proj2 = torch.nn.Linear(proj_dim, input_dim2)
    
    def forward(self, x):
        return self.out_proj1(x), self.out_proj2(x)