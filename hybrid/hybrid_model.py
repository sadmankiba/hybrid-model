from collections import namedtuple

import torch 
from transformers import AutoTokenizer

from .projector import Combiner, Splitter

class HybridModel(torch.nn.Module):
    """
    Currently, only designed for mamba-130m and gpt-neo-125m. 
    
    12 hybrid blocks. Each hybrid block has one transformer layer
    and two mamba layers.
    """
    def __init__(self, transfomer_model, mamba_model, n_hybrid_blocks=12):
        super(HybridModel, self).__init__()
        self.trans_model = transfomer_model
        self.mamba_model = mamba_model
        self.n_blocks = n_hybrid_blocks
        self.n_trans_layers = len(transfomer_model.transformer.h)
        self.n_mamba_layers = len(mamba_model.backbone.layers)
        assert self.n_trans_layers % self.n_blocks == 0
        assert self.n_mamba_layers % self.n_blocks == 0
        dim1 = transfomer_model.transformer.wte.weight.shape[-1]
        dim2 = mamba_model.backbone.embeddings.weight.shape[-1]
        
        # Create intermediate layers and LM head
        self.combiners = torch.nn.ModuleList([Combiner(dim1, dim2) for _ in range(n_hybrid_blocks)])
        self.splitters = torch.nn.ModuleList([Splitter(dim1, dim2) for _ in range(n_hybrid_blocks)])
        self.proj_dim = max(dim1, dim2)
        self.hybrid_lm_head = self.trans_model.lm_head


    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids (torch.Tensor): The input tensor of shape (batch_size, seq_len)
            attention_mask (torch.Tensor): The attention mask tensor of shape (batch_size, seq_len)
        """
        # Get the transformer and mamba model layers
        trans_layers = self.trans_model.transformer.h
        mamba_layers = self.mamba_model.backbone.layers

        # Pass through word and position embeddings
        trans_t_emb = self.trans_model.transformer.wte(input_ids)
        trans_p_emb = self.trans_model.transformer.wpe(torch.tensor([[i for i in range(input_ids.shape[1])]]).to(input_ids.device))
        trans_input_emb = trans_t_emb + trans_p_emb

        # Pass the input through each block and intermediate layers
        combined_emb = trans_input_emb  # (batch_size, seq_len, proj_dim)
        hidden_states = (combined_emb, )
        trans_layers_per_block = self.n_trans_layers // self.n_blocks
        mamba_layers_per_block = self.n_mamba_layers // self.n_blocks
        for i in range(self.n_blocks):
            trans_input_emb, mamba_input_embeds = self.splitters[i](combined_emb)
            for j in range(trans_layers_per_block):
                trans_input_emb = trans_layers[trans_layers_per_block * i + j](trans_input_emb)[0]    
            for k in range(mamba_layers_per_block):
                mamba_input_embeds = mamba_layers[mamba_layers_per_block * i + k](mamba_input_embeds)
            
            combined_emb = self.combiners[i](trans_input_emb, mamba_input_embeds)
            hidden_states += (combined_emb, )
        
        # No norm layer for now 
        lm_head_out = self.hybrid_lm_head(combined_emb)
        
        Output = namedtuple("Output", ["hidden_states", "logits"])
        return(Output(hidden_states=hidden_states, logits=lm_head_out))


class HybridModelTextClassification(torch.nn.Module):
    def __init__(self, transformer_model, mamba_model, n_classes):
        super(HybridModelTextClassification, self).__init__()
        self.hybrid_model = HybridModel(transformer_model, mamba_model)
        self.cls_head = torch.nn.Linear(self.hybrid_model.proj_dim, n_classes)
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        output = self.hybrid_model(input_ids, attention_mask)
        last_hidden_states = output.hidden_states[-1] 
        mean_hidden_states = last_hidden_states.mean(dim=1)
        logits = self.cls_head(mean_hidden_states)
        
        if labels is None:
            ClassificationOutput = namedtuple("ClassificationOutput", ["logits"])
            return ClassificationOutput(logits=logits)
        else:
            ClassificationOutput = namedtuple("ClassificationOutput", ["loss", "logits"])
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)
            return ClassificationOutput(loss=loss, logits=logits)