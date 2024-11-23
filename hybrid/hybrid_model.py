import torch 
from transformers import AutoTokenizer
from model_zoo import trans_model, mamba_model
from projector import Combiner, Splitter

class HybridModel(torch.nn.Module):
    """
    Currently, only designed for mamba-130m and gpt-neo-125m. 
    
    12 hybrid blocks. Each hybrid block has one transformer layer
    and two mamba layers.
    """
    def __init__(self, transfomer_model, mamba_model):
        super(HybridModel, self).__init__()

        dim1 = transfomer_model.transformer.wte.weight.shape[-1]
        dim2 = mamba_model.backbone.embeddings.weight.shape[-1]
        
        # Create intermediate layers and LM head
        self.combiners = torch.nn.ModuleList([Combiner(dim1, dim2) for _ in range(12)])
        self.splitters = torch.nn.ModuleList([Splitter(dim1, dim2) for _ in range(12)])
        self.proj_dim = max(dim1, dim2)
        self.hybrid_lm_head = torch.nn.Linear(self.proj_dim, trans_model.lm_head.out_features)

    def forward(self, input_data):
        # Get the transformer and mamba model layers
        trans_layers = trans_model.transformer.h
        mamba_layers = mamba_model.backbone.layers

        # Pass through word and position embeddings
        trans_t_emb = trans_model.transformer.wte(input_data)
        trans_p_emb = trans_model.transformer.wpe(torch.tensor([[i for i in range(input_data.shape[1])]]))
        trans_input_emb = trans_t_emb + trans_p_emb
        print("Trans input emb shape", trans_input_emb.shape)

        # Pass the input through each block and intermediate layers
        combined_emb = trans_input_emb
        for i in range(12):
            trans_input_emb, mamba_input_embeds = self.splitters[i](combined_emb)
            trans_input_emb = trans_layers[i](trans_input_emb)[0]
            mamba_input_embeds = mamba_layers[2*i](mamba_input_embeds)
            mamba_input_embeds = mamba_layers[2*i+1](mamba_input_embeds)
            combined_emb = self.combiners[i](trans_input_emb, mamba_input_embeds)
            
        print(f"Output of combined: {combined_emb.shape}")
            
        # No norm layer for now 
        return self.hybrid_lm_head(combined_emb)



def test_hybrid_infer():
    prompt = "Hey how are you doing?"
    trans_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

    trans_model_inputs = trans_tokenizer(prompt, return_tensors="pt").to('cpu')

    # Initialize the input
    trans_input_data = trans_model_inputs['input_ids']
    
    hybrid_model = HybridModel(dim1=trans_model.transformer.wte.weight.shape[-1], 
                                 dim2=mamba_model.backbone.embeddings.weight.shape[-1])
    hybrid_output = hybrid_model(trans_input_data)

    # Take the token with the maximum probability
    hybrid_max_prob_token = torch.argmax(hybrid_output, dim=-1)

    # Decode the token to text
    hybrid_decoded_token = trans_tokenizer.decode(hybrid_max_prob_token[0], skip_special_tokens=True)
    print("Hybrid token with max probability:", hybrid_decoded_token)
    
if __name__ == "__main__":
    test_hybrid_infer()
