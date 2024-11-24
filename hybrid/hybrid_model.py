import torch 
from transformers import AutoTokenizer
from projector import Combiner, Splitter
from torch import nn
class HybridModel(nn.Module):
    """
    Currently, only designed for mamba-130m and gpt-neo-125m. 
    
    12 hybrid blocks. Each hybrid block has one transformer layer
    and two mamba layers.
    """
    def __init__(self, transformer_model, mamba_model):
        super(HybridModel, self).__init__()
        self.transformer_model = transformer_model

        self.mamba_model = mamba_model 
        dim1 = self.transformer_model.wte.weight.shape[-1]
        dim2 = self.mamba_model.embeddings.weight.shape[-1]
        
        # Create intermediate layers and LM head
        self.combiners = torch.nn.ModuleList([Combiner(dim1, dim2) for _ in range(12)])
        self.splitters = torch.nn.ModuleList([Splitter(dim1, dim2) for _ in range(12)])
        self.proj_dim = max(dim1, dim2)

        self.hybrid_lm_head = torch.nn.Linear(self.proj_dim, self.transformer_model.wte.weight.shape[0])

    def forward(self, input_data):
        # Get the transformer and mamba model layers
        trans_layers = self.transformer_model.h
        mamba_layers = self.mamba_model.layers

        # Pass through word and position embeddings
        trans_t_emb = self.transformer_model.wte(input_data)
        trans_p_emb = self.transformer_model.wpe(torch.tensor([[i for i in range(input_data.shape[1])]]))
        trans_input_emb = trans_t_emb + trans_p_emb
        print("Trans input emb shape", trans_input_emb.shape)

        # Pass the input through each block and intermediate layers
        combined_emb = trans_input_emb
        # use_ VAR instead of literal
        for i in range(12):
            trans_input_emb, mamba_input_embeds = self.splitters[i](combined_emb)
            trans_input_emb = trans_layers[i](trans_input_emb)[0]
            mamba_input_embeds = mamba_layers[2*i](mamba_input_embeds)
            mamba_input_embeds = mamba_layers[2*i+1](mamba_input_embeds)
            combined_emb = self.combiners[i](trans_input_emb, mamba_input_embeds)
            
        print(f"Output of combined: {combined_emb.shape}")
            
        # No norm layer for now 
        return self.hybrid_lm_head(combined_emb)
    
    def generate(self, input_ids, attention_mask=None, max_length=50):
            generated_ids = input_ids
            for _ in range(max_length):
                logits = self.forward(generated_ids, attention_mask)
                next_token_logits = logits[:, -1, :]
                next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
                generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
                if next_token_id.item() == self.transformer_model.config.eos_token_id:
                    break
            return generated_ids



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




# class HybridTextGenerator(nn.Module):
#     def __init__(self,transformer_model, mamba_model):
#         super(HybridTextGenerator, self).__init__()
#         self.hybrid_model = HybridModel(transformer_model, mamba_model)
#         self.lm_head = nn.Linear(self.hybrid_model.proj_dim, transformer_model.config.vocab_size, bias=False)
#     def forward(self, input_ids, attention_mask =None):
#         hybrid_output = self.hybrid_model(input_ids, attention_mask)
#         logits = self.lm_head(hybrid_output)
#         return logits

#     def generate(self, input_ids, attention_mask=None, max_length=50):
#         generated_ids = input_ids
#         for _ in range(max_length):
#             logits = self.forward(generated_ids, attention_mask)
#             next_token_logits = logits[:, -1, :]
#             next_token_id = torch.argmax(next_token_logits, dim=-1, keepdim=True)
#             generated_ids = torch.cat([generated_ids, next_token_id], dim=-1)
#             if next_token_id.item() == self.transformer_model.config.eos_token_id:
#                 break
#         return generated_ids


