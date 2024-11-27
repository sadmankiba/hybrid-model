import torch
from transformers import AutoTokenizer

from hybrid.hybrid_model import HybridModel
from hybrid.model_zoo import get_gpt_neo_causal, get_mamba_causal
from utils.params import count_parameters


def test_hybrid_model_param_count():
    trans_model = get_gpt_neo_causal()
    mamba_model = get_mamba_causal()
    hybrid_model = HybridModel(trans_model, mamba_model, n_hybrid_blocks=12)
    for param in hybrid_model.trans_model.parameters():
        param.requires_grad = False
        
    for param in hybrid_model.mamba_model.parameters():
        param.requires_grad = False
        
    n_dim_trans = trans_model.transformer.wte.weight.shape[-1]
    n_dim_mamba = mamba_model.backbone.embeddings.weight.shape[-1]
    assert n_dim_trans == 768
    assert n_dim_mamba == 768
    
    trans_total_params, trans_trainable_params = count_parameters(trans_model)
    mamba_total_params, mamba_trainable_params = count_parameters(mamba_model)
    hybrid_total_params, hybrid_trainable_params = count_parameters(hybrid_model)
    assert trans_total_params == 125_198_592
    assert trans_trainable_params == 0
    assert mamba_total_params == 129_135_360
    assert mamba_trainable_params == 0
    
    # trainable : 12 * (2 x (768 * 768 + 768) + 2 x (768 * 50257 + 5057)) = 66,996,049
    assert hybrid_trainable_params == 66_996_049
    assert hybrid_total_params == trans_total_params + mamba_total_params + hybrid_trainable_params


def test_hybrid_infer():
    prompt = "Hey how are you doing?"
    trans_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')

    trans_model_inputs = trans_tokenizer(prompt, return_tensors="pt").to('cpu')

    # Initialize the input
    trans_input_ids = trans_model_inputs['input_ids']
    trans_model = get_gpt_neo_causal()
    mamba_model = get_mamba_causal()
    hybrid_model = HybridModel(trans_model, mamba_model)
    hybrid_output = hybrid_model(trans_input_ids)
    assert hybrid_output.logits.shape == (1, 6, 50257)

    # Take the token with the maximum probability
    hybrid_max_prob_token = torch.argmax(hybrid_output.logits, dim=-1)

    # Decode the token to text
    hybrid_decoded_token = trans_tokenizer.decode(hybrid_max_prob_token[0], skip_special_tokens=True)
    print("Hybrid token with max probability:", hybrid_decoded_token)
    
if __name__ == "__main__":
    test_hybrid_infer()
