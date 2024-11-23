import torch
from transformers import AutoTokenizer

from hybrid.hybrid_model import HybridModel
from hybrid.model_zoo import get_gpt_neo_causal, get_mamba_causal

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

    # Take the token with the maximum probability
    hybrid_max_prob_token = torch.argmax(hybrid_output.logits, dim=-1)

    # Decode the token to text
    hybrid_decoded_token = trans_tokenizer.decode(hybrid_max_prob_token[0], skip_special_tokens=True)
    print("Hybrid token with max probability:", hybrid_decoded_token)
    
if __name__ == "__main__":
    test_hybrid_infer()
