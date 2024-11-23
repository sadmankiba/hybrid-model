import torch
from transformers import MambaForCausalLM, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load tokenizer and pretrained model

def get_mamba_causal():
    mamba_model = MambaForCausalLM.from_pretrained("state-spaces/mamba-130m-hf")
    return mamba_model

 
def get_gpt_neo_causal():
    model_name = 'EleutherAI/gpt-neo-125M'
    trans_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype="auto",
        device_map="cpu"
    )
    return trans_model


def test_mamba_model():
    mamba_tokenizer = AutoTokenizer.from_pretrained("state-spaces/mamba-130m-hf")
    input_ids = mamba_tokenizer("Hey how are you doing?", return_tensors= "pt")["input_ids"]

    out = get_mamba_causal().generate(input_ids, max_new_tokens=10)
    print(mamba_tokenizer.batch_decode(out))

def test_trans_model():
    trans_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    
    prompt = "Hey how are you doing?"
    model_inputs = trans_tokenizer(prompt, return_tensors="pt").to('cpu')
    print('model_inputs:', model_inputs)

    generated_ids = get_gpt_neo_causal().generate(
        **model_inputs,
        max_new_tokens=20
    )

    # Decode the generated tokens to text
    response = trans_tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    print("response:", response)


if __name__ == "__main__":
    test_mamba_model()
    test_trans_model()