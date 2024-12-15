from transformers import ( 
    AutoTokenizer, 
    AutoModelForCausalLM,
    MambaForCausalLM,
)

gpt_neo_model_checkpoint = "EleutherAI/gpt-neo-125M"
mamba_model_checkpoint = "state-spaces/mamba-130m-hf"

# Load pretrained and initialized models
 
def get_gpt_neo_causal():
    trans_model = AutoModelForCausalLM.from_pretrained(
        gpt_neo_model_checkpoint,
        torch_dtype="auto",
    )
    return trans_model

def get_gpt_neo_tokenizer():
    return AutoTokenizer.from_pretrained(gpt_neo_model_checkpoint)

# gpt-neo tokenizer(text = None, add_special_tokens = True, padding = False, truncation = None, max_length = None, 
# stride = 0, is_split_into_words = False, pad_to_multiple_of = None, padding_side = None, return_tensors = None, 
# return_token_type_ids = None, verbose = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding>

def get_mamba_causal():
    mamba_model = MambaForCausalLM.from_pretrained(mamba_model_checkpoint)
    return mamba_model

def get_mamba_tokenizer():
    return AutoTokenizer.from_pretrained(mamba_model_checkpoint)

# default
# mamba_tokenizer(text: = None, add_special_tokens = True, padding = False, truncation = None, max_length = None, stride = 0, 
# is_split_into_words = False, pad_to_multiple_of = None, padding_side = None, return_tensors = None, return_token_type_ids = None, 
# return_attention_mask = None, verbose: bool = True, **kwargs) -> transformers.tokenization_utils_base.BatchEncoding

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
    
# Output
# Mamba model output: ["Hey how are you doing?\n\nI'm so glad you're here."]
# GPT-Neo model output: ["Hey how are you doing?\n\nI'm doing a lot of research on the internet and I'm not sure if I'm"]