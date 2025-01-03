from squad_dataset import SquadPrerprocessing
from transformers import AutoTokenizer

def test_get_answer_template_end_pos():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token # for open-ended generation
    max_length = 32
    encoding = tokenizer("Context: A dog Question: Dog happy? Answer: yes happy", 
        padding=False, truncation=True, max_length=max_length, return_tensors='pt')
    encoding = {k: v.squeeze() for k, v in encoding.items()}
    
    answer_template = " Answer:"
    squadprep = SquadPrerprocessing(tokenizer, max_length)
    pos = squadprep._get_answer_template_end_pos(encoding, answer_template)
    assert pos == 10
    print("test_get_answer_template_end_pos passed")

# text: Context: A dog Question: Dog happy? Answer: yes happy
# whole encoding: {'input_ids': tensor([[21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#             25,  3763,  3772]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]])}
# answer encoding: {'input_ids': tensor([[23998,    25]]), 'attention_mask': tensor([[1, 1]])}

def test_chunk_answer():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token # for open-ended generation
    squadprep = SquadPrerprocessing(tokenizer, 16, split='validation')
    
    squad_dataset = {
        'context': "A dog",
        'question': "Dog happy?",
        'answers': {'text': ["yes happy"]}
    }

    squadprep._chunk_answer(squad_dataset)
    print("test_chunk_answer passed")

# encoding {'input_ids': tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25,  3763,  3772, 50256]), 'attention_mask': tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])}
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25])
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25, 50256, 50256, 50256, 50256, 50256])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
# labels tensor([   25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,    25,
#          3763])
# labels tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 3763, -100,
#         -100, -100, -100, -100])
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25,  3763])
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25,  3763, 50256, 50256, 50256, 50256])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0])
# labels tensor([   25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,    25,
#          3763,  3772])
# labels tensor([-100, -100, -100, -100, -100, -100, -100, -100, -100, -100, 3763, 3772,
#         -100, -100, -100, -100])
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25,  3763,  3772])
# input_ids tensor([21947,    25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,
#            25,  3763,  3772, 50256, 50256, 50256])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
# attention_mask tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0])
# labels tensor([   25,   317,  3290, 18233,    25,  8532,  3772,    30, 23998,    25,
#          3763,  3772, 50256])
# labels tensor([ -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,  -100,
#          3763,  3772, 50256,  -100,  -100,  -100])

def test_get_squad_causal_dataset():
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-neo-125M")
    tokenizer.pad_token = tokenizer.eos_token # for open-ended generation
    num_samples = 5
    squadprep = SquadPrerprocessing(tokenizer, 256, num_samples=num_samples)
    items = squadprep.get_squad_causal_dataset()
    assert len(items) == num_samples
    print("test_get_squad_causal_dataset passed")
    
    
if __name__ == "__main__":
    test_get_answer_template_end_pos()
    test_chunk_answer()
    test_get_squad_causal_dataset()
    