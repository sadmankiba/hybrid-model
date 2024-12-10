import pytest
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from trainer import Trainer
from dataset.squad_dataset import SquadDataset


class Args:
    def __init__(self):
        self.device = 0
        self.use_gpu = torch.cuda.is_available()
        self.max_length = 256
        self.batch_size = 4
        self.max_new_tokens = 20
        self.eval_size = 20
        self.train_size = 20
        self.epochs = 1
        self.use_amp = True
        self.lr = 5e-4
        self.weight_decay = 0.01
        self.log_interval = 3

@pytest.fixture
def model_and_tokenizer():
    args = Args()
    tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neo-125M')
    model = AutoModelForCausalLM.from_pretrained('EleutherAI/gpt-neo-125M').to(args.device)
    return model, tokenizer

@pytest.fixture
def squad_val_dataloader(model_and_tokenizer):
    args = Args()
    model, tokenizer = model_and_tokenizer
    val_ds = SquadDataset(tokenizer, args.max_length, split="validation", num_samples=args.eval_size)
    val_dl = DataLoader(val_ds, shuffle=True, batch_size=args.batch_size)
    return val_dl

@pytest.fixture
def squad_train_dataloader(model_and_tokenizer):
    args = Args()
    model, tokenizer = model_and_tokenizer
    train_ds = SquadDataset(tokenizer, args.max_length, split="train", num_samples=args.eval_size)
    train_dl = DataLoader(train_ds, shuffle=True, batch_size=args.batch_size)
    return train_dl

def test_eval_squad(model_and_tokenizer, squad_dataloader):
    model, tokenizer = model_and_tokenizer
    args = Args()
    results = Trainer.eval_squad(squad_val_dataloader, model, tokenizer, args)
    print("results:", results)
    assert "token_exact_match" in results
    assert "bleu" in results
    assert "rouge" in results
    assert 0 <= results["token_exact_match"] <= 1
    assert 0 <= results["bleu"] <= 1
    assert 0 <= results["rouge"] <= 1


# question: ['What century did the Normans first gain their separate identity?', 'In what country is Normandy located?', 'From which countries did the Norse originate?', 'When were the Normans in Normandy?'] 
# answer: ['10th century', 'France', 'Denmark, Iceland and Norway', '10th and 11th centuries']
# labels: tensor([[-100, -100, -100,  ..., -100, -100, -100],
#         [-100, -100, -100,  ..., -100, -100, -100],
#         [-100, -100, -100,  ..., -100, -100, -100],
#         [-100, -100, -100,  ..., -100, -100, -100]])
# all_preds: [[  262   262   262   262   262   262   262   262   262   262   262   262
#     262   262   262   262   262   262   262   262]
#  [  262 43231   262   262 43231   262   262 43231   262   262   262   262
#     262   262   262   262   262   262   262   262]
#  [  477  2031   262 38419   262   262   262   262   262   262   262   262
#     262   262   262   262   262   262   262   262]
#  [  262   262 43231 43231   262   262 43231   262   262   262   262   262
#     262   262   262   262   262   262   262   262]]
# response: [' the the the the the the the the the the the the the the the the the the the the', ' the Normandy the the Normandy the the Normandy the the the the the the the the the the the the', ' all Europe the Norse the the the the the the the the the the the the the the the the', ' the the Normandy Normandy the the Normandy the the the the the the the the the the the the the']
# valid_indices: (tensor([0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]), tensor([180, 175, 176, 177, 178, 179, 180, 176, 177, 178, 179]))
# labels_first_valid_pos: tensor([180, 175, 176, 176])
# pred_indices: (tensor([0, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3]), tensor([0, 0, 0, 1, 2, 3, 4, 0, 1, 2, 3]))
# b_labels_valid: tensor([  838,  4881, 16490,    11, 17333,   290, 15238,   838,   400,   290,
#          1367])
# preds_valid: [  262   262   477  2031   262 38419   262   262   262 43231 43231]
# eval squad: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████| 5/5 [00:19<00:00,  3.85s/it]
# token_exact_matches: [0.09090909090909091, 0.0, 0.06666666666666667, 0.0, 0.0]
# bleus: [0.0, 0.0, 0.0, 0.0, 0.0]
# rouges: [tensor(0.), tensor(0.), tensor(0.), tensor(0.), tensor(0.)]
# results: {'token_exact_match': 0.03151515151515152, 'bleu': 0.0, 'rouge': tensor(0.)}

def test_train_squad(model_and_tokenizer):
    model, tokenizer = model_and_tokenizer
    args = Args()
    Trainer.train_squad(model, tokenizer, args)
    
#     b_labels: tensor([[   25, 37361, 32682,  ...,  -100,  -100,  -100],
#         [   25, 37361, 32682,  ...,  -100,  -100,  -100],
#         [   25, 37361, 32682,  ...,  -100,  -100,  -100],
#         [   25, 37361, 32682,  ...,  -100,  -100,  -100]], device='cuda:0')
# preds: tensor([[   25, 50093, 32682,  ..., 42076, 25418, 42076],
#         [   25, 50093, 32682,  ...,  6303,  4592,  6303],
#         [   25, 50093, 32682,  ..., 36670, 36670, 36670],
#         [   25, 50093, 32682,  ..., 17886, 17886, 17886]], device='cuda:0')
# loss: tensor(2.1641, device='cuda:0', dtype=torch.float16,
#        grad_fn=<NllLoss2DBackward0>)
    