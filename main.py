import argparse
from typing import NamedTuple

from trainer import Trainer
from mamba.mamba_lmhead import MambaTextClassification
from hybrid.hybrid_model import HybridModelTextClassification
from hybrid.model_zoo import get_mamba_causal, get_gpt_neo_causal

from transformers import ( 
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    MambaForCausalLM
)


# Train a transformer model with Trainer 
def train_gpt_neo(args):
    model_name = 'EleutherAI/gpt-neo-125M'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model =  AutoModelForSequenceClassification.from_pretrained(
            model_name, pad_token_id=tokenizer.pad_token_id, 
            num_labels=2, id2label=id2label, label2id=label2id)

    for param in model.transformer.parameters():
        param.requires_grad = False

    dataset_name = "imdb"
    param_list = model.score.parameters()

    Trainer.train(model, dataset_name, param_list, args)

def train_mamba(args):
    model_name = "state-spaces/mamba-130m-hf"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_classes = 2
    model = MambaTextClassification(model_name, 2)
    
    print("model", model)
    
    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, dataset_name, param_list, args)

def train_hybrid(args):
    trans_model = get_gpt_neo_causal()
    mamba_model = get_mamba_causal()
    
    model = HybridModelTextClassification(trans_model, mamba_model, 2)
    
    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, dataset_name, param_list, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model with Trainer")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for training")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--filepath", type=str, default="models/saved.ptr", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    
    args = parser.parse_args()
    args.num_labels = 2
    args.hidden_dropout_prob = 0.1
    args.option = None
    
    print("args:", args)
    # train_gpt_neo(args)
    # train_mamba(args)
    train_hybrid(args)