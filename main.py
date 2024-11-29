import argparse
from typing import NamedTuple

import torch
from trainer import Trainer
from mamba.mamba_lmhead import MambaTextClassification
from hybrid.hybrid_model import HybridModelTextClassification
from hybrid.model_zoo import (
    get_mamba_causal, 
    get_gpt_neo_causal,
    get_gpt_neo_initd,
    get_gpt_neo_initd_seq_cls,
    get_mamba_initd,
)

from transformers import ( 
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    MambaForCausalLM
)

gpt_neo_model_checkpoint = "EleutherAI/gpt-neo-125M"
mamba_model_checkpoint = "state-spaces/mamba-130m-hf"


# Train a transformer model with Trainer 
def train_gpt_neo(args):
    model_name = gpt_neo_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    model =  AutoModelForSequenceClassification.from_pretrained(
            model_name, pad_token_id=tokenizer.pad_token_id, 
            num_labels=2, id2label=id2label, label2id=label2id)
    print("model:", model)
    
    for param in model.transformer.parameters():
        param.requires_grad = False

    dataset_name = "imdb"
    param_list = model.score.parameters()

    Trainer.train(model, model_name, dataset_name, param_list, args)

def train_mamba(args):
    model_name = mamba_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_classes = 2
    model = MambaTextClassification(model_name, 2)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    print("model:", model)
    
    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, model_name, dataset_name, param_list, args)

def train_gpt_neo_initd(args):
    model_name = gpt_neo_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    config = AutoConfig.from_pretrained(gpt_neo_model_checkpoint, 
            num_layers=args.num_layers, 
            hidden_size=args.hidden_size, 
            num_heads=args.num_heads,
            pad_token_id=tokenizer.pad_token_id, 
            num_labels=args.num_labels, id2label=id2label, label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_config(config)
    
    print("model:", model)
    
    for param in model.transformer.parameters():
        param.requires_grad = False

    dataset_name = "imdb"
    param_list = model.score.parameters()

    Trainer.train(model, model_name, dataset_name, param_list, args)

def train_hybrid(args):
    gpt_neo_tokenizer_id = 'EleutherAI/gpt-neo-125M'
    trans_model = get_gpt_neo_causal()
    mamba_model = get_mamba_causal()
    
    model = HybridModelTextClassification(trans_model, mamba_model, 2)
    print("model:", model)
    for param in model.hybrid_model.trans_model.parameters():
        param.requires_grad = False
        
    for param in model.hybrid_model.mamba_model.parameters():
        param.requires_grad = False
    
    
    dataset_name = "imdb"
    param_list = model.parameters() # TODO: Confirm it works
    Trainer.train(model, gpt_neo_tokenizer_id, dataset_name, param_list, args)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a transformer model with Trainer")
    # Training
    parser.add_argument("--epochs", type=float, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--filepath", type=str, default="models/saved.ptr", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=0, help="Log training loss every n steps")
    parser.add_argument("--train_size", type=int, default=0, help="Number of training examples")
    parser.add_argument("--eval_size", type=int, default=0, help="Number of dev examples")
    parser.add_argument("--use_gpu", action="store_true", help="Whether to use GPU for training")
    
    # Which models to run 
    parser.add_argument("--run_trans", action="store_true", help="Run the transformers model")
    parser.add_argument("--run_mamba", action="store_true", help="Run the Mamba model")
    parser.add_argument("--run_hybrid", action="store_true", help="Run the Hybrid model")
    parser.add_argument("--run_gpt_neo_initd", action="store_true", help="Run the GPT-Neo model")
    
    # Initialized models
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    
    args = parser.parse_args()
    args.num_labels = 2
    args.hidden_dropout_prob = 0.1
    args.option = None
    
    print("args:", args)
    
    if args.run_trans:
        train_gpt_neo(args)
    
    if args.run_mamba:
        train_mamba(args)
        
    if args.run_hybrid:
        train_hybrid(args)
        
    if args.run_gpt_neo_initd:
        train_gpt_neo_initd(args)