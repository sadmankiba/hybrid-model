import argparse
import os 

import torch

from trainer import Trainer
from mamba_model.mamba_lmhead import MambaTextClassification
from hybrid.hybrid_model import HybridModelTextClassification, HybridModel, MambaFormer
from hybrid.model_zoo import (
    get_mamba_causal, 
    get_gpt_neo_causal,
)
from mad.configs import MADConfig, ModelConfig,ImdbConfig

from torch.utils.data import DataLoader
from transformers import ( 
    AutoConfig,
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    AutoModelForCausalLM,
    GPTNeoForSequenceClassification,
    GPTNeoConfig
)

from trainer import Trainer
from dataset.squad_dataset import SquadDataset
from mamba_model.mamba_lmhead import MambaTextClassification
from hybrid.hybrid_model import HybridModelTextClassification, HybridModel, MambaFormer
from hybrid.model_zoo import (
    get_mamba_causal, 
    get_gpt_neo_causal,
    get_gpt_neo_tokenizer,
    get_mamba_tokenizer
)
from mad.configs import MADConfig, ModelConfig,ImdbConfig


gpt_neo_model_checkpoint = "EleutherAI/gpt-neo-125M"
mamba_model_checkpoint = "state-spaces/mamba-130m-hf"

#### Train Pretrained models on Seqclass ####
 
def train_gpt_neo_seqclass_pretrained(args):
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

def train_mamba_seqclass_pretrained(args):
    tokenizer_name = mamba_model_checkpoint
    
    model = MambaTextClassification(tokenizer_name, 
            mamba_model_checkpoint, args.num_labels)
    
    for param in model.backbone.parameters():
        param.requires_grad = False
    
    print("model:", model)
    
    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, tokenizer_name, dataset_name, param_list, args)

def train_hybrid_seqclass_pretrained(args):
    gpt_neo_tokenizer_id = 'EleutherAI/gpt-neo-125M'
    trans_model = get_gpt_neo_causal()
    mamba_model = get_mamba_causal()
    
    num_labels = 2
    model = HybridModelTextClassification(trans_model, mamba_model, args.proj_type, args.num_hybrid_blocks, num_labels)
    print("model:", model)
    for param in model.hybrid_model.trans_model.parameters():
        param.requires_grad = False
        
    for param in model.hybrid_model.mamba_model.parameters():
        param.requires_grad = False
    
    
    dataset_name = "imdb"
    param_list = model.parameters() # TODO: Confirm it works
    Trainer.train(model, gpt_neo_tokenizer_id, dataset_name, param_list, args)

#### Train Initialized models on Seqclass ####

def train_gpt_neo_seqclass_initd(args):
    model_name = gpt_neo_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
        
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    config = AutoConfig.from_pretrained(gpt_neo_model_checkpoint, 
            num_layers=args.num_layers, 
            hidden_size=args.hidden_size, 
            num_heads=args.num_heads,
            pad_token_id=tokenizer.pad_token_id, # 50256 for GPT-Neo tokenizer
            num_labels=args.num_labels, id2label=id2label, label2id=label2id
    )
    model = AutoModelForSequenceClassification.from_config(config)
    print("model:", model)

    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, model_name, dataset_name, param_list, args)


def train_mamba_seqclass_initd(args):
    tokenizer_name = mamba_model_checkpoint
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    config = AutoConfig.from_pretrained(mamba_model_checkpoint,
            num_hidden_layers=args.num_layers, 
            hidden_size=args.hidden_size, 
            pad_token_id=tokenizer.pad_token_id, 
            output_hidden_states=True
    )
    backbone_model = AutoModelForCausalLM.from_config(config)
    model = MambaTextClassification(tokenizer_name, backbone_model, args.num_labels)
    print("model:", model)
    
    dataset_name = "imdb"
    param_list = model.parameters()
    Trainer.train(model, tokenizer_name, dataset_name, param_list, args)

### Train Initialized models on MAD Tasks ###

def get_gpt_neo_causal_initd(model_config):
    config = AutoConfig.from_pretrained(gpt_neo_model_checkpoint, 
        vocab_size=model_config.vocab_size,
        num_layers=model_config.num_trans_layers, 
        hidden_size=model_config.hidden_size, 
        num_heads=model_config.num_heads,
        pad_token_id=0
    )
    model = AutoModelForCausalLM.from_config(config)
    return model

def get_gpt_neo_seq_initd(model_config):
    id2label = {0: "NEGATIVE", 1: "POSITIVE"}
    label2id = {"NEGATIVE": 0, "POSITIVE": 1}

    config = GPTNeoConfig(
            vocab_size=model_config.vocab_size,
            num_layers=model_config.num_trans_layers, 
            hidden_size=model_config.hidden_size, 
            num_heads=model_config.num_heads,
            attention_types=[[['global'], model_config.num_trans_layers]],
            pad_token_id=50256,
            num_labels=model_config.num_labels, id2label=id2label, label2id=label2id)
    model = GPTNeoForSequenceClassification(config)
    model.score = torch.nn.Linear(in_features=128, out_features=2)
    print(model)
    #exit()
    return model


def get_mamba_causal_initd(model_config):
    config = AutoConfig.from_pretrained(mamba_model_checkpoint,
            vocab_size=model_config.vocab_size,
            num_hidden_layers=model_config.num_mamba_layers, 
            hidden_size=model_config.hidden_size, 
            pad_token_id=50256, # eos_token_id from Mamba tokenizer
            output_hidden_states=True
    )
    model = AutoModelForCausalLM.from_config(config)
    return model

def get_mamba_sequence_initd(model_config):
    gpt_neo_tokenizer_id = 'EleutherAI/gpt-neo-125M'
    model = MambaTextClassification(gpt_neo_tokenizer_id, model_config)
    return model

def get_hybrid_causal_initd(model_config):
    trans_model = get_gpt_neo_causal_initd(model_config)
    mamba_model = get_mamba_causal_initd(model_config)
    
    num_blocks = model_config.num_hybrid_blocks
    assert model_config.num_trans_layers % num_blocks == 0
    assert model_config.num_mamba_layers % num_blocks == 0
    model = HybridModel(trans_model, mamba_model, model_config.proj_type, num_blocks)
    print("model:", model)
    
    return model

def get_hybrid_seq_initd(model_config):
    trans_model = get_gpt_neo_seq_initd(model_config)
    mamba_model = get_mamba_sequence_initd(model_config)
    
    num_blocks = model_config.num_hybrid_blocks
    assert model_config.num_trans_layers % num_blocks == 0
    assert model_config.num_mamba_layers % num_blocks == 0
    model = HybridModelTextClassification(trans_model, mamba_model, num_blocks, model_config.num_labels)
    print("model:", model)
    
    return model

def get_mambaformer_initd(model_config):
    trans_model = get_gpt_neo_causal_initd(model_config)
    mamba_model = get_mamba_causal_initd(model_config)
    
    model = MambaFormer(trans_model, mamba_model)
    print("model:", model)
    
    return model

def train_mad(model_type: str):
    mad_config = MADConfig()
    mad_config.update_from_kwargs(vars(args))
    model_config = ModelConfig()
    model_config.update_from_kwargs(vars(args))
    print("mad_config:", mad_config)
    print("model_config:", model_config)
    
    if model_type == "transformers":
        model = get_gpt_neo_causal_initd(model_config)
    elif model_type == "mamba":
        model = get_mamba_causal_initd(model_config)
    elif model_type == "hybrid":
        model = get_hybrid_causal_initd(model_config)
    elif model_type == "mamform":
        model = get_mambaformer_initd(model_config)
    
    results = Trainer.train_mad(model=model, config=mad_config)
    return results

### Train functions ###

def train_imdb(model_type: str):
    imdb_config = ImdbConfig()
    imdb_config.update_from_kwargs(vars(args))
    model_config = ModelConfig()
    model_config.update_from_kwargs(vars(args))
    
    if model_type == "transformers":
        model = get_gpt_neo_seq_initd(model_config)
    elif model_type == "mamba":
        model = get_mamba_sequence_initd(model_config)
        model = get_hybrid_seq_initd(model_config)
    Trainer.train(model, 'EleutherAI/gpt-neo-125M', "imdb", model.parameters(), imdb_config)

def train_imdb_pretrained(model_type: str):
    imdb_config = ImdbConfig()
    imdb_config.update_from_kwargs(vars(args))
    
    if model_type == "transformers":
        model = AutoModelForSequenceClassification.from_pretrained('EleutherAI/gpt-neo-125M')
    elif model_type == "mamba":
        model = MambaTextClassification("state-spaces/mamba-130m-hf","state-spaces/mamba-130m-hf", 2)
    elif model_type == "hybrid":
        model = HybridModelTextClassification(path="RioHassen/manticore-hybrid-gptneo-mamba")
        # for param in model.hybrid_model.parameters():
        #     param.requires_grad = False
    print(model)
    Trainer.train(model, 'EleutherAI/gpt-neo-125M' if model_type!= "mamba" else "state-spaces/mamba-130m-hf", "imdb", model.parameters(), imdb_config)


def train_squad_pretrained(model_type: str):
    if model_type == "transformers":
        model = get_gpt_neo_causal()
        tokenizer = get_gpt_neo_tokenizer()
    elif model_type == "mamba":
        model = get_mamba_causal()
        tokenizer = get_mamba_tokenizer()
    elif model_type == "hybrid":
        trans_model = get_gpt_neo_causal()
        mamba_model = get_mamba_causal()
        tokenizer = get_gpt_neo_tokenizer()
        model = HybridModel(trans_model, mamba_model, args.proj_type, args.num_hybrid_blocks)
        
    Trainer.train_squad(model, tokenizer, args)
    

### Eval functions ###

def eval_squad_pretrained(args, model_type: str):
    if model_type == "transformers":
        model = get_gpt_neo_causal()
        tokenizer = get_gpt_neo_tokenizer()
    elif model_type == "mamba":
        model = get_mamba_causal()
        tokenizer = get_mamba_tokenizer()
    elif model_type == "hybrid":
        trans_model = get_gpt_neo_causal()
        mamba_model = get_mamba_causal()
        tokenizer = get_gpt_neo_tokenizer()
        model = HybridModel(trans_model, mamba_model, args.proj_type, args.num_hybrid_blocks)
    
    squad_ds = SquadDataset(tokenizer, args.max_length, split="validation", num_samples=args.eval_size)
    squad_val_dl = DataLoader(squad_ds, shuffle=True, batch_size=args.batch_size) # make individual predictions
    Trainer.eval_squad(squad_val_dl, model, tokenizer, args)

### Parsing ###

def parse_args():
    parser = argparse.ArgumentParser(description="Train a transformer model with Trainer")
    # Training
    parser.add_argument("--epochs", type=float, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate for training")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay for optimizer")
    parser.add_argument("--filepath", type=str, default="models/saved.ptr", help="Path to save the trained model")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--log_interval", type=int, default=0, help="Log training loss every n steps")
    parser.add_argument("--eval_interval", type=int, default=0, help="Run evaluation every n steps")
    parser.add_argument("--train_size", type=int, default=0, help="Number of training examples")
    parser.add_argument("--eval_size", type=int, default=0, help="Number of dev examples")
    parser.add_argument("--max_length", type=int, default=256, help="Max length of input sequence")
    parser.add_argument("--use_gpu", type=bool, default=True, help="Use GPU?")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument("--use_amp", type=bool, default=True, help="Use automatic mixed precision")
    parser.add_argument("--output_file", type=str, default="results.txt", help="File to save results")
    parser.add_argument("--save_model", type=bool, default=False, help="Save the model")
    
    # Inference 
    parser.add_argument("--max_new_tokens", type=int, default=20, help="Max new tokens to generate")
    
    # Which models and tasks to run
    parser.add_argument("--task", type=str, default="seqclass", 
        help="Task to run. pret_seqclass | initd_seqclass | mad | initd_imdb | eval_squad | tune_squad") 
    parser.add_argument("--model", type=str, default="transformers", 
        help="Model to run. transformers | mamba | hybrid | mamform")  
    parser.add_argument("--run_hybrid_pretrained", action="store_true", help="Run hybrid from pretrained")
    
    # Initialized models
    parser.add_argument("--num_layers", type=int, default=12, help="Number of layers for the model")
    parser.add_argument("--num_trans_layers", type=int, default=12, help="Number of transformer layers for the model")
    parser.add_argument("--num_mamba_layers", type=int, default=12, help="Number of mamba layers for the model")
    parser.add_argument("--hidden_size", type=int, default=768, help="Hidden size for the model")
    parser.add_argument("--num_heads", type=int, default=12, help="Number of heads for the model")
    
    # Hybrid model
    parser.add_argument("--num_hybrid_blocks", type=int, default=1, help="Number of hybrid model blocks")
    parser.add_argument("--proj_type", type=str, default="res", help="Projection type for hybrid: null, res, gres")
    
    # MAD Tasks
    parser.add_argument('--mad_task', type=str, default='in-context-recall')
    parser.add_argument('--vocab_size', type=int, default=16)
    parser.add_argument('--seq_len', type=int, default=128)
    parser.add_argument('--frac_noise', type=float, default=0.0)
    parser.add_argument('--noise_vocab_size', type=int, default=0)
    parser.add_argument('--num_tokens_to_copy', type=int, default=0)
    parser.add_argument('--k_motif_size', type=int, default=1)
    parser.add_argument('--v_motif_size', type=int, default=1)
    parser.add_argument('--multi_query', type=bool, default=True)
    parser.add_argument('--num_train_examples', type=int, default=12_800)
    parser.add_argument('--num_test_examples', type=int, default=1_280)
    parser.add_argument('--data_path', type=str, default='./data')
    parser.add_argument('--optimizer', type=str, default='adamw')
    parser.add_argument('--scheduler', type=str, default='cosine')
    parser.add_argument('--min_lr', type=float, default=1e-6)
    parser.add_argument('--early_stop', type=bool, default=False)
    parser.add_argument('--precision', type=str, default='bf16')
    parser.add_argument('--seed', type=int, default=12345)
    parser.add_argument('--target_ignore_index', type=int, default=-100)

    args = parser.parse_args()
    args.num_labels = 2
    args.hidden_dropout_prob = 0.1
    args.option = None
    # args.use_gpu = torch.cuda.is_available() 
    
    return args


if __name__ == "__main__":
    os.environ['PJRT_DEVICE'] = 'CUDA' # For GCP
    args = parse_args() 
    print("args:", args)
    
    results = None
    if args.task == "seqclass":
        if args.model == "transformers":
            train_gpt_neo_seqclass_pretrained(args)
        elif args.model == "mamba":
            train_mamba_seqclass_pretrained(args)
        elif args.model == "hybrid":
            train_hybrid_seqclass_pretrained(args)
    elif args.task == "initd_seqclass":
        if args.model == "transformers":
            train_gpt_neo_seqclass_initd(args)
        elif args.model == "mamba":
            train_mamba_seqclass_initd(args)
    elif args.task == "mad":
        results = train_mad(args.model)
    elif args.task == "initd_imdb":
        train_imdb(args.model)
    elif args.task == "eval_squad":
        eval_squad_pretrained(args, args.model)
    elif args.task == "tune_squad":
        train_squad_pretrained(args.model)
    
    if args.run_hybrid_pretrained:
        train_imdb_pretrained("hybrid")

    if results and args.output_file:
        with open(args.output_file, 'a') as f:
            f.write(str(args) + '\n' + str(results) + '\n')
