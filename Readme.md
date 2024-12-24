# Hybrid Model 

This repository contains reimplementation of [Manticore Hybrid](https://arxiv.org/abs/2406.00894) architecture. Manticore automates the search of parallel-block hybrid architecture with language models such as Transformer and Mamba. 

## Run

```sh
# Example MAD task
python3 main.py --run_mad_trans  --task 'in-context-recall' --vocab_size 16  --seq_len 32 \
    --num_train_examples 4096 --num_test_examples 256 \
    --num_layers 2 --hidden_size 64 --num_heads 4 --epochs 20 --batch_size 32 \
    --log_interval 20 --lr 5e-4

# Example train SQUAD
python3 main.py --task tune_squad --model mamba --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 2 --train_size 2000 --eval_size 500
``` 

## Methods

* `Hybrid-Architecture`: Blocks from HuggingFace `GPT-Neo` and `Mamba` models are combined with custom built projectors. 
* `Pretraining Projectors`: Projectors in pre-trained hybrids are initialized with random weights and trained on ELI5 dataset. 

# Results

Evaluation performed on MAD tasks, IMDB sentiment analysis and SQUAD question-answering. 

See [presentation](./Presentation.pdf) and [report](./Report.pdf). 

## Contributors

* [Sadman Sakib](https://github.com/sadmankiba)
* [Riyad Hassen](https://github.com/RiyadHassen)
* [Fahad Touseef](https://github.com/Fahad-Touseef)