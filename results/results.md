# Results


## Memory Usage

Mamba 130m backbone-freezed fine-tuning
* IMDB batch size 1 : 2200 MB (without amp)
* IMDB batch size 1 : 1400 MB (with amp)

Hybrid GPT-Neo + Mamba 130m backbone-freezed fine-tuning
* IMDB batch size 1: 9800 MB

## IMDB Classification 

**GPT-Neo Pretrained Frozen**
Epoch 1. Train size 20000. Batch size 1. LR 5e-5
dev acc: 0.85

batch size 1, with log-softmax on logits
Train loss was stably above 0 (around 0.3, 0.2), decreased gradually.
Epoch 0, Step 2000: dev acc: 0.75
Epoch 0, Step 3000: dev acc: 0.79
Epoch 0, Step 4000: dev acc: 0.80

batch size 1, without log-softmax on logits
Train loss decreased abruptly (to -50, -80 etc.)
Epoch 0, Step 2000: dev acc: 0.62
Epoch 0, Step 3000: dev acc: 0.68
Epoch 0, Step 4000: dev acc: 0.58

**Mamba Pretrained Frozen**    
Epoch = 1, lr = 5e-5, batch_size = 1, train_size 5000 eval_size 200.
dev acc: 0.8

**Hybrid**  


## MAD Tasks 

Epoch is 1-indexed.

### Selective Copying

```sh
--vocab_size 16  --seq_len 20     --num_tokens_to_copy 5 --num_train_examples 4000 --num_test_examples 200     --num_layers 2 --hidden_size 32 --num_heads 2 --epochs 50 --batch_size 8  --lr 5e-4 
```

| Model | Parameters |  Epoch  | Train loss | Eval loss | Eval acc |
| Transformer| layers 2 | 6   | 0.127      | 0.159     |  0.950   |
| Mamba      | layers 2 | 3   | 0.025      | 0.004     |  1.000   |
| Hybrid     | blocks 1, Null tLrs 2 mLrs 2 | 2 | 0.061 | 0.056 | 0.989 |  
| Hybrid     | blocks 2, Null tLrs 2 mLrs 2 | 2 | 0.098 | 0.055 | 0.984 |
| Hybrid     | blocks 1, Res tLrs 2 mLrs 2 | 3  | 0.287 | 0.059 | 0.984 |
| Hybrid     | blocks 2, Res tLrs 2 mLrs 2 | 3  | 0.131 | 0.066 | 0.980 |
| Hybrid     | blocks 1, Gated-res tLrs 2 mLrs 2 | 2 | 0.080 | 0.038 | 0.986 |
| Hybrid     | blocks 2, Gated-res tLrs 2 mLrs 2 | 3 | 0.148 | 0.071 | 0.979 |
| Hybrid     | blocks 1, Gated-res tLrs 1 mLrs 1 | 2 | 0.076 | 0.091 | 0.980 |
| Hybrid     | blocks 1, Gated-res-soft tLrs 2 mLrs 2 | 2 | 0.044 | 0.023 | 0.993 | 
| Hybrid     | blocks 2, Gated-res-soft tLrs 2 mLrs 2 | 5 | 0.101 | 0.076 | 0.967 | 
| Hybrid     | blocks 1, Gated-res-soft tLrs 1 mLrs 1 | 10 | 0.742 | 0.765 | 0.732 |
| MambaFormer| tLrs 2 mLrs 3 | 6 | 0.121 | 0.062 | 0.987 |



### In-context Recall 

```sh
--vocab_size 16 --num_train_examples 4096 --num_test_examples 256 --hidden_size 128 --num_heads 16 --epochs 20 --batch_size 32 --log_interval 20 --lr 5e-4
```

| Model      | Parameters       | Epoch  | Train loss | Eval loss | Eval acc |
| Transformer| layers 2, seq 64 | 20     | 1.510      | 0.589     | 0.757    |
| Mamba      | layers 4, seq 64 | 2      | 1.337      | 0.002     | 1.000    |
| Hybrid     | blocks 1, Gated-res tLrs 2 mLrs 2, seq 16 | 20 | 1.539 | 0.126 | 0.968 |


```sh
--vocab_size 16  --seq_len 32 --num_train_examples 4096 --num_test_examples 256 \
    --num_layers 2 --hidden_size 64 --num_heads 4 --epochs 20 --batch_size 32 \
    --log_interval 20 --lr 5e-4
```

| Model      | Parameters       | Epoch  | Train loss | Eval loss | Eval acc |
| Transformer| layers 2 | 20   | 1.96       | 1.58      |  0.34    | 
| Mamba      | layers 2 | 2 | 1.67  | 0.20   | 1.00|
| Hybrid     | blocks 1, Gated-res tLrs 2 mLrs 2 | 3 | 1.566 | 0.005 | 1.000 |
| MambaFormer| tLrs 2 mLrs 3 | 20  | 1.84 | 1.19 | 0.516 | 


## Benchmarking

## Configs

**MAD Tasks**
VOCAB_SIZE=16
SEQ_LEN=32
NUM_TRAIN=4096
NUM_TEST=256
NUM_LAYERS=2
HIDDEN_SIZE=64
NUM_HEADS=4
NUM_TOKENS_TO_COPY=6
K_MOTIF_SIZE=3
V_MOTIF_SIZE=3
FRAC_NOISE=0
NOISE_VOCAB_SIZE=0
NOISY_FRAC_NOISE=0.2
NOISY_NOISE_VOCAB_SIZE=4
MULTI_QUERY=False

**Training**
EPOCHS=20
BATCH_SIZE=32
LOG_INTERVAL=20
LR=5e-4

(Loss, iter) and (acc, iter)

| Task     | Transformer | Mamba | Hybrid 1-gres | Hybrid 1-gressf | Hybrid 2-gres | Hybrid 2-gressf | MambaFormer |
| Sel-Copy | (0.243, 17), (0.91, 16) | (0.0004, 14), (0.999, 2) | (0.005, 13), (0.994, 8) | (0.008, 17), (0.996, 16) | () | () | (0.05, 14), (0.98, 12)
| ICR      | (1.55, 18), (0.35, 12) | (0.004, 13), (0.999, 5) | (0.002, 8), (1.0, 2) | (0.003, 3), (1, 2) | () | () | (0.02, 17), (0.99, 17) 
| Noisy ICR| (1.31, 10), (0.42, 10) | (0.010, 3), (1, 3) | (0.002, 2), (1.0, 1) | (0.001, 3), (1, 3) | () | () | (0.001, 13), (1.0, 12)
| Fuzzy ICR| (1.49, 13), (0.35, 12) | () | (1.68, 4), (0.42, 6) | (1.64, 4), (0.42, 8) | () | () | (2.13, 12), (0.15, 0)
| Mem      | (0.0003, 12), (1, 0) | () | (0, 0), (1, 0) | (0, 1), (1, 0) | () | () | (0.0, 19), (0.996, 17)
| Comp     | (0.0005, 12), (1, 0) | () | (0, 0), (1, 0) | (0, 0), (1, 0) | () | () | (0.0, 2), (1.0, 1)


**MAD Tasks**
VOCAB_SIZE=64
SEQ_LEN=32

Trans does not learn in-context

| Task     | Transformer | Mamba | Hybrid 2-gressf | Mambaformer | 
| Sel-copy | (0.84, 11), (0.73, 3) |
| ICR      | (3.4, 6), (0.03, 1) |
| 