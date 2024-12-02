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
| Hybrid     | blocks 1, Gated-res tLrs 2 mLrs 2 | 3 | 0.046 | 0.010 | 0.998 | 


### In-context Recall 

```sh
--vocab_size 16  --seq_len 64 --num_train_examples 4096 --num_test_examples 256 --hidden_size 128 --num_heads 16 --epochs 20 --batch_size 32 --log_interval 20 --lr 5e-4
```

Transformers: num_layers 2, seq_len 64
Mamba : num_layers 4, seq_len 64
Hybrid: num_blocks 1, num_trans_layers 2, num_mamba_layers 2, seq_len 16

Transformers: epoch: 19 train loss: 1.510, eval loss: 0.589, eval acc: 0.757
Mamba: epoch: 1 train loss: 1.337, eval loss: 0.002, eval acc: 1.000
Hybrid: epoch: 19 train loss: 1.539, eval loss: 0.126, eval acc: 0.968