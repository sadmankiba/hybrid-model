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

```sh
python3 main.py --run_mad  --task 'selective-copying' --vocab_size 16  --seq_len 20     --num_tokens_to_copy 5 --num_train_examples 4000 --num_test_examples 200     --num_layers 2 --hidden_size 32 --num_heads 2 --epochs 50 --batch_size 8     --log_interval 100 --lr 5e-4 --use_gpu
```

epoch: 5 train loss: 0.127, eval loss: 0.159, eval acc: 0.950
