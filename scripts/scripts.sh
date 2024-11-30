# Mamba train 
python3 main.py --use_gpu --epochs 1 --log_interval 50 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200

# Pretrained frozen with Classification Head
python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 20000 --eval_size 1000 --run_trans
python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200 --run_mamba
python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200 --run_hybrid

# Initialized models
python3 main.py --run_gpt_neo_initd --num_layers 4 --hidden_size 128 --num_heads 4 \
    --epochs 10 --log_interval 200 --lr 5e-5 --batch_size 8 \
    --train_size 20000 --eval_size 1000 --use_gpu 
python3 main.py --run_mamba_initd --num_layers 4 --hidden_size 128 \
    --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 4 \
    --train_size 2000 --eval_size 200 --use_gpu 

# MAD Tasks
python3 main.py --run_mad  --task 'selective-copying' --vocab_size 16  --seq_len 20 \
    --num_tokens_to_copy 5 --num_train_examples 4000 --num_test_examples 200 \
    --num_layers 2 --hidden_size 32 --num_heads 2 --epochs 10 --batch_size 8 \
    --log_interval 100 --lr 5e-4 --use_gpu