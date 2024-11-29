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