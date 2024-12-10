
tune_mamba_and_trans_slow() {
    python3 main.py --task tune_squad --model mamba --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 2 --train_size 2000 --eval_size 250
    python3 main.py --task tune_squad --model transformers --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 4 --train_size 10000 --eval_size 500
}

# for Mamba fast path
tune_mamba_and_trans_fast() {
    python3 main.py --task tune_squad --model mamba --epochs 1 --log_interval 500 --lr 5e-5 --batch_size 4 --train_size 20000 --eval_size 1000
    python3 main.py --task tune_squad --model transformers --epochs 1 --log_interval 500 --lr 5e-5 --batch_size 4 --train_size 20000 --eval_size 500
}

tune_mamba_and_trans_mini() {
    python3 main.py --task tune_squad --model mamba --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 2 --train_size 20 --eval_size 4
    python3 main.py --task tune_squad --model transformers --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 4 --train_size 40 --eval_size 8
}