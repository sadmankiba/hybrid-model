


tune_mamba_and_trans() {
    python3 main.py --task tune_squad --model mamba --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 2 --train_size 2000 --eval_size 250
    python3 main.py --task tune_squad --model transformers --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 4 --train_size 10000 --eval_size 500
}