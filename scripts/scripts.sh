# Mamba train 
python3 main.py --use_gpu --epochs 1 --log_interval 50 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200

python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 20000 --eval_size 1000 --run_trans
python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200 --run_mamba
python3 main.py --use_gpu --epochs 1 --log_interval 200 --lr 5e-5 --batch_size 1 --train_size 5000 --eval_size 200 --run_hybrid