

# Run Transformer, Mamba, Hybrid and MambaFormer model on all six MAD tasks

# To run, in project-root/,
# $ source scripts/mad-bench.sh
# $ run_mad_trans

## MAD Tasks
VOCAB_SIZE=64
SEQ_LEN=32
NUM_TRAIN=4096
NUM_TEST=256
NUM_LAYERS=2
HIDDEN_SIZE=64
NUM_HEADS=4
NUM_TOKENS_TO_COPY=8
K_MOTIF_SIZE=3
V_MOTIF_SIZE=3
FRAC_NOISE=0
NOISE_VOCAB_SIZE=0
NOISY_FRAC_NOISE=0.4
NOISY_NOISE_VOCAB_SIZE=16
MULTI_QUERY=False

## Training
EPOCHS=20
BATCH_SIZE=32
LOG_INTERVAL=20
LR=5e-4

tasks=('selective-copying' 'in-context-recall' 'noisy-in-context-recall' 'fuzzy-in-context-recall' 'memorization' 'compression')
# tasks=('fuzzy-in-context-recall' 'memorization' 'compression')

common_args="--vocab_size $VOCAB_SIZE \
             --seq_len $SEQ_LEN \
             --num_train_examples $NUM_TRAIN \
             --num_test_examples $NUM_TEST \
             --num_layers $NUM_LAYERS \
             --hidden_size $HIDDEN_SIZE \
             --num_heads $NUM_HEADS \
             --num_tokens_to_copy $NUM_TOKENS_TO_COPY \
             --k_motif_size $K_MOTIF_SIZE \
             --v_motif_size $V_MOTIF_SIZE \
             --multi_query $MULTI_QUERY \
             --epochs $EPOCHS \
             --batch_size $BATCH_SIZE \
             --log_interval $LOG_INTERVAL \
             --lr $LR"


non_noisy_args="$common_args --frac_noise $FRAC_NOISE  --noise_vocab_size $NOISE_VOCAB_SIZE"
noisy_args="$common_args --frac_noise $NOISY_FRAC_NOISE  --noise_vocab_size $NOISY_NOISE_VOCAB_SIZE"

run_mad_trans() {
    for task in ${tasks[@]}; do
        if [ "$task" == "noisy-in-context-recall" ]; then
            args=$noisy_args
        else
            args=$non_noisy_args
        fi
        python3 main.py --task mad --model transformers --mad_task $task $args --output_file 'trans.txt' 
    done
}


run_mad_mamba() {
    for task in ${tasks[@]}; do
        if [ "$task" == "noisy-in-context-recall" ]; then
            args=$noisy_args
        else
            args=$non_noisy_args
        fi
        python3 main.py --run_mad_mamba --task $task $args --output_file 'mamba.txt'
    done
}


run_mad_hybrid() {
    proj_types=('gres' 'gressf')
    num_hybrid_blocks=(1 2)
    for num_hybrid_block in ${num_hybrid_blocks[@]}; do
        for proj_type in ${proj_types[@]}; do    
            for task in ${tasks[@]}; do
                if [ "$task" == "noisy-in-context-recall" ]; then
                    args=$noisy_args
                else
                    args=$non_noisy_args
                fi
                hybrid_args="$args --num_trans_layers $NUM_LAYERS --num_mamba_layers $NUM_LAYERS \
                                   --num_hybrid_blocks $num_hybrid_block --proj_type $proj_type"
                python3 main.py --run_mad_hybrid  --task $task $hybrid_args --output_file 'hybrid.txt'
            done
        done
    done
}


run_mad_mamform() {
    for task in ${tasks[@]}; do
        if [ "$task" == "noisy-in-context-recall" ]; then
            args=$noisy_args
        else
            args=$non_noisy_args
        fi
        mamform_args="$args --num_trans_layers $NUM_LAYERS --num_mamba_layers $(($NUM_LAYERS + 1))"
        python3 main.py --run_mad_mamform  --task $task $mamform_args --output_file 'mamform.txt'
    done
}


