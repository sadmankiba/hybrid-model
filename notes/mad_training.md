# MAD Training Settings 

## Manipulating Task Difficulty 

For all variants of in-context recall, we evaluate input sequence lengths of 128, 256, 512, and 1024 tokens, training dataset sizes with 12800, 6400, 3200, 1600 and 800 samples, and vocabulary sizes (equally divided into keys and values), of 16, 32, 64, and 128 tokens.

For noisy in-context recall, we additionally evaluate shares of 20%, 40%, 60%, and 80% noise tokens in the inputs.

For the selective copying task, we evaluate sequence lengths of 256, 512, and 1024 tokens, training dataset sizes with 12800, 6400, 3200, 1600 and 800 samples, vocabulary sizes of 16, 32, 64, and 128 tokens, and 16, 32, 64, and 96 tokens of a the input that are to be copied.

For the compression task, we evaluate input sequence lengths of 32, 64, 128 and 256 tokens, vocabulary sizes of 16, 32, 64, and 128 tokens, and training dataset sizes of 12800, 6400, 3200, 1600 and 800 samples.

For the memorization task, we evaluate vocabulary sizes of 256, 512, 1024, 2048, 4096, and 8192 tokens, while keeping the training dataset fixed at 256 samples with an input length of 32 (thereby effectively varying the rate at which each fact appears in the training data, with average rates of 32, 16, 8, 4, 2, and
1).


## Architecture

Each architecture is 2 blocks with a total of 4 layers. A block combines a sequence mixing layer with a subsequent channel mixing layer. Exception is Mamba layers, which combine sequence and
channel mixing into a single layer. All layers have dim 128. 

**Channel Mixing** 
* SwiGLU MLP: Inner dim 512. 
* MoE MLP: Number of experts 8, expert width 16

**Sequence Mixing**
* Hyena
* Mamba: state dim = 4, conv dim = 4, width expansion = 2, no bias
* Multi-head gated Linear Attention
* Multi-head Attention: #heads = 16, head dim = 8
* Multi-head Hyena
* Hyena experts

## Training
Optimizer AdamW
Optimizer momentum β1, β2 = 0.9, 0.98
Dropout None
Batch size 128
Training epochs 200
Learning rate schedule cosine decay
Number of layers 4
Number of evaluation samples 1,280
Base learning rate [0.0001, 0.0005, 0.001]
Weight decay [0.0, 0.1]