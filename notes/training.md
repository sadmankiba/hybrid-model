# Training

IMDB sequences have token_id length varies between 100 to 500. Truncating text to 

For Hybrid pretrained model, GPU memory usage reaches upto 12000 MB. 

Use Wandb. 

## SQUAD 

**Pretrained Mamba**

Slow path 
* Evaluaton - 30s/step, 8GB for batch size 2  
* Training - 30s/step, 8GB for batch size 2

**Pretrained GPT-Neo**

* Evaluation - 6s/step 3.2 GB for batch size 4
* Training - 2s/step, 3.2 GB for batch size 4