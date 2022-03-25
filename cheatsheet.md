## Constants and hyperparameters

### Scaling laws and growth operators

(Thresholds are mostly the same, constants are a bit different)

| name | value |
| ---- | ----- |
| `threshold_optimality` | `-0.052` |
| `threshold_depth_growth` | `-0.0575` |
| `threshold_width_growth` | `-0.0475` |
| `threshold_depth_width_growth_ours`\* | `-0.03` | 
| `constant_op_width` | `1.776` |
| `constant_op_depth` | `1.412` |
| `constant_op_depth_width` | `2.455` |

> \* slightly different from the one you get from the scaling laws

### Learning rate

We set the initial learning rate based on the heuristic from [Kaplan et. al (2020)](https://api.semanticscholar.org/CorpusID:210861095) (Appendix D.6):

```
LR(N) ≈ 0.003239 + −0.0001395 * log(N)
```

| model | value |
| ----- | ----- |
| `GPT2_base` | `0.002132892651963921` |
| `GPT2_base_div2_depth` | `0.0021748863363590465` |
| `GPT2_base_div2_width` | `0.002216880020754172` |
| `GPT2_large` | `0.002006911598778545` |
| `GPT2_large_div2_width` | `0.002090898967568796` |
| `GPT2_large_div2_depth` | `0.0020489052831736704` |
| `GPT2_large_div4_depth` | `0.002090898967568796` |
| `GPT2_large_div4_width` | `0.0021748863363590465` |


## Direct pretraining on 4 GPUs

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.002006911598778545  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 4 --size GPT2_large --random 
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_div2_depth_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.0020489052831736704  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 2 --size GPT2_large_div2_depth --random 
```

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_div2_width_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.002090898967568796  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 2 --size GPT2_large_div2_width --random 
```

##  Using the operator to double the weights (width-wise operator)

First apply the operator to the ckeckpoint:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py     \
  --save_prefix final_gpt2_large_div2_width_check_bs512_lr0.0020_warmup3k_seqlen1024_debug     \
  --gpu_count -1   --model gpt2  --tokenizer gpt2     --batch_size 4 --grad_accum 32  --lr 0.002006911598778545  --warmup_steps 3000    \  --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024      --log_rate 10 --num_workers 4 \
  --size GPT2_large_div2_width --random   \
  --resume final_runs/final_gpt2_large_div2_width_check_bs512_lr0.0021_warmup3k_seqlen1024_debug/checkpoint-epoch=0-step=6249.ckpt   \
  --doubling weights --restart_warmup_steps 200  --restart_steps 3319 \
  --reset_lr_scheduler  
```

Then resume the grown checkpoint and set the restart step and re-warmup steps:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_div2_depthx2_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.002006911598778545  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 4 --size GPT2_large --random \
    --resume final_runs/final_gpt2_large_div2_width_check_bs512_lr0.0020_warmup3k_seqlen1024_debug/checkpoint-epoch=0-step=6499.ckpt.doubled_weights \
    --restart_warmup_steps 150 --restart_steps 3319 --reset_lr_scheduler 
```

## Using the operator to double the layers (depth-wise operator)

First apply the operator to the checkpoint:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_div2_depthx2_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.002006911598778545  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 4 --size GPT2_large_div2_depth --random \
    --resume final_runs/final_gpt2_large_div2_depth_check_bs512_lr0.0020_warmup3k_seqlen1024_debug/checkpoint-epoch=0-step=6499.ckpt \
    --doubling layers  --restart_warmup_steps 150 --restart_steps 4449 --reset_lr_scheduler  --doubling_layers alternate_id
```

Then resume the grown checkpoint and set the restart step and re-warmup steps:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py    \
    --save_prefix final_gpt2_large_div2_depthx2_check_bs512_lr0.0020_warmup3k_seqlen1024_debug   \
    --gpu_count -1   --model gpt2  --tokenizer gpt2 \
    --batch_size 4 --grad_accum 32  --lr 0.002006911598778545  --warmup_steps 3000  \
    --train_steps 250000  --val_every 50  --val_batches 50   --fp16   --seqlen 1024  \
    --log_rate 10 --num_workers 4 --size GPT2_large --random \
    --resume final_runs/final_gpt2_large_div2_depth_check_bs512_lr0.0020_warmup3k_seqlen1024_debug/checkpoint-epoch=0-step=6499.ckpt.doubled_layer \
    --restart_warmup_steps 150 --restart_steps 4449 --reset_lr_scheduler 
```
