# staged-training

In our paper [**Staged Training for Transformer Language Models**](https://arxiv.org/abs/2203.06211), we propose a staged training setup that begins with a small model and incrementally increases the amount of compute used for training by applying a "growth operator" to increase the model depth and width. By initializing each stage with the output of the previous one, the training process effectively re-uses the compute from prior stages and becomes more efficient. 

We release the code for computing the optimal schedule, applying the growth operator, and evaluating the checkpoints.

## Setup

The scripts in this repository require Python 3.7 or newer.
Once you have a suitable Python environment, first install PyTorch v1.9.0 according the [official instructions](https://pytorch.org/get-started/previous-versions/#v190). Then run
```
pip install -r requirements.txt
```

Alternatively, you can run any of the scripts in this repository using Docker:

```bash
docker build -t staged-training:latest .
docker run --rm --gpus all staged-training:latest NAME_OF_SCRIPT
```

## Usage

### Computing the Optimal Schedule

TODO

### Direct pretraining on 4 GPUs

Training a GPT2-large size model:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix gpt2_large \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.002006911598778545 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 4 --size GPT2_large --random 
```

Training a model that when doubled width-wise will be the size of GPT2-large:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix gpt2_large_div2_width \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.002090898967568796 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 2 --size GPT2_large_div2_width --random
```

Training a model that when doubled depth-wise will be the size of GPT2-large:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix gpt2_large_div2_depth \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.0020489052831736704 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 2 --size GPT2_large_div2_depth --random
```

###  Using the operator to double the size

#### Width-wise operator

First apply the operator to the ckeckpoint:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_width \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 --batch_size 4 --grad_accum 32 \
  --lr 0.002006911598778545 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 --log_rate 10 --num_workers 4 \
  --size GPT2_large_div2_width --random \
  --resume gpt2_large_div2_width/checkpoint-epoch=0-step=6249.ckpt \
  --doubling weights --restart_warmup_steps 200 --restart_steps 3319 \
  --reset_lr_scheduler
```

Then resume the grown checkpoint and set the restart step and re-warmup steps:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_depthx2 \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.002006911598778545 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 4 --size GPT2_large --random \
  --resume final_gpt2_large_div2_width/checkpoint-epoch=0-step=6499.ckpt.doubled_weights \
  --restart_warmup_steps 150 --restart_steps 3319 --reset_lr_scheduler 
```

#### Depth-wise operator

First apply the operator to the checkpoint:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_depth \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.002006911598778545 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 4 --size GPT2_large_div2_depth --random \
  --resume gpt2_large_div2_depth/checkpoint-epoch=0-step=6499.ckpt \
  --doubling layers --restart_warmup_steps 150 --restart_steps 4449 \
  --reset_lr_scheduler --doubling_layers alternate_id
```

Then resume the grown checkpoint and set the restart step and re-warmup steps:

```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_depthx2 \
  --gpu_count -1 --model gpt2 --tokenizer gpt2 \
  --batch_size 4 --grad_accum 32 --lr 0.002006911598778545 --warmup_steps 3000 \
  --train_steps 250000 --val_every 50 --val_batches 50 --fp16 --seqlen 1024 \
  --log_rate 10 --num_workers 4 --size GPT2_large --random \
  --resume final_gpt2_large_div2_depth/checkpoint-epoch=0-step=6499.ckpt.doubled_layer \
  --restart_warmup_steps 150 --restart_steps 4449 --reset_lr_scheduler 
```
### Evaluation

Use [`eval_wikitext.py`](./eval_wikitext.py) or [`eval_lambada.py`](./eval_lambada.py) to evaluate [GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) on one of the supported datasets. For example:

```bash
python eval_wikitext.py
```

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

## Reference

If you use staged training in your research or wish to refer to the baseline results published here, 
please use the following BibTeX entry. 
```
@misc{shen2022staged,
    title={Staged Training for Transformer Language Models},
    author={Sheng Shen and Pete Walsh and Kurt Keutzer and Jesse Dodge and Matthew Peters and Iz Beltagy},
    year={2022},
    eprint={2203.06211},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
