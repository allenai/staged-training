# staged-training

In our paper [**Staged Training for Transformer Language Models**](https://arxiv.org/abs/2203.06211), we propose a staged training setup that begins with a small model and incrementally increases the amount of compute used for training by applying a "growth operator" to increase the model depth and width. By initializing each stage with the output of the previous one, the training process effectively re-uses the compute from prior stages and becomes more efficient. 

We release the reproducible code for the growth operator and evaluation scripts here.

## Structure
```
.
├── requirements.txt        # Python requirements needed to run our scripts
├── dev-requirements.txt    # Python requirements needed for development only
├── scripts/
│  ├── cheatsheet.txt       # list of commands for using the width/depth growth operators
│  ├── gpt_pretrain.py      # pretraining script that implements the width/depth growth operators
│  └── remove_oracle.ipynb  # used to estimate the contants to determine the growth points
└── evaluation/
   ├── tools/               # common code
   ├── Dockerfile           # for building a Docker image to run evaluation scripts
   ├── eval_lambada.py      # evaluate a checkpoint on LAMBADA
   ├── eval_wikitext.py     # evaluate a checkpoint on WikiText
   └── requirements.txt     # additional requirements for evaluation scripts
```

## Growth Operator

Our growth operators (width/depth) each take as input the entire training state (including model parameters, optimizer state, learning rate schedule, etc.) and output a new training state from which training continues.

Please see the `scripts/cheatsheet.txt` for more examples on how to use the corresponding scripts. 

For example, you can apply the width operator with:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_width_check_bs512_lr0.0020_warmup3k_seqlen1024_debug \
  --gpu_count -1 \
  --model gpt2  \
  --tokenizer gpt2 \
  --batch_size 4 \
  --grad_accum 32  \
  --lr 0.002006911598778545  \
  --warmup_steps 3000 \  \
  --train_steps 250000  \
  --val_every 50  \
  --val_batches 50 \
  --fp16 \
  --seqlen 1024 \
  --log_rate 10 \
  --num_workers 4 \
  --size GPT2_large_div2_width \
  --random \
  --resume final_runs/final_gpt2_large_div2_width_check_bs512_lr0.0021_warmup3k_seqlen1024_debug/checkpoint-xxx.ckpt \
  --doubling weights
```

Or the depth operator with:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python scripts/gpt_pretrain.py \
  --save_prefix final_gpt2_large_div2_depthx2_check_bs512_lr0.0020_warmup3k_seqlen1024_debug \
  --gpu_count -1 \
  --model gpt2  \
  --tokenizer gpt2 \
  --batch_size 4 \
  --grad_accum 32 \
  --lr 0.002006911598778545 \
  --warmup_steps 3000 \
  --train_steps 250000 \
  --val_every 50 \
  --val_batches 50 \
  --fp16 \
  --seqlen 1024 \
  --log_rate 10 \
  --num_workers 4 \
  --size GPT2_large_div2_depth \
  --random \
  --resume final_runs/final_gpt2_large_div2_depth_check_bs512_lr0.0020_warmup3k_seqlen1024_debug/checkpoint-epoch=0-step=6499.ckpt \
  --doubling layers
``` 

## Reference

If you use staged training in your research or wish to refer to the baseline results published here, 
please use the following BibTeX entry. 
```shell
@misc{shen2022staged,
    title={Staged Training for Transformer Language Models},
    author={Sheng Shen and Pete Walsh and Kurt Keutzer and Jesse Dodge and Matthew Peters and Iz Beltagy},
    year={2022},
    eprint={2203.06211},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
```
