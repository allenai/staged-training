import argparse
import copy
import glob
import logging
import math
import multiprocessing
import os
import random
import re
import time

import numpy as np
import pytorch_lightning as ptl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers.test_tube import TestTubeLogger
from pytorch_lightning.trainer.connectors.checkpoint_connector import \
    CheckpointConnector
from pytorch_lightning.trainer.training_loop import TrainLoop
from pytorch_lightning.utilities import AMPType, rank_zero_warn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.upgrade_checkpoint import \
    KEYS_MAPPING as DEPRECATED_CHECKPOINT_KEYS
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import (AutoConfig, AutoModelForCausalLM,
                          AutoModelForMaskedLM, AutoTokenizer, BertForMaskedLM,
                          DataCollatorForLanguageModeling, GPT2LMHeadModel,
                          RobertaForMaskedLM)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

try:
    from apex import amp
except ImportError:
    amp = None

import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    import torch_xla.core.xla_model as xm
except ImportError:
    XLA_AVAILABLE = False
else:
    XLA_AVAILABLE = True

from tools.growth_operator import double_param, double_state_dict


# =======restart the linear warmup strategy with linear warmup==========
def get_restart_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps,
    num_training_steps,
    last_epoch=-1,
    restart_warmup_steps=0,
    restart_steps=0,
):
    """
    Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after
    a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer.

    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
            The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
            The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
            The total number of training steps.
        last_epoch (:obj:`int`, `optional`, defaults to -1):
            The index of the last epoch when resuming training, will be modified.
        restart_warmup_steps:
            the restart_warmup_steps should be set last_epoch + restart_warmup_steps;

    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """

    def lr_lambda(current_step: int):

        if (
            restart_steps != 0
            and restart_warmup_steps != 0
            and current_step < restart_steps + restart_warmup_steps
            and current_step >= restart_steps
        ):
            assert current_step >= restart_steps

            # pre-warmup + restart-warmup
            if current_step < num_warmup_steps:
                return (
                    float(current_step - restart_steps)
                    / float(max(1, restart_warmup_steps))
                    * float(restart_steps + restart_warmup_steps)
                    / float(max(1, num_warmup_steps))
                )
            else:
                return (
                    float(current_step - restart_steps)
                    / float(max(1, restart_warmup_steps))
                    * float(num_training_steps - restart_steps - restart_warmup_steps)
                    / float(max(1, num_training_steps - num_warmup_steps))
                )

        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0,
            float(num_training_steps - current_step)
            / float(max(1, num_training_steps - num_warmup_steps)),
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


# the dataset object we are using
class MMapTextDataset(Dataset):
    def __init__(self, mmap_filename, chunk_size, bos_token_id, eos_token_id):
        # `chunk_size - 2` to reserve space for <s> and </s>
        self.num_instances = np.memmap(mmap_filename, mode="r", dtype=np.uint16).shape[
            0
        ] // (chunk_size - 2)
        # defer loading the token_ids memmap until after the first __getitem__ call.
        # when spawning new processes for ddp, there is a hard limit in python < 3.8 that
        # pickle files need to be < 4GB. By waiting until after the first __getitem__ we
        # don't have to pickle the memmap
        self.token_ids = None
        self._mmap_filename = mmap_filename
        self._chunk_size = chunk_size
        self._bos_token_id = bos_token_id
        self._eos_token_id = eos_token_id

    def __len__(self):
        return self.num_instances

    def __getitem__(self, i):
        if self.token_ids is None:
            self.token_ids = np.memmap(self._mmap_filename, mode="r", dtype=np.uint16)
        from_index = i * (self._chunk_size - 2)
        to_index = (i + 1) * (self._chunk_size - 2)
        data = np.concatenate(
            (
                [self._bos_token_id],
                self.token_ids[from_index:to_index],
                [self._eos_token_id],
            )
        )
        return torch.tensor(data, dtype=torch.long)

    # ========================= preprocessing code ========================= #
    @staticmethod
    def _process_file(full_fname):
        "Step 1: tokenize an input text file then save token ids into `np.memmap` shards of size `args.shard_size`"
        fname = full_fname.split("/")[-1]
        if args.data_type == "tfrecord":
            log_filename = f"{args.output_dir}/logs-{fname}.log"
        elif args.data_type == "raw_text":
            log_filename = f"{args.output_dir}/logs-{args.shard_size}/{fname}.log"
        if os.path.isfile(log_filename):
            logging.info(f"Skipping {full_fname} ...")
            return  # log file already exists. Skip current file.

        if args.num_workers > 1:
            current = multiprocessing.current_process()
            process_identity = int(current._identity[0])
        else:
            process_identity = 1

        if process_identity == 1:
            logging.info(f"Processing {full_fname} ...")

        def _write_shard():
            if len(token_list) == 0:
                return
            # if token_list[-1] != MMapTextDataset.tokenizer.sep_token_id:  # handle a rare case
            #     token_list.append(MMapTextDataset.tokenizer.sep_token_id)
            if args.data_type in ["tfrecord", "s2"]:
                shared_filename = f"{args.output_dir}/{fname}.bin"
            elif args.data_type == "raw_text":
                shared_filename = f"{args.output_dir}/shards-{args.shard_size}/{fname}-{shard_count}.bin"
            else:
                raise NotImplementedError
            logging.info(
                f"Writing {len(token_list)} tokens to shared {shared_filename}"
            )
            fp = np.memmap(
                shared_filename, dtype=np.uint16, mode="w+", shape=len(token_list)
            )
            fp[:] = token_list[:]
            del fp  # flush and close file

        token_list = []
        shard_count = 0
        tokens_count = 0

        if args.data_type == "raw_text":  # the input file is one doc per line
            with open(full_fname, "r") as fin:
                for line in tqdm(fin):
                    line = line.strip()
                    if line == "":  # drop empty lines
                        continue
                    tokens = MMapTextDataset.tokenizer.encode(
                        line, add_special_tokens=False
                    )  # `__getitem__` adds special tokens
                    token_list.extend(tokens)
                    if len(token_list) > args.shard_size:
                        _write_shard()
                        tokens_count += len(token_list)
                        token_list = []
                        shard_count += 1
                    else:
                        token_list.append(MMapTextDataset.tokenizer.sep_token_id)
                _write_shard()
                tokens_count += len(token_list)
        elif (
            args.data_type == "tfrecord"
        ):  # the input file is tfrecord format of the c4 dataset
            fin = tf.data.TFRecordDataset(full_fname)
            for raw_example in tqdm(iter(fin), disable=process_identity != 1):
                parsed = tf.train.Example.FromString(raw_example.numpy())
                feature_keys = set(parsed.features.feature.keys())
                if "text" in feature_keys:
                    line = (
                        parsed.features.feature["text"].bytes_list.value[0].decode()
                    )  # raw text
                    tokens = MMapTextDataset.tokenizer.encode(
                        line, add_special_tokens=False
                    )  # `__getitem__` adds special tokens
                    if args.add_sep_after_doc:
                        tokens.append(MMapTextDataset.tokenizer.sep_token_id)
                    token_list.extend(tokens)
                    tokens_count += len(token_list)
                shard_count += 1
            _write_shard()

        with open(log_filename, "w") as f:
            f.write(f"Generated {tokens_count} tokens in {shard_count + 1} shards")

    @staticmethod
    def _combine_shards(output_fname, shards_list):
        "Step 2: combining memmap shards into one `train.bin` or `val.bin` file"
        total_size = 0
        for filename in shards_list:
            total_size += np.memmap(filename, mode="r", dtype=np.uint16).shape[0]
        logging.info(f"Writing {total_size} tokens to {output_fname}")
        all_token_ids = np.empty(total_size, dtype=np.uint16)
        last_token_index = 0
        for filename in tqdm(shards_list):
            shared = np.memmap(filename, mode="r", dtype=np.uint16)
            all_token_ids[last_token_index : last_token_index + len(shared)] = shared[:]
            last_token_index += len(shared)
        fp = np.memmap(output_fname, dtype=np.uint16, mode="w+", shape=total_size)
        fp[:] = all_token_ids[:]
        del fp

    @staticmethod
    def raw_text_to_mmap(args):
        """This is the main preprocessing function. It processes all the text files in `args.input_dir` and
        outputs two np.memmap files, one for training and one for validation with ratio `args.train_dev_split`.
        Processing each input file involves tokenizing it, sharding it into shards of size `args.shard_size`,
        then writing each shard as an np.memmap file, shuffle the shards, split them into train and dev shards,
        then combine the shards of each set into one big file (train.bin and val.bin).
        Notice that only the shards are shuffled not the instances inside each shard. Therefor, it is important
        to use `args.shard_size` that's small enough to have a good train/dev split, but also not small enough
        to end up with a huge number of shards that might be difficult to work with.
        The stream of tokens in the memmap files represents documents separated with `tokenizer.sep_token`.
        In `__getitem__`, the `tokenizer.bos_token` and `tokenizer.eos_token`
        are added. The reason for not adding them at preprocessing time is to allow different sequence lengths
        later on. Notice that this is the "FULL-SENTENCES" setting in the RoBERTa paper, Table2.
        Example running the preprocessing:
            >>> python scripts/pretrain.py --input_dir dirWithTextFiles --train_dev_split 0.05  \
                                           --shard_size  268435456  --num_preprocessing_workers 16
        """
        MMapTextDataset.tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer, use_fast=True
        )
        assert (
            len(MMapTextDataset.tokenizer) < 65535
        )  # will use uint16 to store token ids
        all_files = glob.glob(f"{args.input_dir}/c4-*")
        print(len(all_files), MMapTextDataset.tokenizer)
        if os.path.exists(f"{args.output_dir}/cache/train.bin") and os.path.exists(
            f"{args.input_dir}/cache/val.bin"
        ):
            logger.info(
                "Cache already exists. Remove the cache directory to regenerate"
            )
            return
        try:
            os.mkdir(f"{args.output_dir}/cache/")
        except FileExistsError:
            pass
        try:
            os.mkdir(f"{args.output_dir}/shards-{args.shard_size}/")
        except FileExistsError:
            pass
        try:
            os.mkdir(
                f"{args.output_dir}/logs-{args.shard_size}/"
            )  # log progrss to be able to resume
        except FileExistsError:
            pass

        # STEP1: tokenizing and saving to shards
        if args.num_preprocessing_workers > 1:
            from multiprocessing.pool import Pool

            with Pool(args.num_preprocessing_workers) as p:
                list(
                    tqdm(
                        p.imap(MMapTextDataset._process_file, all_files),
                        total=len(all_files),
                    )
                )
        else:
            [MMapTextDataset._process_file(f) for f in tqdm(all_files)]

        if args.data_type == "raw_text":  # c4 tfrecords are already sharded
            # STEP2: shuffling shards and combining them into train.bin and val.bin files
            all_shards = glob.glob(f"{args.output_dir}/shards-{args.shard_size}/*.bin")
            random.shuffle(all_shards)  # shuffling based on shards not individual lines
            val_shards_count = int(args.train_dev_split * len(all_shards))
            val_shards = all_shards[:val_shards_count]
            train_shards = all_shards[val_shards_count:]
            # TODO: if MMapTextDataset._combining_shards is very slow for large files, it can be skipped but we nned to
            # update the dataset to read from multiple shards directly
            MMapTextDataset._combine_shards(
                f"{args.output_dir}/cache/val.bin", val_shards
            )
            MMapTextDataset._combine_shards(
                f"{args.output_dir}/cache/train.bin", train_shards
            )
        elif args.data_type == "tfrecord":
            train_shards = glob.glob(f"{args.output_dir}/*train*.bin")
            val_shards = glob.glob(f"{args.output_dir}/*val*.bin")
            MMapTextDataset._combine_shards(f"{args.output_dir}/val.bin", val_shards)
            MMapTextDataset._combine_shards(
                f"{args.output_dir}/train.bin", train_shards
            )
        del MMapTextDataset.tokenizer

    # ========================= end preprocessing code ========================= #


class MyCheckpointConnector(CheckpointConnector):
    def __init__(
        self,
        trainer,
        reset_optimizer=False,
        reset_lr_scheduler=False,
        set_global_step=None,
    ):
        super().__init__(trainer)
        self.reset_optimizer = reset_optimizer
        self.reset_lr_scheduler = reset_lr_scheduler
        self.set_global_step = set_global_step

    def restore_training_state(self, checkpoint, load_optimizer_states: bool = True):
        """
        COPIED from https://github.com/PyTorchLightning/pytorch-lightning/blob/1.0.8/pytorch_lightning/trainer/connectors/checkpoint_connector.py#L130-L199
        and updated to support reset_optimizer and reset_lr_scheduler
        """
        # validation
        if "optimizer_states" not in checkpoint or "lr_schedulers" not in checkpoint:
            raise KeyError(
                "Trying to restore training state but checkpoint contains only the model."
                " This is probably due to `ModelCheckpoint.save_weights_only` being set to `True`."
            )

        if any([key in checkpoint for key in DEPRECATED_CHECKPOINT_KEYS]):
            raise ValueError(
                "The checkpoint you're attempting to load follows an"
                " outdated schema. You can upgrade to the current schema by running"
                " `python -m pytorch_lightning.utilities.upgrade_checkpoint --file model.ckpt`"
                " where `model.ckpt` is your checkpoint file."
            )

        # restore amp scaling
        if (
            self.trainer.amp_backend == AMPType.NATIVE
            and "native_amp_scaling_state" in checkpoint
        ):
            self.trainer.scaler.load_state_dict(checkpoint["native_amp_scaling_state"])
        elif (
            self.trainer.amp_backend == AMPType.APEX
            and "amp_scaling_state" in checkpoint
        ):
            amp.load_state_dict(checkpoint["amp_scaling_state"])

        # restore callback states
        self.trainer.on_load_checkpoint(checkpoint)

        self.trainer.global_step = checkpoint["global_step"]
        if self.set_global_step is not None:
            self.trainer.global_step = self.set_global_step
        self.trainer.current_epoch = checkpoint["epoch"]

        # crash if max_epochs is lower then the current epoch from the checkpoint
        if self.trainer.current_epoch > self.trainer.max_epochs:
            m = f"""
            you restored a checkpoint with current_epoch={self.trainer.current_epoch}
            but the Trainer(max_epochs={self.trainer.max_epochs})
            """
            raise MisconfigurationException(m)

        # Division deals with global step stepping once per accumulated batch
        # Inequality deals with different global step for odd vs even num_training_batches
        n_accum = (
            1
            if self.trainer.accumulate_grad_batches is None
            else self.trainer.accumulate_grad_batches
        )
        expected_steps = self.trainer.num_training_batches / n_accum
        if (
            self.trainer.num_training_batches != 0
            and self.trainer.global_step % expected_steps > 1
        ):
            rank_zero_warn(
                "You're resuming from a checkpoint that ended mid-epoch. "
                "This can cause unreliable results if further training is done, "
                "consider using an end of epoch checkpoint. "
            )

        if not load_optimizer_states:
            return

        # restore the optimizers
        if not self.reset_optimizer:
            optimizer_states = checkpoint["optimizer_states"]
            for optimizer, opt_state in zip(self.trainer.optimizers, optimizer_states):
                print(opt_state.keys(), optimizer)
                # print(optimizer.param_groups.keys(), optimizer.param_groups)
                print([x.keys() for x in optimizer.param_groups])
                print([x.keys() for x in opt_state["param_groups"]])
                optimizer.load_state_dict(opt_state)

                # move optimizer to GPU 1 weight at a time
                # avoids OOM
                if self.trainer.root_gpu is not None:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.trainer.root_gpu)

        if not self.reset_lr_scheduler:
            # restore the lr schedulers
            lr_schedulers = checkpoint["lr_schedulers"]
            if self.set_global_step is not None:
                for lrs_state in lr_schedulers:
                    lrs_state["last_epoch"] = self.set_global_step
                    lrs_state["_step_count"] = self.set_global_step + 1

            for scheduler, lrs_state in zip(self.trainer.lr_schedulers, lr_schedulers):
                scheduler["scheduler"].load_state_dict(lrs_state)
        else:
            if self.set_global_step is not None:
                for scheduler in self.trainer.lr_schedulers:
                    scheduler["scheduler"].last_epoch = self.set_global_step
                    scheduler["scheduler"]._step_count = self.set_global_step + 1


# rewrite the MyTrainLoop from pytorch-lightning to support batch size and sequence length warmup
class MyTrainLoop(TrainLoop):
    def __init__(self, trainer, multiple_trainloader_mode, args):
        super().__init__(trainer, multiple_trainloader_mode)
        self.args = args

    def grad_norm(self, model, norm_type, should_accumulate=False):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        # parameters = [p for p in self.parameters() if p.grad is not None]
        # device = parameters[0].device
        # total_norm = torch.zeros([], device=device if parameters else None)
        # norm_type = float(norm_type)
        # for p in parameters:
        #     param_norm = p.grad.norm(norm_type)
        #     total_norm.add_(param_norm)
        norm_type = float(norm_type)

        norms, all_norms = {}, []
        # local_norm = torch.zeros([], device=model.device)
        for name, p in model.named_parameters():
            if p.grad is None:
                continue

            if not should_accumulate:
                # param_norm = float(p.grad.data.norm(norm_type))
                p_grad = p.grad.data / args.batch_size / args.grad_accum
                param_norm = float(p_grad.norm(norm_type))
            else:
                p_grad = (
                    p.grad.data
                    / self.trainer.accelerator.precision_plugin.scaler.get_scale()
                    / args.batch_size
                )
                param_norm = float(p_grad.norm(norm_type))
            all_norms.append(param_norm)
            # local_norm.add_(p.grad.norm(norm_type))

        total_norm = float(torch.tensor(all_norms).norm(norm_type))
        # norms[f'grad_{norm_type}_norm_total'] = round(total_norm, 4)
        # print("total_norm", total_norm, model.device, local_norm, self.trainer.accelerator.precision_plugin.scaler.get_scale())
        if not should_accumulate:
            return {
                "total_grad_norm": total_norm,
                "batch_size": args.batch_size * self.trainer.world_size,
                "grad_accum": args.grad_accum,
            }
        else:
            return {
                "local_grad_norm %s" % model.device: total_norm,
                "local_scale": self.trainer.accelerator.precision_plugin.scaler.get_scale(),
            }

    def _track_gradient_norm(self):
        grad_norm_dict = {}
        if (self.trainer.global_step + 1) % self.trainer.log_every_n_steps == 0:
            if float(self.trainer.track_grad_norm) > 0:
                model = self.trainer.lightning_module
                grad_norm_dict = self.grad_norm(model, self.trainer.track_grad_norm)
        return grad_norm_dict

    def backward(self, result, optimizer, opt_idx, *args, **kwargs):
        self.trainer.dev_debugger.track_event("backward_call")

        should_accumulate = self.should_accumulate()
        # print(should_accumulate)
        # backward can be called manually in the training loop
        if isinstance(result, torch.Tensor):
            self.trainer.accelerator.backward(
                result, optimizer, opt_idx, should_accumulate, *args, **kwargs
            )
        else:
            result.closure_loss = self.trainer.accelerator.backward(
                result.closure_loss,
                optimizer,
                opt_idx,
                should_accumulate,
                *args,
                **kwargs,
            )

        if not self.should_accumulate():
            # track gradients
            # print("track gradient with should_accumulate False")
            cur_grad_norm_dict = self.track_and_norm_grad(optimizer=optimizer)
            if "total_grad_norm" in self._cur_grad_norm_dict:
                B_small, B_big = (
                    self._cur_grad_norm_dict["batch_size"],
                    self._cur_grad_norm_dict["batch_size"]
                    * self._cur_grad_norm_dict["grad_accum"],
                )
                grad_norm_B_big = self._cur_grad_norm_dict["total_grad_norm"]
                grad_norm_B_small = []
                if not hasattr(self, "grad_norm_dict") or (
                    hasattr(self, "grad_norm_dict") and self.grad_norm_dict is None
                ):
                    B_critical = B_big
                else:
                    for item in self.grad_norm_dict:
                        if "local_grad_norm" in item:
                            grad_norm_B_small.append(self.grad_norm_dict[item])

                    grad_norm_B_small = np.average(grad_norm_B_small)
                    g2 = (
                        1
                        / (B_big - B_small)
                        * (B_big * grad_norm_B_big - B_small * grad_norm_B_small)
                    )
                    s = (
                        1
                        / (1 / B_small - 1 / B_big)
                        * (grad_norm_B_small - grad_norm_B_big)
                    )
                    B_critical = s / g2
                    self._cur_grad_norm_dict.update(self.grad_norm_dict)
                self._cur_grad_norm_dict.update({"critical_batch_size": B_critical})
                for e in ["batch_size", "grad_accum"]:
                    self._cur_grad_norm_dict.pop(e)
                # print(self._cur_grad_norm_dict)
            self.grad_norm_dict = None
        else:
            # print("track gradient with should_accumulate True")
            # first gradient accumulation step !!!!!!!!!!!!
            if hasattr(self, "grad_norm_dict") and self.grad_norm_dict is None:
                model = self.trainer.lightning_module
                self.grad_norm_dict = self.grad_norm(
                    model, self.trainer.track_grad_norm, True
                )

    def run_training_epoch(self):
        # modify dataloader if needed (ddp, etc...)
        train_dataloader = self.trainer.accelerator.process_dataloader(
            self.trainer.train_dataloader
        )

        # track epoch output
        epoch_output = [[] for _ in range(self.num_optimizers)]

        train_dataloader = self.trainer.data_connector.get_profiled_train_dataloader(
            train_dataloader
        )
        dataloader_idx = 0

        batch_idx = None
        is_last_batch = None

        accum_bsz, accum_bsz_grad_step = 0, 0
        for batch_idx, (batch, is_last_batch) in train_dataloader:
            self.trainer.batch_idx = batch_idx
            self.trainer.is_last_batch = is_last_batch

            # warmup the batch size via truncation and gradient accumulation
            # hack the deepest into the PTL to make it happen
            if self.args.warmup_bsz != 0:
                # for key in batch.keys():
                #     print(key, batch[key].shape, batch[key].device, batch[key].numel(), self.trainer.accumulate_grad_batches, self.trainer.model.device)
                input_ids = batch["input_ids"]

                final_bsz = (
                    input_ids.shape[0] * self.args.grad_accum * self.trainer.world_size
                )
                start_bsz = 64

                current_bsz = start_bsz + (final_bsz - start_bsz) * min(
                    1.0, accum_bsz / self.args.warmup_bsz
                )
                # print("before current_bsz", current_bsz, accum_bsz)
                if current_bsz >= final_bsz:
                    self.trainer.accumulate_grad_batches = self.args.grad_accum
                else:
                    current_bsz = current_bsz // self.trainer.world_size
                    # try to reset gradient accum steps
                    grad_accum = int(max(1, current_bsz // input_ids.shape[0]))

                    if grad_accum == 1 or accum_bsz_grad_step <= 0:
                        if grad_accum != 1 and accum_bsz_grad_step == 0:
                            accum_bsz_grad_step = grad_accum
                        self.trainer.accumulate_grad_batches = grad_accum
                        bsz_after_chunk = int(
                            current_bsz // self.trainer.accumulate_grad_batches
                        )
                    else:
                        accum_bsz_grad_step -= 1

                    # try to chunk the inputs
                    # print("current_bsz", current_bsz, "grad_accum", grad_accum, self.trainer.accumulate_grad_batches, accum_bsz_grad_step, self.should_accumulate(), 'bsz_after_chunk', bsz_after_chunk, input_ids.shape[0])
                    if bsz_after_chunk < input_ids.shape[0]:
                        for key in batch.keys():
                            batch[key] = torch.narrow(
                                batch[key], 0, 0, bsz_after_chunk
                            )  # .to( self.trainer.model.device )

                accum_bsz += batch["input_ids"].numel()

            if self.args.warmup_seq != 0:

                input_ids = batch["input_ids"]

                start_seq = 64
                final_seq = input_ids.shape[1]

                current_seq = int(
                    start_seq
                    + (final_seq - start_seq)
                    * min(1.0, accum_bsz / self.args.warmup_seq)
                )
                if accum_bsz_grad_step <= 0:
                    accum_bsz_grad_step = self.trainer.accumulate_grad_batches
                else:
                    accum_bsz_grad_step -= 1

                if current_seq < final_seq:
                    for key in batch.keys():
                        batch[key] = torch.narrow(batch[key], 1, 0, current_seq)

                accum_bsz += batch["input_ids"].numel()

            # ------------------------------------
            # TRAINING_STEP + TRAINING_STEP_END
            # ------------------------------------
            with self.trainer.profiler.profile("run_training_batch"):
                batch_output = self.run_training_batch(batch, batch_idx, dataloader_idx)

            # when returning -1 from train_step, we end epoch early
            if batch_output.signal == -1:
                break

            # hook
            # TODO: add outputs to batches
            self.on_train_batch_end(
                epoch_output,
                batch_output.training_step_output_for_epoch_end,
                batch,
                batch_idx,
                dataloader_idx,
            )

            # -----------------------------------------
            # SAVE METRICS TO LOGGERS
            # -----------------------------------------
            self.trainer.logger_connector.log_train_step_metrics(batch_output)

            # -----------------------------------------
            # VALIDATE IF NEEDED
            # -----------------------------------------
            should_check_val = self._should_check_val_fx(batch_idx, is_last_batch)
            if should_check_val:
                self.trainer.validating = True
                self.trainer.run_evaluation()
                self.trainer.training = True

            # -----------------------------------------
            # SAVE LOGGERS (ie: Tensorboard, etc...)
            # -----------------------------------------
            self.save_loggers_on_train_batch_end()

            # update LR schedulers
            monitor_metrics = copy.deepcopy(
                self.trainer.logger_connector.callback_metrics
            )
            self.update_train_loop_lr_schedulers(monitor_metrics=monitor_metrics)
            self.trainer.checkpoint_connector.has_trained = True

            self.trainer.total_batch_idx += 1

            # max steps reached, end training
            if (
                self.trainer.max_steps is not None
                and self.trainer.max_steps <= self.trainer.global_step + 1
                and self._accumulated_batches_reached()
            ):
                break

            # end epoch early
            # stop when the flag is changed or we've gone past the amount
            # requested in the batches
            if self.trainer.should_stop:
                break

            # stop epoch if we limited the number of training batches
            if self._num_training_batches_reached(is_last_batch):
                break

            # progress global step according to grads progress
            self.increment_accumulated_grad_global_step()

        if batch_idx is None:
            # dataloader/iterator did not produce a batch
            return

        # handle epoch_output on epoch end
        self.on_train_epoch_end(epoch_output)

        # log epoch metrics
        self.trainer.logger_connector.log_train_epoch_end_metrics(epoch_output)

        should_check_val = self._should_check_val_fx(
            batch_idx, is_last_batch, on_epoch=True
        )
        should_skip_eval = self.trainer.evaluation_loop.should_skip_evaluation(
            self.trainer.num_val_batches
        )
        should_train_only = self.trainer.disable_validation or should_skip_eval

        # update epoch level lr_schedulers if no val loop outside train loop is triggered
        if not should_check_val or should_train_only:
            self.trainer.optimizer_connector.update_learning_rates(interval="epoch")

        if should_train_only:
            self.check_checkpoint_callback(True)

        if should_check_val:
            self.trainer.validating = True
            self.trainer.run_evaluation(on_epoch=True)
            self.trainer.training = True

        if batch_output.signal != -1:
            self.increment_accumulated_grad_global_step()


class Pretrainer(ptl.LightningModule):
    def __init__(self):
        super().__init__()

        self.args = args  # hparams
        self._set_hparams(self.args)  # v1.3.5 ptl issue
        # self.hparams = self.args

        # self.model = AutoModelForMaskedLM.from_pretrained(args.model)
        self.model = AutoModelForCausalLM.from_pretrained(args.model)
        if args.random:
            if args.layers is not None and args.size is not None:
                raise False
            if args.layers is not None:
                self.model.config.n_layer = args.layers
            if args.size is not None:
                if args.size == "GPT2_base":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8
                elif args.size == "GPT2_large":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                elif args.size == "GPT2_base_div2_width":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 384
                    self.model.config.n_head = 4
                elif args.size == "GPT2_base_div2_depth":
                    self.model.config.n_layer = 6
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8

                elif args.size == "GPT2_large_div4_width":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 384
                    self.model.config.n_head = 4

                elif args.size == "GPT2_large_div2_width":
                    self.model.config.n_layer = 24
                    self.model.config.n_embd = 768
                    self.model.config.n_head = 8
                elif args.size == "GPT2_large_div4_depth":
                    self.model.config.n_layer = 6
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                elif args.size == "GPT2_large_div2_depth":
                    self.model.config.n_layer = 12
                    self.model.config.n_embd = 1536
                    self.model.config.n_head = 16
                else:
                    assert False

            assert self.model.config.n_positions == 1024
            self.model.config.n_positions = args.seqlen
            self.model = GPT2LMHeadModel(config=self.model.config)
        else:
            assert args.layers is None
            assert args.size is None

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
        self.pad_token_id = tokenizer.pad_token_id
        self.eos_token_id = tokenizer.eos_token_id or tokenizer.sep_token_id
        self.bos_token_id = tokenizer.bos_token_id or tokenizer.cls_token_id

        logger.info(
            f"Creating dataset cache from dir {self.args.input_dir}. This could be slow the first time."
        )
        MMapTextDataset.raw_text_to_mmap(args)

        # TODO: add support for other objective functions (whole word masking, BART, Pegasus)
        # self.data_collator = DataCollatorForLanguageModeling(
        #     tokenizer=tokenizer, mlm=True, mlm_probability=self.args.mlm_prob
        # )
        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer, mlm=False
        )
        self.start_time = 0

    def to(self, *args, **kwargs):
        param_count_before_to = len(list(self.parameters()))
        super().to(*args, **kwargs)
        if self.trainer.on_tpu:
            # if self.trainer.use_tpu:
            # need to re-tie the weights after moving to XLA!
            self.model.tie_weights()
            if "roberta" in self.args.model or "longformer" in self.args.model:
                self.model.lm_head.bias = self.model.lm_head.decoder.bias
        param_count_after_to = len(list(self.parameters()))
        assert param_count_before_to == param_count_after_to

    def forward(self, inputs):
        # for MLM
        # get the padding mask - 1 for NOT masked, 0 for MASKED/PAD
        # attention_mask = (input_ids != self.pad_token_id).int()

        # output is loss, prediction_scores, hidden_states
        # output = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # for LM
        output = self.model(**inputs)
        return output[0]  # loss

    def training_step(self, batch, batch_nb):
        loss = self(batch)
        input_ids = batch["input_ids"]
        tensorboard_logs = {
            "input_size": input_ids.numel(),
            "token_per_step": input_ids.numel()
            * self.trainer.accumulate_grad_batches
            * self.trainer.world_size,
        }
        # if not self.use_tpu:
        if not self.trainer.on_tpu:
            # logging additional losses is slow on tpu
            tensorboard_logs["lm_loss"] = loss
            tensorboard_logs["lm_bpc"] = loss / math.log(2)
            tensorboard_logs["lm_perplexity"] = torch.exp(loss)

        if self.start_time != 0:
            # torch.cuda.synchronize()
            elapsed_time = time.monotonic() - self.start_time
            tensorboard_logs["second_per_batch"] = elapsed_time
        self.start_time = time.monotonic()

        if self.on_gpu:
            tensorboard_logs["memory"] = (
                torch.cuda.memory_allocated(loss.device) / 1024 ** 3
            )

        for k, v in tensorboard_logs.items():
            self.log(k, v)

        return {"loss": loss}

    def on_train_batch_start(self, *args, **kwargs):
        self._start = time.monotonic()

    def on_train_batch_end(self, *args, **kwargs):
        delta = time.monotonic() - self._start
        self.log("time_per_batch", delta, on_step=True, on_epoch=False)

    def validation_step(self, batch, batch_nb):
        # TODO: log how long evaluation takes
        self.start_time = 0  # reset training_step timer

        loss = self(batch)
        tensorboard_logs = {
            "val_lm_loss": loss.detach(),
        }
        return {"val_loss": tensorboard_logs["val_lm_loss"], "log": tensorboard_logs}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack(
            [x["log"]["val_lm_loss"] for x in outputs if "val_lm_loss" in x["log"]]
        ).mean()
        if self.trainer.accelerator_connector.use_ddp:
            # TODO: PTL is already doing this. Is it still needed here?
            # https://github.com/PyTorchLightning/pytorch-lightning/blob/0.8.5/pytorch_lightning/metrics/converters.py#L251
            torch.distributed.all_reduce(avg_loss, op=torch.distributed.ReduceOp.SUM)
            avg_loss /= torch.distributed.get_world_size()
        elif self.on_tpu:
            avg_loss = xm.all_reduce(xm.REDUCE_SUM, avg_loss) / xm.xrt_world_size()

        self.log(
            "val_loss",
            avg_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )

    def configure_optimizers(self):
        # no_decay = ["bias", "LayerNorm.weight"]

        # optimizer_grouped_parameters = [
        #     {
        #         "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
        #         "weight_decay": self.args.weight_decay,
        #     },
        #     {
        #         "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
        #         "weight_decay": 0.0,
        #     },
        # ]
        # optimizer_grouped_parameters

        optimizer = AdamW(
            self.parameters(),
            lr=self.args.lr,
            eps=self.args.adam_epsilon,
            betas=(self.args.adam_beta1, self.args.adam_beta2),
            correct_bias=False,
        )
        if self.args.restart_warmup_steps != 0 and self.args.restart_steps != 0:
            scheduler = get_restart_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.train_steps,
                restart_steps=self.args.restart_steps,
                restart_warmup_steps=self.args.restart_warmup_steps,
            )
        else:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.args.warmup_steps,
                num_training_steps=self.args.train_steps,
            )
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _get_loader(self, fname, is_train):
        dataset = MMapTextDataset(
            fname,
            chunk_size=self.args.seqlen,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
        )

        # TODO: consider `replace_sampler_ddp=True` and removing the following if statement
        # if self.trainer.use_ddp:
        if self.trainer.accelerator_connector.use_ddp:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset, shuffle=is_train
            )
            shuffle = False
        elif self.trainer.on_tpu:
            sampler = torch.utils.data.distributed.DistributedSampler(
                dataset,
                num_replicas=xm.xrt_world_size(),
                rank=xm.get_ordinal(),
                shuffle=is_train,
            )
            shuffle = False
        else:
            sampler = None
            shuffle = is_train

        loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=self.args.num_workers,
            collate_fn=self.data_collator,
            drop_last=is_train,
        )
        return loader

    def train_dataloader(self):
        return self._get_loader(f"{self.args.input_dir}/cache/train.bin", True)

    def val_dataloader(self):
        return self._get_loader(f"{self.args.input_dir}/cache/val.bin", False)

    def grad_norm(self, norm_type):
        # Override PTL `grad_norm` function to only return `total_grad_norm` instead norms of individual params
        # TODO: grad_norm reporting needs to take fp16 loss scale into account
        parameters = [p for p in self.parameters() if p.grad is not None]
        device = parameters[0].device
        total_norm = torch.zeros([], device=device if parameters else None)
        norm_type = float(norm_type)
        for p in parameters:
            param_norm = p.grad.norm(norm_type)
            total_norm.add_(param_norm)
        return {"total_grad_norm": total_norm}

    @staticmethod
    def add_args(parser):
        parser.add_argument("--seed", type=int, default=3)

        # Dataset. Some of these params are only useful when generating the dataset cache
        parser.add_argument(
            "--input_dir", type=str, default="/net/nfs2.allennlp/shengs/c4/"
        )
        parser.add_argument(
            "--output_dir", type=str, default="/net/nfs2.allennlp/shengs/c4/"
        )

        parser.add_argument("--data_type", type=str, default="tfrecord")
        parser.add_argument(
            "--add_sep_after_doc",
            action="store_true",
            default=False,
            help="add sep token after document",
        )

        # Used only at the preprocessing phase
        parser.add_argument("--train_dev_split", type=float, default=0.05)
        parser.add_argument("--shard_size", type=int, default=1024 ** 3 // 4)  # 250MB
        parser.add_argument("--num_preprocessing_workers", type=int, default=1)
        # Used only at the training phase
        parser.add_argument("--seqlen", type=int, default=512)

        # HF model loading
        parser.add_argument("--tokenizer", type=str, default="gpt2")
        parser.add_argument("--model", type=str, default="gpt2")
        parser.add_argument("--doubling", type=str)  # could be layers / weights
        parser.add_argument(
            "--doubling_layers", type=str
        )  # could be alternate_id, append_id,  alternate_copy, append_copy
        # parser.add_argument("--noise_std", type=float, default=0.0)
        parser.add_argument(
            "--warmup_bsz", type=int, default=0, help="# warmup batch size"
        )
        parser.add_argument(
            "--warmup_seq", type=int, default=0, help="# warmup sequence length"
        )

        parser.add_argument("--random", default=False, action="store_true")
        parser.add_argument("--layers", type=int)
        parser.add_argument("--size", type=str)

        # Checkpointing and logging
        parser.add_argument("--save_dir", type=str, default="runs/")
        parser.add_argument(
            "--save_prefix",
            type=str,
            default="test",
            help="path of output directory is --save_dir/--save_prefix",
        )
        parser.add_argument(
            "--resume",
            type=str,
            default=None,  # It is better to use a different output dir.
            help="Path to a checkpoint to load model weights and training state. It overwrites args",
        )
        parser.add_argument(
            "--resume_model_only",
            type=str,
            default=None,
            help="Path to a checkpoint to load model weights but not training state",
        )
        parser.add_argument("--reset_optimizer", default=False, action="store_true")
        parser.add_argument("--reset_lr_scheduler", default=False, action="store_true")
        parser.add_argument("--log_rate", type=int, default=10)
        parser.add_argument(
            "--disable_checkpointing", action="store_true", default=False
        )

        # Training hyperparams
        parser.add_argument("--lr", type=float, default=1e-5)
        parser.add_argument(
            "--train_steps", type=int, default=3000, help="# training grad. updates"
        )
        parser.add_argument(
            "--warmup_steps", type=int, default=1000, help="# warmup grad. updates"
        )
        parser.add_argument(
            "--val_every",
            type=int,
            default=100,
            help="# training grad. updates between evaluations",
        )
        parser.add_argument(
            "--val_batches", type=int, default=1000, help="# evaluation **batches**"
        )
        parser.add_argument("--weight_decay", type=float, default=0.01)
        parser.add_argument("--adam_epsilon", type=float, default=1e-6)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.98)
        parser.add_argument(
            "--grad_clip", type=float, default=0
        )  # TODO: test this with fp16. Likely not working

        # RoBERTa's tokens_per_step = 2^18 = 512(seqlen) x 1(gpu_count) x 32(batch_size) x 16(grad_accum)
        parser.add_argument("--batch_size", type=int, default=32)
        parser.add_argument("--grad_accum", type=int, default=1)

        # Compute resources
        parser.add_argument("--fp16", default=False, action="store_true")
        parser.add_argument("--num_workers", type=int, default=0)
        parser.add_argument(
            "--gpu_count",
            type=int,
            default=1,  # `--gpus` is reserved for internal use by PTL
            help="Number of gpus. This respects `CUDA_VISIBLE_DEVICES`",
        )

        # For restarting with warmup
        parser.add_argument(
            "--restart_warmup_steps",
            type=int,
            default=0,
            help="# warmup grad. updates after restart",
        )
        parser.add_argument(
            "--restart_steps",
            type=int,
            default=0,
            help="# restart steps, should be the same as set_global_steps",
        )
        # For multi-node training, use the PyTorch launch script. The script and instructions can be found here:
        # https://github.com/pytorch/pytorch/blob/master/torch/distributed/launch.py.
        # To run PTL in a mode compatible with the launch script, two things are needed:
        #   - pass the argument `--use_env` to `torch.distributed.launch`
        #   - make sure `--nproc_per_node` matches `--gpu_count` and `--nnodes` matches `--node_count`.
        # For example, to run on 2 nodes, 3 gpus each, the command line on node rank 1 would be like:
        #   >>>> python -m torch.distributed.launch  \
        #               --use_env  --nnodes 2  --nproc_per_node 3  \
        #               --node_rank 1  --master_addr s2-server4  --master_port 12343  \
        #               scripts/pretrain.py  \
        #               --gpu_count 2  --node_count 2  \
        #               --input_dir my_data_dir  --save_prefix test_multinode
        parser.add_argument(
            "--node_count",
            type=int,
            default=1,
            help="Number of nodes. It needs to match --nnodes of torch.distributed.launch",
        )
        parser.add_argument("--tpu_core_count", type=int, default=None)

        return parser


def main(args):
    random.seed(args.seed * 10)
    np.random.seed(args.seed * 100)
    torch.manual_seed(args.seed * 1000)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed * 10000)

    if args.resume_model_only is not None:
        pretrainer = Pretrainer.load_from_checkpoint(args.resume_model_only)
    else:
        pretrainer = Pretrainer()

    if args.doubling is not None:

        doubled_resume = (
            args.resume + ".doubled_weights"
            if args.doubling == "weights"
            else args.resume + ".doubled_layer"
        )
        print(doubled_resume)
        exsit_flag = os.path.isfile(doubled_resume)

        if exsit_flag:
            args.resume = doubled_resume
            print(
                "================== warning: reusing old ckpt ======================="
            )

        # doubling the checkpoint before doubling the in-memory model
        if args.resume is not None and not exsit_flag:
            ckpt = torch.load(args.resume)

            # doubling state dict of the saved model
            if args.doubling == "weights":
                model_state_dict = ckpt["state_dict"]
                ckpt["state_dict"] = double_state_dict(
                    model_state_dict, is_double_embedding=True
                )

                # doubling state dict of the saved optimizer
                # no_decay = ["bias", "LayerNorm.weight"]
                # optimizer_params_by_name = [(n, p.shape) for n, p in pretrainer.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad]
                # optimizer_params_by_name.extend([(n, p.shape) for n, p in pretrainer.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad])
                optimizer_params_by_name = [
                    (n, p.shape) for n, p in pretrainer.named_parameters()
                ]
                assert len(optimizer_params_by_name) == len(
                    ckpt["optimizer_states"][0]["state"]
                )
                for (param_name, param_shape), param in zip(
                    optimizer_params_by_name,
                    ckpt["optimizer_states"][0]["state"].values(),
                ):
                    assert param["exp_avg"].shape == param_shape
                    assert param["exp_avg_sq"].shape == param_shape
                    param["exp_avg"] = double_param(
                        param_name,
                        param["exp_avg"],
                        is_double_embedding=True,
                        is_grad=True,
                        is_avg_sq=False,
                    )
                    param["exp_avg_sq"] = double_param(
                        param_name,
                        param["exp_avg_sq"],
                        is_double_embedding=True,
                        is_grad=True,
                        is_avg_sq=True,
                    )

                    # print(name_shape[0])
                args.resume += ".doubled_weights"
            elif args.doubling == "layers":
                model_state_dict = ckpt["state_dict"]
                # hack for doubling the layers
                prefix = "model.transformer.h"
                map_positions, copy_positions = {}, {}
                for key in model_state_dict:
                    if prefix in key:
                        layer_idx = re.findall("[-\d]+", key)[0]
                        origin_idx = prefix + "." + str(int(layer_idx))
                        if "alternate" in args.doubling_layers:
                            insert_idx = prefix + "." + str(int(layer_idx) * 2 + 1)
                            origin_key = key.replace(
                                origin_idx, prefix + "." + str(int(layer_idx) * 2)
                            )
                        elif "append" in args.doubling_layers:
                            insert_idx = (
                                prefix
                                + "."
                                + str(pretrainer.model.config.n_layer + int(layer_idx))
                            )
                            origin_key = key

                        insert_key = key.replace(origin_idx, insert_idx)

                        map_positions[key] = [(origin_key, False), (insert_key, False)]
                        copy_positions[insert_key] = (key, False)
                        copy_positions[origin_key] = (key, True)

                is_identical = "id" in args.doubling_layers

                ckpt["state_dict"] = deep_state_dict(
                    model_state_dict,
                    is_identical=is_identical,
                    map_positions=map_positions,
                )

                # deal with the optimizer state
                original_optimizer_params_by_name = [
                    (n, p.shape) for n, p in pretrainer.named_parameters()
                ]
                # print( "original_optimizer_params_by_name", original_optimizer_params_by_name )
                # print( "ckpt optimizer_states", ckpt['optimizer_states'][0]['state'].keys() )
                layers = pretrainer.model.transformer.h
                n = len(layers)
                for i in range(n):
                    if "alternate" in args.doubling_layers:
                        layers.insert(i * 2, copy.deepcopy(layers[i * 2]))
                    elif "append" in args.doubling_layers:
                        layers.append(copy.deepcopy(layers[i]))

                pretrainer.model.config.n_layer *= 2
                pretrainer.model.tie_weights()
                new_optimizer_params_by_name = [
                    (n, p.shape) for n, p in pretrainer.named_parameters()
                ]

                new_optimizer_state = {
                    _: {} for _ in range(len(new_optimizer_params_by_name))
                }
                assert len(original_optimizer_params_by_name) == len(
                    ckpt["optimizer_states"][0]["state"]
                )
                original_optimizer_param_name_dict = {}
                for (param_name, param_shape), param in zip(
                    original_optimizer_params_by_name,
                    ckpt["optimizer_states"][0]["state"].values(),
                ):
                    assert param["exp_avg"].shape == param_shape
                    assert param["exp_avg_sq"].shape == param_shape
                    original_optimizer_param_name_dict[param_name] = copy.deepcopy(
                        param
                    )

                for param_idx, (param_name, param_shape) in enumerate(
                    new_optimizer_params_by_name
                ):
                    if copy_positions.get(param_name):
                        copy_param_name, copy_param_flag = copy_positions.get(
                            param_name
                        )
                        param_is_identical = copy_param_flag and is_identical
                        new_optimizer_state[param_idx] = copy.deepcopy(
                            original_optimizer_param_name_dict[copy_param_name]
                        )
                        new_optimizer_state[param_idx]["exp_avg"] = deep_param(
                            param_name,
                            original_optimizer_param_name_dict[copy_param_name][
                                "exp_avg"
                            ],
                            is_identical=param_is_identical,
                            is_grad=True,
                            is_avg_sq=False,
                        )
                        new_optimizer_state[param_idx]["exp_avg_sq"] = deep_param(
                            param_name,
                            original_optimizer_param_name_dict[copy_param_name][
                                "exp_avg_sq"
                            ],
                            is_identical=param_is_identical,
                            is_grad=True,
                            is_avg_sq=True,
                        )
                    else:
                        new_optimizer_state[param_idx] = copy.deepcopy(
                            original_optimizer_param_name_dict[param_name]
                        )

                ckpt["optimizer_states"][0]["state"] = new_optimizer_state
                ckpt["optimizer_states"][0]["param_groups"][0]["params"] = list(
                    new_optimizer_state.keys()
                )
                del original_optimizer_param_name_dict
                args.resume += ".doubled_layer"

            torch.save(ckpt, args.resume)
            exit()

        # we need to resume the model after the doubling
        if args.doubling == "layers":
            assert True
        elif args.doubling == "weights":
            assert True
        else:
            assert False

    # logger here is a SummaryWritter for tensorboard
    # it is used by the trainer, and certain return variables
    # from the model are automatically logged
    logger = TestTubeLogger(
        save_dir=args.save_dir, name=args.save_prefix, version=0  # always use version=0
    )

    checkpoint_callback = ModelCheckpoint(
        # model saved to filepath/prefix_....
        # filepath=os.path.join(args.save_dir, args.save_prefix, 'checkpoint'),
        # prefix='',
        dirpath=os.path.join(args.save_dir, args.save_prefix),
        filename="checkpoint-{epoch}-{step}",
        save_top_k=-1,
        # save_top_k=10,
        every_n_train_steps=250,
        save_last=True,
        verbose=True,
        # monitor='val_loss',
        # mode='min',
    )
    args.val_every *= args.grad_accum  # PTL is expecting number of batches_per_gpu
    print(args.val_every, args.disable_checkpointing, checkpoint_callback.__dict__)
    trainer = ptl.Trainer(
        gpus=args.gpu_count,
        num_nodes=args.node_count,
        tpu_cores=args.tpu_core_count,
        distributed_backend="ddp",  #  if (args.gpu_count > 1 or args.node_count > 1) else None,
        replace_sampler_ddp=False,
        track_grad_norm=2
        if args.tpu_core_count is None
        else -1,  # gradnorm logging is slow on tpus
        max_epochs=10000,
        min_epochs=0,
        max_steps=args.train_steps,  # run for many epochs, but stop after max_steps
        val_check_interval=args.val_every,
        limit_val_batches=args.val_batches,
        log_every_n_steps=args.log_rate,
        progress_bar_refresh_rate=args.log_rate,
        logger=logger,
        # checkpoint_callback=checkpoint_callback if not args.disable_checkpointing else None,
        accumulate_grad_batches=args.grad_accum,
        resume_from_checkpoint=args.resume,
        gradient_clip_val=args.grad_clip,
        precision=16 if args.fp16 else 32,
        amp_level="O2",
        num_sanity_val_steps=2,
        callbacks=[LearningRateMonitor(), checkpoint_callback],
        profiler="simple",
    )
    trainer.profiler.dirpath = os.path.join(args.save_dir, args.save_prefix)
    trainer.profiler.filename = "profiler"
    trainer.train_loop = MyTrainLoop(
        trainer, multiple_trainloader_mode="max_size_cycle", args=args
    )
    trainer.checkpoint_connector = MyCheckpointConnector(
        trainer,
        reset_lr_scheduler=args.reset_lr_scheduler,
        reset_optimizer=args.reset_optimizer,
        set_global_step=args.restart_steps + 1,
    )
    trainer.fit(pretrainer)


if __name__ == "__main__":
    parser = Pretrainer.add_args(argparse.ArgumentParser(description="pretrain"))
    args = parser.parse_args()
    main(args)
