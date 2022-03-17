"""
Evaluate pretrained GPT-2 models on standard datasets.

Adapted from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py.
"""  # noqa: E501

from itertools import islice
import math
import random
import os
import shutil
import tempfile
from typing import Optional, Any, Dict

from accelerate import Accelerator
import click
from click_help_colors import HelpColorsCommand, HelpColorsGroup
from more_itertools import chunked
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler, DistributedSampler
from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    GPT2Config,
    default_data_collator,
    DataCollatorForLanguageModeling,
)
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm

from tools.mmap_dataset import get_mmap_dataset
from tools.openwebtext_dataset import get_openwebtext_dataset
from tools.wikitext_dataset import get_wikitext_dataset


@click.group(
    cls=HelpColorsGroup,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
def main():
    if torch.cuda.is_available():
        click.echo("CUDA is available :)\n")
    else:
        click.secho("No CUDA devices available!\n", fg="red")


@main.command(
    cls=HelpColorsCommand,
    help_options_color="green",
    help_headers_color="yellow",
    context_settings={"max_content_width": 115},
)
@click.option("--model-name", default="gpt2")
@click.option(
    "--dataset",
    default="wikitext2",
    type=click.Choice(["wikitext2", "openwebtext", "mmap"]),
    show_choices=True,
    show_default=True,
    help="The dataset to evaluate on.",
)
@click.option(
    "--dataset-path",
    default=None,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="Path to the memory-mapped dataset (only valid when --dataset=mmap)",
)
@click.option(
    "--block-size",
    default=1024,
    show_default=True,
    help="""Input texts are blocked together into blocks of this size.
    This should probably match the max input size of the model.""",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="The batch size to use for evaluation.",
)
@click.option(
    "--checkpoint-file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
    help="A checkpoint file to load the weights from.",
)
@click.option(
    "--skip-loading-weights",
    is_flag=True,
    help="Leave the model's weights at their random initialization.",
)
@click.option(
    "--max-steps",
    default=None,
    type=click.INT,
)
def eval(
    model_name: str,
    dataset: str,
    dataset_path: Optional[str],
    block_size: int,
    batch_size: int,
    checkpoint_file: Optional[str],
    skip_loading_weights: bool,
    max_steps: Optional[int],
):
    """
    Evaluate a GPT-2 model on a dataset.
    """
    # Validate params.
    if dataset != "mmap" and dataset_path is not None:
        raise click.UsageError("'--dataset-path' only valid when '--dataset=mmap'")
    if dataset == "mmap" and dataset_path is None:
        raise click.UsageError("'--dataset-path' is required for this dataset type")

    click.secho("[1/3] Loading tokenizer and model...", fg="green")

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    if checkpoint_file is not None:
        click.echo(f"Loading checkpoint from {checkpoint_file}")

        # Load state dict.
        state_dict = torch.load(checkpoint_file)
        state_dict = state_dict['state_dict']
        state_dict = {k.replace("model.", ""): p for k, p in state_dict.items()}

        config = GPT2Config.from_pretrained(model_name)

        # Guess model size.
        print(state_dict.keys)
        config.n_embd = state_dict["transformer.wte.weight"].shape[1]
        config.n_layer = len(
            [key for key in state_dict if key.endswith("mlp.c_proj.bias")]
        )
        click.echo(
            f"Adjusting hidden_size to {config.n_embd}, num_layers to {config.n_layer}"
        )
        if config.n_embd == 1536 or config.n_embd == 3072:
            config.n_head = 16
        else:
            config.n_head = 8
        
        config.n_positions = 1024
        if "alibi" in checkpoint_file:
            config.alibi_embeddings = True
        if "rotary" in checkpoint_file:
            config.rotary_embeddings = True
        # Initialize model.
        model = GPT2LMHeadModel(config)
        if not skip_loading_weights:
            model.load_state_dict(state_dict, strict=True)
    elif skip_loading_weights:
        config = GPT2Config.from_pretrained(model_name)
        model = GPT2LMHeadModel(config)
    else:
        model = GPT2LMHeadModel.from_pretrained(model_name)

    model = model.cuda()

    click.secho("\n[2/3] Preprocessing data...", fg="green")

    dataloader = get_dataloader(
        dataset,
        tokenizer,
        block_size=block_size,
        batch_size=batch_size,
        dataset_path=dataset_path,
    )

    click.secho("\n[3/3] Evaluating model on data...", fg="green")

    model.eval()

    running_loss = 0.0
    total_batches = (
        len(dataloader) if max_steps is None else min([max_steps, len(dataloader)])
    )
    with tqdm(
        islice(dataloader, total_batches), desc="Evaluating", total=total_batches
    ) as batch_iterator:
        for i, batch in enumerate(batch_iterator):
            batch = {k: v.cuda() for k, v in batch.items()}
            # with torch.inference_mode():
            with torch.no_grad():
                outputs = model(**batch)

            loss = outputs.loss
            running_loss += loss.item()

            if i % 50 == 0 or i == total_batches - 1:
                mean_loss = running_loss / (i + 1)
                ppl = math.exp(mean_loss)
                batch_iterator.set_postfix(loss=mean_loss, ppl=ppl)

    mean_loss = running_loss / total_batches
    ppl = math.exp(mean_loss)

    click.secho(
        f"\nDone! Final loss: {mean_loss:.4f} (ppl = {ppl:.4f})", fg="green", bold=True
    )


@main.command()
@click.argument(
    "train-dataset-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "validation-dataset-path",
    type=click.Path(exists=True, dir_okay=False, resolve_path=True),
)
@click.argument(
    "log-dir",
    type=click.Path(exists=False, dir_okay=True, resolve_path=True),
)
@click.option(
    "--block-size",
    default=1024,
    show_default=True,
    help="""Input texts are blocked together into blocks of this size.
    This should probably match the max input size of the model.""",
)
@click.option(
    "--batch-size",
    default=32,
    show_default=True,
    help="The batch size to use for training and validation.",
)
@click.option(
    "--grad-accum",
    default=1,
    show_default=True,
    help="The number of gradient accumulation steps per update.",
)
@click.option(
    "--num-heads",
    default=4,
    show_default=True,
    help="The number of attention heads.",
)
@click.option(
    "--num-layers",
    default=4,
    show_default=True,
    help="The number of transformer layers.",
)
@click.option(
    "--hidden-size",
    default=256,
    show_default=True,
    help="The hidden size of the model.",
)
@click.option(
    "--lr",
    default=None,
    type=click.FLOAT,
    show_default=True,
    help="The learning rate. Defaults to '0.003239 - 0.0001395 log(N)'.",
)
@click.option(
    "--adam-epsilon",
    default=1e-6,
    show_default=True,
)
@click.option(
    "--adam-beta1",
    default=0.9,
    show_default=True,
)
@click.option(
    "--adam-beta2",
    default=0.95,
    show_default=True,
)
@click.option(
    "--warmup-steps",
    default=3000,
    show_default=True,
)
@click.option(
    "--train-steps",
    default=100000,
    show_default=True,
)
@click.option(
    "--validation-steps",
    default=50,
    show_default=True,
)
@click.option(
    "--validate-every",
    default=100,
    show_default=True,
)
@click.option(
    "--checkpoint-every",
    default=100,
    show_default=True,
)
@click.option(
    "--wandb-entity",
    default="allenai-team1",
    show_default=True,
)
@click.option(
    "--wandb-project",
    default="staged-training",
    show_default=True,
)
@click.option(
    "--amp", is_flag=True, help="""Train with automatic mixed-precision enabled."""
)
@click.option(
    "--recover", is_flag=True, help="""Restart training from a previous run."""
)
@click.option(
    "--recover-from",
    type=click.Path(exists=True, dir_okay=True, resolve_path=True),
    help="""Log directory to recover from if different.""",
)
@click.option(
    "--init-seed",
    default=42,
    show_default=True,
)
@click.option(
    "--data-seed",
    default=42,
    show_default=True,
)
@click.option(
    "--wandb-tags", help="""A comma-separated list of tags to assign to the W&B run."""
)
def train(
    train_dataset_path: str,
    validation_dataset_path: str,
    log_dir: str,
    block_size: int,
    batch_size: int,
    grad_accum: int,
    num_heads: int,
    num_layers: int,
    hidden_size: int,
    lr: float,
    adam_epsilon: float,
    adam_beta1: float,
    adam_beta2: float,
    warmup_steps: int,
    train_steps: int,
    validation_steps: int,
    validate_every: int,
    checkpoint_every: int,
    wandb_entity: str,
    wandb_project: str,
    amp: bool,
    recover: bool,
    recover_from: Optional[str],
    init_seed: int,
    data_seed: int,
    wandb_tags: Optional[str],
):
    """
    Train a GPT-2 model on C4.
    """
    accelerator = Accelerator(fp16=amp)
    device = accelerator.device
    is_distributed = accelerator.num_processes > 1

    state_path = os.path.join(
        log_dir, f"state_worker_{accelerator.local_process_index}.pt"
    )

    # Check log_dir.
    initial_state: Optional[Dict[str, Any]] = None
    if recover:
        if recover_from is not None:
            # Copy over contents to log_dir
            assert os.path.isdir(recover_from)
            assert os.path.isfile(
                os.path.join(
                    recover_from, f"state_worker_{accelerator.local_process_index}.pt"
                )
            )
            if accelerator.is_local_main_process:
                assert not os.path.exists(log_dir) or not os.listdir(log_dir)
                shutil.copytree(
                    recover_from,
                    log_dir,
                    #  dirs_exist_ok=True, only available for python >= 3.8
                )
            accelerator.wait_for_everyone()
        assert os.path.isdir(log_dir)
        assert os.path.isfile(state_path)
        click.echo(
            f"[Worker {accelerator.local_process_index}] Loading training state from {state_path}"
        )
        initial_state = torch.load(state_path)
    else:
        assert not os.path.exists(log_dir) or not os.listdir(log_dir)
        if accelerator.is_local_main_process:
            os.makedirs(log_dir, exist_ok=True)

    if accelerator.is_local_main_process:
        click.echo(f"Training on {accelerator.num_processes} devices")
        click.secho(
            "\n[1/3] Initializing tokenizer, model, and optimizer...", fg="green"
        )

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    config = GPT2Config.from_pretrained("gpt2")
    config.n_head = num_heads
    config.n_layer = num_layers
    config.n_embd = hidden_size

    # Set random seeds for model initialization.
    set_seeds(init_seed)

    model = GPT2LMHeadModel(config)

    total_params = 0
    for name, param in model.named_parameters():
        # Ignore embedding matrix when calculating size.
        if name == "transformer.wte.weight" or name == "transformer.wpe.weight":
            continue
        total_params += param.numel()
    if accelerator.is_local_main_process:
        click.echo(f"Total non-embedding parameters: {total_params:,}")

    if lr is None:
        lr = 0.003239 - 0.0001395 * np.log(total_params)

    optimizer = AdamW(
        model.parameters(),
        lr=lr,
        eps=adam_epsilon,
        betas=(adam_beta1, adam_beta2),
        correct_bias=False,
    )
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=train_steps
    )

    if accelerator.is_local_main_process:
        click.secho("\n[2/3] Loading data...", fg="green")

    # Set random seeds for data shuffling.
    set_seeds(data_seed)

    train_dataloader = get_dataloader(
        "mmap",
        tokenizer,
        dataset_path=train_dataset_path,
        block_size=block_size,
        batch_size=batch_size,
        shuffle=True,
        is_distributed=is_distributed,
        seed=data_seed,
    )

    validation_dataloader = get_dataloader(
        "mmap",
        tokenizer,
        dataset_path=validation_dataset_path,
        block_size=block_size,
        batch_size=batch_size,
        is_distributed=is_distributed,
        seed=data_seed,
    )

    # NOTE: We don't call `prepare()` on the dataloaders because that causes a memory leak,
    # and it's not necessary anyway.
    model, optimizer = accelerator.prepare(model, optimizer)

    validation_steps = min([len(validation_dataloader), validation_steps])

    # Load state.
    if initial_state is not None:
        optimizer.load_state_dict(initial_state["optimizer"])
        scheduler.load_state_dict(initial_state["scheduler"])
        model.load_state_dict(initial_state["model"])

    wandb_run_id: Optional[str] = None
    if accelerator.is_main_process:
        import wandb

        if initial_state is not None:
            wandb_run_id = initial_state["wandb_run_id"]
        else:
            wandb_run_id = wandb.util.generate_id()

        wandb.init(
            id=wandb_run_id,
            dir=log_dir,
            entity=wandb_entity,
            resume="auto",
            project=wandb_project,
            tags=None if not wandb_tags else wandb_tags.split(","),
            config={
                "init_seed": init_seed,
                "data_seed": data_seed,
                "total_params": total_params,
                "learning_rate": lr,
                "adam_beta1": adam_beta1,
                "adam_beta2": adam_beta2,
                "adam_epsilon": adam_epsilon,
                "batch_size": batch_size,
                "grad_accum": grad_accum,
                "num_processes": accelerator.num_processes,
                "effective_batch_size": batch_size
                * grad_accum
                * accelerator.num_processes,
            },
        )

    accelerator.wait_for_everyone()

    if accelerator.is_local_main_process:
        click.secho("\n[3/3] Training...", fg="green")

    model.train()
    val_loss: Optional[float] = None
    training_batches = enumerate(
        islice(
            chunked(cycle_through_epochs(train_dataloader, is_distributed), grad_accum),
            train_steps,
        )
    )

    # Catch data loader up to where we left off before.
    if initial_state is not None:
        click.echo(
            f"[Worker {accelerator.local_process_index}] "
            f"Catching data loader up to step {initial_state['training_steps']}..."
        )
        training_steps = initial_state["training_steps"]
        for step, batch in training_batches:
            del batch
            if step >= training_steps - 1:
                break
            accelerator.wait_for_everyone()

    with tqdm(
        training_batches,
        desc="Training",
        initial=0 if initial_state is None else initial_state["training_steps"],
        total=train_steps,
        disable=not accelerator.is_local_main_process,
    ) as train_batch_iterator:
        for step, batch in train_batch_iterator:

            def save_state():
                temp_state_file = tempfile.NamedTemporaryFile(
                    "w+b", dir=log_dir, delete=False, suffix="pt"
                )
                try:
                    torch.save(
                        {
                            "optimizer": optimizer.state_dict(),
                            "scheduler": scheduler.state_dict(),
                            "model": model.state_dict(),
                            "wandb_run_id": wandb_run_id,
                            "training_steps": step + 1,
                        },
                        temp_state_file.name,
                    )
                    temp_state_file.close()
                    os.replace(temp_state_file.name, state_path)
                finally:
                    if os.path.exists(temp_state_file.name):
                        os.remove(temp_state_file.name)

            optimizer.zero_grad()
            batch_loss = 0.0
            batch_ppl: Optional[float] = None
            for micro_batch in batch:
                # Move tensors to right device.
                micro_batch = {k: v.to(device) for k, v in micro_batch.items()}

                # Get loss.
                outputs = model(**micro_batch)
                micro_batch_loss = outputs.loss / len(batch)
                batch_loss += micro_batch_loss.detach().item()

                # Calculate gradients.
                accelerator.backward(micro_batch_loss)

                # Clean up.
                del micro_batch
                del outputs
                del micro_batch_loss

            del batch

            # Take step.
            optimizer.step()
            scheduler.step()

            should_log_this_step = step % 10 == 0 or step == train_steps - 1
            should_checkpoint_this_step = step > 0 and step % checkpoint_every == 0
            should_validate_this_step = (
                step > 0 and step % validate_every == 0
            ) or step == train_steps - 1

            # Gather average loss across all workers.
            if should_log_this_step or should_validate_this_step:
                batch_loss = (
                    accelerator.gather(
                        torch.tensor(batch_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )
                batch_ppl = math.exp(batch_loss)  # type: ignore[arg-type]

            # Update progress bar and log to W&B.
            if accelerator.is_local_main_process and should_log_this_step:
                if val_loss is not None:
                    train_batch_iterator.set_postfix(
                        batch_loss=batch_loss,
                        batch_ppl=batch_ppl,
                        val_loss=val_loss,
                        val_ppl=math.exp(val_loss),
                    )
                else:
                    train_batch_iterator.set_postfix(
                        batch_loss=batch_loss, batch_ppl=batch_ppl
                    )

                if accelerator.is_main_process:
                    wandb.log(
                        {
                            "batch_loss": batch_loss,
                            "batch_ppl": batch_ppl,
                            "lr": optimizer.param_groups[0]["lr"],
                        },
                        step=step,
                    )

            # Checkpoint.
            if should_checkpoint_this_step:
                save_state()

            # Validate.
            if should_validate_this_step:
                # Prepare model for validation.
                model.eval()
                optimizer.zero_grad()  # Not strictly necessary.

                running_loss = 0.0
                with tqdm(
                    islice(validation_dataloader, validation_steps),
                    desc="Validating",
                    total=validation_steps,
                    leave=False,
                    disable=not accelerator.is_local_main_process,
                ) as val_batch_iterator:
                    for val_step, val_batch in enumerate(val_batch_iterator):
                        # Move tensors to right device.
                        val_batch = {k: v.to(device) for k, v in val_batch.items()}

                        # Get loss.
                        with torch.inference_mode():
                            outputs = model(**val_batch)
                        loss = outputs.loss

                        running_loss += loss.item()
                        val_loss = running_loss / (val_step + 1)

                        # Update progress bar.
                        if accelerator.is_local_main_process and val_step % 10 == 0:
                            val_batch_iterator.set_postfix(
                                loss=val_loss, ppl=math.exp(val_loss)
                            )

                        # Clean up.
                        del val_batch
                        del outputs
                        del loss

                # Average loss across all workers.
                val_loss = (
                    accelerator.gather(
                        torch.tensor(val_loss, device=device).unsqueeze(0)
                    )
                    .mean()
                    .item()
                )

                # Reset model to train mode.
                model.train()

                # Update progress bar again with validation stats and log to W&B.
                val_ppl = math.exp(val_loss)  # type: ignore[arg-type]
                if accelerator.is_local_main_process:
                    train_batch_iterator.set_postfix(
                        batch_loss=batch_loss,
                        batch_ppl=batch_ppl,
                        val_loss=val_loss,
                        val_ppl=val_ppl,
                    )
                if accelerator.is_main_process:
                    wandb.log({"val_loss": val_loss, "val_ppl": val_ppl}, step=step)

    if accelerator.is_main_process:
        wandb.finish()

    click.secho("\nDone!", fg="green", bold=True)


def get_dataloader(
    dataset: str,
    tokenizer: GPT2Tokenizer,
    *,
    block_size: int = 1024,
    batch_size: int = 32,
    dataset_path: Optional[str] = None,
    shuffle: bool = False,
    is_distributed: bool = False,
    seed: int = 0,
) -> DataLoader:
    dataset_object: Dataset
    collator = default_data_collator
    if dataset == "wikitext2":
        dataset_object = get_wikitext_dataset(
            tokenizer,
            split="test",
            block_size=block_size,
            num_workers=1,
        )
    elif dataset == "openwebtext":
        dataset_object = get_openwebtext_dataset(
            tokenizer,
            block_size=block_size,
            num_workers=8,
        )
    elif dataset == "mmap":
        assert dataset_path is not None
        dataset_object = get_mmap_dataset(
            tokenizer, dataset_path, chunk_size=block_size
        )
        collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    else:
        raise ValueError(f"Unexpected dataset '{dataset}'")

    sampler: Optional[Sampler] = (
        DistributedSampler(dataset_object, shuffle=shuffle, seed=seed)
        if is_distributed
        else None
    )

    dataloader: DataLoader = DataLoader(
        dataset_object,
        collate_fn=collator,
        batch_size=batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
    )

    return dataloader


def cycle_through_epochs(dataloader: DataLoader, is_distributed: bool):
    epoch = 0
    while True:
        if is_distributed and isinstance(dataloader.sampler, DistributedSampler):
            dataloader.sampler.set_epoch(epoch)
        for batch in dataloader:
            yield batch
        epoch += 1


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    main()
