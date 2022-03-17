from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from .common import get_group_texts_function


def get_openwebtext_dataset(
    tokenizer: GPT2Tokenizer,
    *,
    block_size: int = 1024,
    num_workers: int = 1,
    test_size: float = 0.01,
    seed: int = 17,
) -> Dataset:
    dataset_dict = load_dataset("openwebtext")

    # This dataset only comes with a single split, so we need to create our own train/test splits.
    dataset_dict = dataset_dict["train"].train_test_split(  # type: ignore[index,union-attr]
        shuffle=True, test_size=test_size, seed=seed
    )
    dataset = dataset_dict["test"]  # type: ignore[index]

    def tokenize_function(example):
        return tokenizer(example["text"])

    dataset = dataset.map(  # type: ignore[union-attr,call-arg]
        tokenize_function,
        batched=True,
        num_proc=num_workers,
        remove_columns=["text"],
        desc="Tokenizing dataset",
    )

    group_texts = get_group_texts_function(block_size)

    dataset = dataset.map(
        group_texts,
        batched=True,
        num_proc=num_workers,
        desc=f"Grouping texts into chunks of {block_size}",
    )

    return dataset  # type: ignore[return-value]
