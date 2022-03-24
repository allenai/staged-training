from datasets import load_dataset
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

from .common import get_group_texts_function


def get_wikitext_dataset(
    tokenizer: GPT2Tokenizer,
    *,
    split: str = "test",
    block_size: int = 1024,
    num_workers: int = 1,
) -> Dataset:
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")  # type: ignore[assignment]
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="test")
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
