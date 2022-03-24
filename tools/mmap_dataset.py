import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer


class MMapTextDataset(Dataset):
    def __init__(
        self,
        mmap_filename: str,
        *,
        chunk_size: int = 1024,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256
    ):
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

    def __getitem__(self, idx: int):
        if self.token_ids is None:
            self.token_ids = np.memmap(self._mmap_filename, mode="r", dtype=np.uint16)
        from_index = idx * (self._chunk_size - 2)
        to_index = (idx + 1) * (self._chunk_size - 2)
        data = np.concatenate(
            (
                [self._bos_token_id],
                self.token_ids[from_index:to_index],  # type: ignore[index]
                [self._eos_token_id],
            )
        )
        return torch.tensor(data, dtype=torch.long)


def get_mmap_dataset(tokenizer: GPT2Tokenizer, filename: str, **kwargs) -> Dataset:
    return MMapTextDataset(
        filename,
        bos_token_id=tokenizer.bos_token_id or tokenizer.cls_token_id,
        eos_token_id=tokenizer.eos_token_id or tokenizer.sep_token_id,
        **kwargs,
    )
