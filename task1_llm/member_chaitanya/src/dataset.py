"""Fixed-length autoregressive windows over an encoded character stream.

Window i covers stream[i*stride : i*stride + block_size + 1];
input = window[:-1], target = window[1:] (next-character prediction).
"""
import numpy as np
import torch
from torch.utils.data import Dataset


class CharWindowDataset(Dataset):
    def __init__(self, ids: np.ndarray, block_size: int, stride: int):
        if stride is None or stride <= 0:
            raise ValueError("stride must be a positive integer (set data.train_stride in the config)")
        self.ids = torch.from_numpy(ids.astype(np.int64))
        self.block_size, self.stride = block_size, stride
        self.n = (len(ids) - (block_size + 1)) // stride + 1

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int):
        start = i * self.stride
        window = self.ids[start : start + self.block_size + 1]
        return window[:-1], window[1:]


def num_windows(stream_len: int, block_size: int, stride: int) -> int:
    return (stream_len - (block_size + 1)) // stride + 1
