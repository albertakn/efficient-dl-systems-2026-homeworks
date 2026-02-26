import os
import random

import torch
from torch.utils.data.dataset import Dataset
from torch.utils.data import Sampler, IterableDataset
from transformers import AutoTokenizer


MAX_LENGTH = 640
# Берём только первые DATASET_SIZE строк из данных ввиду большого размера датасета
DATASET_SIZE = 30000


def _read_train_texts(data_path: str) -> list[str]:
    texts = []
    for filename in sorted(os.listdir(data_path)):
        if "train" in filename and filename.endswith(".txt"):
            with open(os.path.join(data_path, filename), "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        texts.append(line)
    return texts[:DATASET_SIZE]


def _tokenize_and_filter(
    texts: list[str], tokenizer: AutoTokenizer, max_length: int
) -> list[list[int]]:
    tokenized = []
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_length:
            tokenized.append(ids)
    return tokenized


class BrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = _read_train_texts(data_path)
        self.tokenized = _tokenize_and_filter(self.texts, self.tokenizer, self.max_length)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx: int):
        ids = self.tokenized[idx]
        padded = ids + [self.tokenizer.pad_token_id] * (self.max_length - len(ids))
        padding_mask = torch.tensor([False] * len(ids) + [True] * (self.max_length - len(ids)))
        return torch.tensor(padded), padding_mask



class BigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = _read_train_texts(data_path)
        self.tokenized = _tokenize_and_filter(self.texts, self.tokenizer, self.max_length)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx: int):
        return self.tokenized[idx]

def collate_fn(
    batch: list[list[int]],
    tokenizer: AutoTokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pad each sequence of the incoming sequences list
    :param batch: a list of the objects received from the dataset by __getitem__
    :param tokenizer: tokenizer to use for padding
    :return: tuple of padded sequences and corresponding padding masks
    """
    max_batch_length = max(len(ids) for ids in batch)
    padded = [
        ids + [tokenizer.pad_token_id] * (max_batch_length - len(ids))
        for ids in batch
    ]
    padding_mask = [
        [False] * len(ids) + [True] * (max_batch_length - len(ids))
        for ids in batch
    ]
    return torch.tensor(padded), torch.tensor(padding_mask)
  



class UltraBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        super().__init__()
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.texts = _read_train_texts(data_path)
        self.tokenized = _tokenize_and_filter(self.texts, self.tokenizer, self.max_length)

        self.length_map = {}
        for i, ids in enumerate(self.tokenized):
            length = len(ids)
            if length not in self.length_map:
                self.length_map[length] = []
            self.length_map[length].append(i)

    def __len__(self):
        return len(self.tokenized)

    def __getitem__(self, idx: int):
        return self.tokenized[idx]


class UltraBigBrainBatchSampler(Sampler):
    def __init__(self, length_map: dict[int, list[int]], batch_size: int, k_value: int):
        super().__init__()
        self.batch_size = batch_size
        self.batches = []

        all_lengths = sorted(length_map.keys())
        current_bucket_indices = []
        
        bucket_min_len = all_lengths[0]

        for length in all_lengths:
            if length > bucket_min_len + k_value:
                self._make_batches(current_bucket_indices)
                current_bucket_indices = []
                bucket_min_len = length

            current_bucket_indices.extend(length_map[length])
        
        if current_bucket_indices:
            self._make_batches(current_bucket_indices)

    def _make_batches(self, indices):
        random.shuffle(indices)
        for i in range(0, len(indices), self.batch_size):
            self.batches.append(indices[i : i + self.batch_size])

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        random.shuffle(self.batches)
        for batch in self.batches:
            yield batch



class UltraDuperBigBrainDataset(Dataset):
    def __init__(self, data_path: str, max_length: int = MAX_LENGTH):
        pass

    def __getitem__(self, idx: int):
        pass
