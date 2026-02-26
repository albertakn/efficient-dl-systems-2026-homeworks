from enum import Enum

import torch
import logging
import time
import numpy as np
from typing import Optional
from torch import nn
from torch.utils.data import DataLoader
from torch.nn import functional as F
from dataset import BrainDataset, BigBrainDataset, UltraBigBrainDataset, UltraBigBrainBatchSampler, collate_fn
from transformer import PositionalEncoding
from transformer import generate_square_subsequent_mask

from pydantic import BaseModel
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataMode(Enum):
    BRAIN = 1
    BIG_BRAIN = 2
    ULTRA_BIG_BRAIN = 3
    ULTRA_DUPER_BIG_BRAIN = 4

class ModelConfig(BaseModel):
    vocab_size: int = 30522
    hidden_dim: int = 1024
    n_heads: int = 8
    feed_forward_dim: int = 2048
    num_layers: int = 1
    dropout: float = 0.1


class MockGPT2Model(nn.Module):
    def __init__(self, vocab_size: int, hidden_dim: int, n_heads: int, feed_forward_dim: int, num_layers: int, dropout: float):
        super().__init__()

        self.vocab_size = vocab_size
        self.n_heads = n_heads
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        #  В задании кажется ошибка, нужно использовать TransformerEncoderLayer вместо TransformerDecoderLayer,
        #  так как в TransformerDecoderLayer еще кросс-аттеншен есть и соответственно требуется обязательный аргумент: memory - выход encoder-а
        decoder_layers = nn.TransformerEncoderLayer(hidden_dim, n_heads, feed_forward_dim)
        self.decoder = nn.TransformerEncoder(decoder_layers, num_layers)
        self.lm_head = nn.Linear(hidden_dim, vocab_size)


    def forward(self, src: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
        # src: [batch, seq_len]
        src = src.transpose(0, 1)                   # [seq_len, batch]
        src = self.embedding(src)                    # [seq_len, batch, dim]
        src = self.pos_encoder(src)                  # [seq_len, batch, dim]
        seq_len = src.size(0)
        mask = generate_square_subsequent_mask(seq_len).to(src.device)
        output = self.decoder(src, mask=mask, src_key_padding_mask=padding_mask)        # [seq_len, batch, dim]
        output = output.transpose(0, 1)             # [batch, seq_len, dim]
        output = self.lm_head(output)               # [batch, seq_len, vocab]
        return output


def get_gpt2_model() -> torch.nn.Module:
    config = ModelConfig()
    return MockGPT2Model(**config.model_dump())


def run_epoch(data_mode: DataMode, data_path: str, model: torch.nn.Module, device: torch.device, k_value: Optional[int] = 10) -> None:

    if data_mode == DataMode.BRAIN:
        dataloader = DataLoader(BrainDataset(data_path), batch_size=64, shuffle=True)
    elif data_mode == DataMode.BIG_BRAIN:
        dataloader = DataLoader(BigBrainDataset(data_path), batch_size=64, collate_fn=collate_fn, shuffle=True)
    elif data_mode == DataMode.ULTRA_BIG_BRAIN: 
        dataset = UltraBigBrainDataset(data_path)
        sampler = UltraBigBrainBatchSampler(
            length_map=dataset.length_map, 
            batch_size=64, 
            k_value=k_value
        )
        dataloader = DataLoader(
            dataset = dataset,
            batch_sampler = sampler,
            collate_fn = collate_fn
        )
    elif data_mode == DataMode.ULTRA_DUPER_BIG_BRAIN:
        pass
    else:
        raise ValueError(f"Invalid data mode: {data_mode}")


    # считаем latency только после warmup_steps
    latencies = []
    warmup_steps = 5


    model.to(device)
    model.train(False)
    with torch.no_grad():
        for i, (src, padding_mask) in enumerate(tqdm(dataloader, desc="Running epoch")):

            src, padding_mask = src.to(device), padding_mask.to(device)
            inputs, targets = src[:, :-1], src[:, 1:]
            padding_mask = padding_mask[:, :-1]

            # Синхронизируемся перед стартом
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            
            output = model(inputs, padding_mask)

            # Добавил ignore_index=0, чтобы не учитывать padding токены при вычислении loss
            _ = F.cross_entropy(output.transpose(1, 2), targets, ignore_index=0)

            # Синхронизируемся после завершения вычислений на GPU
            torch.cuda.synchronize()
            end_time = time.perf_counter()
            
            latency = end_time - start_time
            if i >= warmup_steps:
                latencies.append(latency)
    

    logger.info(f"Mean latency: {np.mean(latencies)}")
    logger.info(f"Median latency: {np.median(latencies)}")
    logger.info(f"Min latency: {np.min(latencies)}")
    logger.info(f"Max latency: {np.max(latencies)}")
