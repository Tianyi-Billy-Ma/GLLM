import os
import sys
import logging
import copy
import torch
import torch.nn as nn
from torch.optim import Adam

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch_sparse import add_
from torchmetrics.classification import (
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
    BinaryAccuracy,
)

import transformers
from transformers import AutoTokenizer, AutoConfig, HfArgumentParser
from transformers.optimization import AdamW, get_scheduler

from dataclasses import dataclass, field, fields
from typing import Optional

from model import Encoder, ContrastiveLoss
from data import TableDataModule
from src.augmentation import aug
from src.contrastive import ContrastiveLoss
from src.preprocess import add_special_tokens

@dataclass
class DataArguments:
    tokenizer_config_type: str = field(
        default="bert-base-uncased",
        metadata={"help": "bert-base-cased, bert-base-uncased etc"},
    )

    data_path: str = field(
        default=".data/pretrain/", metadata={"help": "Path to pretrain data"}
    )
    max_token_length: int = field(
        default=64,
        metadata={
            "help": "The maximum total input token length for cell/caption/header after tokenization. Sequences longer "
            "than this will be truncated."
        },
    )

    num_workers: Optional[int] = field(
        default=1,
        metadata={"help": "Number of workers for dataloader"},
    )

    seed: int = 3
    max_epoch: int = 5
    aug_ratio: float = 0.3


@dataclass
class OptimizerConfig:
    batch_size: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.02

    optimizer: str = "Adam"
    save_every_n_epochs: int = 1


class PlModel(pl.LightningModule):
    def __init__(self, model_cfg, optimizer_cfg):
        super().__init__()
        self.model = Encoder(model_cfg)
        self.model_cfg = model_cfg
        self.optimizer_cfg = optimizer_cfg

        self.loss = ContrastiveLoss(temperature=0.2)

    def training_step(self, batch, batch_index):
        batch_sub = copy.deepcopy(batch)

        emb_V, emb_E = self.model(batch)
        emb1 = torch.index_select(emb_E, 0, torch.nonzero(batch.mask).squeeze())

        emb_V, emb_E = self.model(batch)


def main():
    parser = HfArgumentParser((DataArguments, OptimizerConfig))
    parser = pl.Trainer.add_argparse_args(parser)

    data_args, optimizer_args, trainer_args = parser.parse_args_into_dataclasses()

    pl.utilities.seed_everything(data_args.seed)

    # ********************************* set up tokenizer and model config*********************************
    tokenizer = AutoTokenizer.from_pretrained(data_args.tokenizer_config_type)
    tokenizer = add_special_tokens(tokenizer)
    model_config = AutoConfig.from_pretrained(data_args.tokenizer_config_type)
    
    
    data_module = TableDataModule(tokenizer=tokenizer, data_args=data_args, seed=data_args.seed, batch_size=optimizer_args.batch_size)
    
    
if __name__ == "__main__":