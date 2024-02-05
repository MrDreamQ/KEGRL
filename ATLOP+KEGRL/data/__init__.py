import os
import random

import numpy as np
import torch
import json
from torch.utils.data import DataLoader

from .docred_prepro import docred_collate_fn
from .docred_dataset import DocRedDataset
from .dwie_dataset import DWIEDataset
from .dwie_prepro import dwie_roberta_train_collate_fn
from .dwie_prepro import dwie_collate_fn
from config import cfg


def get_loader(name: str, split: str, tokenizer, max_seq_length, batch_size, shuffle) -> DataLoader:
    if name in ['DocRED', 'Re-DocRED']:
        dataset = DocRedDataset(split, tokenizer, max_seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                            shuffle=shuffle, collate_fn=docred_collate_fn, drop_last=False)
        return loader, dataset
    elif name == 'DWIE':
        dataset = DWIEDataset(split, tokenizer, max_seq_length)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=8,
                            shuffle=shuffle, collate_fn=dwie_roberta_train_collate_fn if split == 'train' and cfg.MODEL.PLM_NAME == 'roberta-large' else dwie_collate_fn, 
                            drop_last=False)
        return loader, dataset
    else:
        raise NotImplementedError
