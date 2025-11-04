from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import h5py
import numpy as np
import lmdb
import pickle
import os
import random

from run_timer import TIMER
from data_preprocess import decode_input_tensor, index_to_policy_vector, decode_legal_mask


class CCRL4040LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None
        with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def _init_file(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if self.env is None or self.txn is None:
            self._init_file()
        data = self.txn.get(f"{idx:08}".encode("ascii")) # type: ignore
        if data is None:
            raise RuntimeError(f"Missing LMDB entry for key={f'{idx:08}'.encode('ascii')}")
        try:
            sample = pickle.loads(data)
        except Exception as e:
            raise RuntimeError(f"Corrupt LMDB entry at idx={idx}, key={f'{idx:08}'.encode('ascii')}") from e

        board_tensor, metadata, enpassant, halfmoves, policy, value, legal_mask = sample
        legal_mask = decode_legal_mask(legal_mask)
        policy = index_to_policy_vector(policy)
        x = decode_input_tensor(board_tensor, metadata, enpassant, halfmoves)

        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(policy, dtype=torch.float32),
            torch.as_tensor(value, dtype=torch.float32),
            torch.as_tensor(legal_mask, dtype=torch.bool)
        )


class SoftCCRL4040LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.lmdb_path = lmdb_path
        self.env = None
        self.txn = None
        with lmdb.open(lmdb_path, readonly=True, lock=False) as env:
            with env.begin() as txn:
                self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def _init_file(self):
        self.env = lmdb.open(self.lmdb_path, readonly=True, lock=False)
        self.txn = self.env.begin()

    def __getitem__(self, idx):
        if self.env is None or self.txn is None:
            self._init_file()
        data = self.txn.get(f"{idx:08}".encode("ascii")) # type: ignore
        if data is None:
            raise RuntimeError(f"Missing LMDB entry for key={f'{idx:08}'.encode('ascii')}")
        try:
            sample = pickle.loads(data)
        except Exception as e:
            raise RuntimeError(f"Corrupt LMDB entry at idx={idx}, key={f'{idx:08}'.encode('ascii')}") from e

        board_tensor, metadata, enpassant, halfmoves, policy, value, legal_mask = sample
        legal_mask = decode_legal_mask(legal_mask)
        x = decode_input_tensor(board_tensor, metadata, enpassant, halfmoves)

        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(policy, dtype=torch.float32),
            torch.as_tensor(value, dtype=torch.float32),
            torch.as_tensor(legal_mask, dtype=torch.bool)
        )



if __name__ == "__main__":

    random.seed(2025)

    TIMER.start("Initializing Dataloader")
    # lmdb_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-20000000-500000-0.2-0.8-1.lmdb"
    lmdb_path = r"/teamspace/studios/this_studio/chess_bot/datasets/processed/soft-CCRL-4040-train-1000000-100000.lmdb"

    dataset = SoftCCRL4040LMDBDataset(lmdb_path)

    print(f"Size of dataset: {len(dataset)}")

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=256,
        # pin_memory=True,
        shuffle=False,
        num_workers=4,
        persistent_workers=True
    )
    # dataloader = DataLoader(
    #     CCRL4040LMDBDataset(lmdb_path),
    #     batch_size=320,
    #     num_workers=0,
    # )
    TIMER.stop("Initializing Dataloader")

    TIMER.start("Loading data batch")
    for i, (x, policy, value, legal_mask) in enumerate(dataloader):
        TIMER.lap("Loading data batch", i + 1, len(dataloader))
        print(x.shape, policy.shape, value.shape, legal_mask.shape)
        for j in range(x.shape[0]):
            if policy[j].max() != 1:
                print(f"Policy differs at index {j}")