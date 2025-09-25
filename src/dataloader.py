from typing import Any
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import h5py
import numpy as np
import lmdb
import pickle
import os

from run_timer import TIMER


class CCRL4040H5Dataset(Dataset):
    def __init__(self, h5_path, batch_size=320, shuffle=True):
        super().__init__()
        self.h5_path: str = h5_path
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self._file: h5py.File | None = None

        with h5py.File(self.h5_path, "r") as f:
            self.length: int = f.attrs["num_samples"] # type: ignore

        self.indices: np.ndarray = np.arange(self.length)
        if self.shuffle:
            np.random.shuffle(self.indices)

    def _init_file(self):
        if self._file is None:
            self._file = h5py.File(self.h5_path, "r")
            self._features: Any = self._file["features"]
            self._policies: Any = self._file["policies"]
            self._values: Any = self._file["values"]

    def __len__(self):
        return (self.length + self.batch_size - 1) // self.batch_size

    def __getitem__(self, idx):
        # TIMER.start("loading data batch")

        # TIMER.start("setting up indices")
        self._init_file()
        start = idx * self.batch_size
        end = min((idx + 1) * self.batch_size, self.length)

        # h5py requires increasing order
        batch_idx = self.indices[start:end]
        sorted_idx = np.sort(batch_idx)
        # TIMER.stop("setting up indices")

        # TIMER.start("accessing h5py")
        x = self._features[sorted_idx]
        policy = self._policies[sorted_idx]
        value = self._values[sorted_idx]
        # TIMER.stop("accessing h5py")

        # restore original order
        # TIMER.start("converting to tensors")
        inverse = np.argsort(batch_idx)

        # convert to tensors
        x = torch.as_tensor(x[inverse], dtype=torch.float32)
        policy = torch.as_tensor(policy[inverse], dtype=torch.float32)
        value = torch.as_tensor(value[inverse], dtype=torch.float32)
        # TIMER.stop("converting to tensors")

        # TIMER.stop("loading data batch")
        return x, policy, value


def worker_init_fn(worker_id):
    """
    Worker init function for DataLoader. Ensures clean data loading
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset # type: ignore

    dataset._file = None # type: ignore
    dataset._features = None # type: ignore
    dataset._policies = None # type: ignore
    dataset._values = None # type: ignore


class CCRL4040LMDBDataset(Dataset):
    def __init__(self, lmdb_path):
        self.env = lmdb.open(lmdb_path, readonly=True, lock=False)
        with self.env.begin() as txn:
            self.length = pickle.loads(txn.get(b"__len__"))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        with self.env.begin() as txn:
            data = txn.get(f"{idx:08}".encode("ascii"))
            try:
                sample = pickle.loads(data)
            except Exception as e:
                raise RuntimeError(f"Corrupt LMDB entry at idx={idx}") from e
            # sample = pickle.loads(txn.get(f"{idx:08}".encode("ascii")))
        x, policy, value = sample
        return (
            torch.as_tensor(x, dtype=torch.float32),
            torch.as_tensor(policy, dtype=torch.float32),
            torch.as_tensor(value, dtype=torch.float32)
        )



if __name__ == "__main__":

    TIMER.start("Initializing Dataloader")
    h5_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5"
    lmdb_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040-train-small.lmdb"

    dataset = CCRL4040LMDBDataset(lmdb_path)

    dataloader = DataLoader(
        CCRL4040LMDBDataset(lmdb_path),
        batch_size=320,
        num_workers=4,
        worker_init_fn=worker_init_fn,
        persistent_workers=True
    )
    TIMER.stop("Initializing Dataloader")

    TIMER.start("Loading data batch")
    for i, (x, policy, value) in enumerate(dataloader):
        TIMER.lap("Loading data batch", i + 1, len(dataloader))
        print(x.shape, policy.shape, value.shape)
