import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np

class HDF5Dataset(Dataset):
    def __init__(self, h5_path, split="train"):
        self.h5_path = h5_path
        self.split = split
        breakpoint()
        
        # Open file once to get dataset length (don't keep it open here to prevent workers from crashing out)
        with h5py.File(self.h5_path, "r") as f:
            self.length = h5py.Dataset(f["features"]).shape[0]

    def __len__(self):
        
        return self.length

    def __getitem__(self, idx):
        # Open file in __getitem__ to avoid issues with multiprocessing (DataLoader workers)
        with h5py.File(self.h5_path, "r") as f:
            x = h5py.Dataset(f[f"features"])[idx]
            policy = h5py.Dataset(f[f"policies"])[idx]
            value = h5py.Dataset(f[f"values"])[idx]

        # Convert to torch tensors
        x = torch.from_numpy(np.array(x))
        policy = torch.tensor(policy, dtype=torch.long)
        value = torch.tensor(value, dtype=torch.long)

        return x, policy, value

if __name__ == "__main__":

    h5_path = "/teamspace/studios/this_studio/chess_bot/datasets/processed/CCRL-4040.h5"

    dataset = HDF5Dataset(h5_path)
