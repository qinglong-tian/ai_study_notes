# data_module.py
# Simple HDF5 prior loader for nanoTabPFN

import h5py
import torch


def get_default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class PriorDumpDataLoader:
    """
    Iterates over an HDF5 prior dump.
    Yields dicts:
      - x: (B, R, C)
      - y: (B, R)
      - train_test_split_index: int
    """

    def __init__(self, filename: str, num_steps: int, batch_size: int, device=None):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.device = device or get_default_device()
        self.pointer = 0

        with h5py.File(self.filename, "r") as f:
            self.n_tasks = f["X"].shape[0]
            self.max_num_classes = int(f["max_num_classes"][0])

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                s = self.pointer
                e = s + self.batch_size

                nf = int(f["num_features"][s:e].max())
                nd = f["num_datapoints"][s:e]
                R = int(nd.max())

                x = torch.from_numpy(f["X"][s:e, :R, :nf]).to(self.device)
                y = torch.from_numpy(f["y"][s:e, :R]).to(self.device)
                split = int(f["single_eval_pos"][s])  # follow repo: take first split

                self.pointer += self.batch_size
                if self.pointer >= self.n_tasks:
                    self.pointer = 0

                yield {"x": x, "y": y, "train_test_split_index": split}

    def __len__(self):
        return self.num_steps
