from torch.utils.data.sampler import BatchSampler
from torch_geometric.data import Data
import torch
from torch.utils.data import default_collate
import random
import os
import numpy as np


class HypergraphBatchSampler(BatchSampler):
    def __init__(self, num_HGs, batch_size):
        self.num_HGs = num_HGs
        self.batch_size = batch_size

    def __iter__(self):
        batch = []
        for idx in range(self.num_HGs):
            batch.append(idx)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        return len(self.HGs) // self.batch_size


def hypergraph_collcate_fn(batch):
    batched_data = default_collate(batch)

    num_hyperedges = torch.sum(batched_data.num_hyperedges)
    batched_data.num_hyperedges = num_hyperedges
    return batched_data


class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        
        elif key == "tidx":
            if self.reversed:
                return self.num_nodes
            else:
                return self.num_hyperedges
        elif key == "num_hyperedges":
            return 0
        return super().__inc__(key, value, *args, **kwargs)


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
