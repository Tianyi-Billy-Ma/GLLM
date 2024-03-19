
from torch.utils.data.sampler import BatchSampler
from torch_geometric.data import Data
import torch


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
    
class BipartiteData(Data):
    def __inc__(self, key, value, *args, **kwargs):
        if key == 'edge_index':
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)





