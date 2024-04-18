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
        if key == "edge_index":
            return torch.tensor([[self.x_s.size(0)], [self.x_t.size(0)]])
        return super().__inc__(key, value, *args, **kwargs)


PROMPT_DICT = {
    "QA": ("INSTRUCTION:\n{instruction}\n\nQUESTION:\n{question}\n\nResponse:\n"),
    "RAGTableQA": (
        "<s>[INST]<<SYS>>\n"
        "Instruction: {instruction}\n<</SYS>>\n\n"
        "Question: {question}\n\n"
        "Documents: {documents}\n[/INST]\n\n"
        "Response:"
    ),
    "TableQA": (
        "<s>[INST]<<SYS>>\n"
        "Instruction: {instruction}\n<</SYS>>\n\n"
        "Table:\n {table}\n\n"
        "Question: {question}\n[/INST]\n\n"
        "Response:"
    ),
}

TASK_DICT = {
    "RAGTableQA": "Answer the question according the documents. Each line in the documents is a row or column extracted from a table.",
    "TableQA": "Answer the Question according to the Table.",
    "QA": "Answer the question in short without explaination.",
}
