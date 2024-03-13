from src import load_pickle, load_data
import os, os.path as osp
from arguments import parse_args
import torch
import numpy as np
from torch_geometric.data import Data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def construct_hypergraph(args, tables, embs):
    edge_index = None
    for t_idx, (key, val) in enumerate(tables.items()):
        num_cols, num_rows = len(val["header"]), len(val["rows"])

        row_edge_index, col_edge_index = [], []
        for he_idx in range(num_rows):
            row_edge_index.extend(
                [[he_idx * num_cols + col, he_idx] for col in range(num_cols)]
            )
        for he_idx in range(num_cols):
            col_edge_index.extend(
                [[row * num_cols + he_idx, he_idx] for row in range(num_rows)]
            )
        row_edge_index = np.array(row_edge_index)
        col_edge_index = np.array(col_edge_index)
        col_edge_index[:, 1] += num_rows  # shift the index of column hyperedge index
        c_edge_index = np.concatenate(
            [row_edge_index, col_edge_index], axis=0
        ).T  # (2, c_num_edges)

        assert (
            c_edge_index[0, :].max(0) + 1 == num_cols * num_rows
        ), "Nodes are not properly indexed"
        assert (
            c_edge_index[1, :].max() + 1 == num_cols + num_rows
        ), "Hyperedges are not properly indexed"
        if not edge_index:
            edge_index = c_edge_index
        else:
            c_edge_index[1, :] += edge_index[1].max()  # shift the index of hyperedges
            c_edge_index[0, :] += edge_index[0].max()
            edge_index = np.concatenate(
                [edge_index, c_edge_index], axis=1
            )  # (2, num_edges)

    HG = Data(x=torch.FloatTensor(embs), edge_index=torch.LongTensor(edge_index))
    return HG


def main(args):
    # Load data saved by pretrain_embedding.py
    pickle_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", f"{args.pretrain_model}.pickle"
    )
    dict_tables = load_pickle(pickle_path)
    t2n_mappings, q2t_mappings, qas, embeddings, tables = (
        dict_tables["t2n_mappings"],
        dict_tables["q2t_mappings"],
        dict_tables["qas"],
        dict_tables["embeddings"],
        dict_tables["tables"],
    )
    # Construct hypergraph
    HG = construct_hypergraph(args, tables, embeddings)
    

    

if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("")
