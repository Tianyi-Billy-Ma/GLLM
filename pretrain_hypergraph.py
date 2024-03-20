from src import (
    load_pickle,
    save_pickle,
    load_data,
    aug,
    contrastive_loss_node,
    BipartiteData,
)
import os, os.path as osp
from arguments import parse_args
import torch
from tqdm import tqdm
import numpy as np
import copy
from torch_geometric.data import Data, DataLoader, Batch
import torch.nn.functional as F
from models.GNNs.allset import SetGNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_edge_index(table):
    num_cols, num_rows = len(table["header"]), len(table["rows"])
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
    edge_index = np.concatenate(
        [row_edge_index, col_edge_index], axis=0
    ).T  # (2, c_num_edges)
    return edge_index


def construct_hypergraph(args, tables, embeddings):
    hypergraph_list = []
    num_nodes = 0
    for t_idx, (key, val) in enumerate(tables.items()):
        edge_index = generate_edge_index(val)
        x_s = torch.FloatTensor(
            embeddings[num_nodes : num_nodes + edge_index[0, :].max() + 1, :]
        )
        num_nodes += x_s.shape[0]
        edge_index = torch.LongTensor(edge_index)
        # TODO:  support initialization for hyperedge attribute features.
        x_t = torch.zeros((edge_index[1].max() + 1, x_s.shape[1]))
        hg = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
        hg.x = hg.x_s
        hypergraph_list.append(hg)
    assert (
        embeddings == torch.cat([hg.x_s for hg in hypergraph_list], dim=0).numpy()
    ).all(), "Nodes mismatch"
    HG = DataLoader(
        hypergraph_list, batch_size=args.GNNs_batch_size, shuffle=False, drop_last=False
    )
    return HG


# region
# def construct_hypergraph(args, tables, embs):
#     edge_index = None
#     hypergraph_list = []
#     for t_idx, (key, val) in enumerate(tables.items()):
#         num_cols, num_rows = len(val["header"]), len(val["rows"])

#         row_edge_index, col_edge_index = [], []
#         for he_idx in range(num_rows):
#             row_edge_index.extend(
#                 [[he_idx * num_cols + col, he_idx] for col in range(num_cols)]
#             )
#         for he_idx in range(num_cols):
#             col_edge_index.extend(
#                 [[row * num_cols + he_idx, he_idx] for row in range(num_rows)]
#             )
#         row_edge_index = np.array(row_edge_index)
#         col_edge_index = np.array(col_edge_index)
#         col_edge_index[:, 1] += num_rows  # shift the index of column hyperedge index
#         c_edge_index = np.concatenate(
#             [row_edge_index, col_edge_index], axis=0
#         ).T  # (2, c_num_edges)

#         assert (
#             c_edge_index[0, :].max() + 1 == num_cols * num_rows
#         ), "Nodes are not properly indexed"
#         assert (
#             c_edge_index[1, :].max() + 1 == num_cols + num_rows
#         ), "Hyperedges are not properly indexed"

#         if edge_index is None:
#             c_edge_index[1, :] += embs.shape[0] # shift the index of hyperedges
#             edge_index = c_edge_index
#         else:
#             c_edge_index[1, :] += edge_index[1].max() + 1  # shift the index of hyperedges
#             c_edge_index[0, :] += edge_index[0].max() + 1
#             edge_index = np.concatenate(
#                 [edge_index, c_edge_index], axis=1
#             )  # (2, num_edges)
#         emb_idx_min, emb_idx_max = c_edge_index[0, :].min(), c_edge_index[0, :].max() + 1
#         x = torch.FloatTensor(embs[emb_idx_min:emb_idx_max, :])
#         c_HG = Data(x=x, edge_index=torch.LongTensor(c_edge_index).contiguous())
#         hypergraph_list.append(c_HG)
#         if t_idx == 31:
#             print("")
#     assert embs.shape[0] == sum([hg.x.shape[0] for hg in hypergraph_list]), "Nodes mismatch"
#     assert edge_index.shape[1] ==  sum([hg.edge_index.shape[1] for hg in hypergraph_list]), "Edges mismatch"
#     assert (np.arange(embs.shape[0]) == np.unique(edge_index[0], return_counts=True)[0]).all(), "Nodes are not properly indexed"
#     assert (np.unique(edge_index[0], return_counts=True)[1] == 2).all(), "Nodes are not properly indexed"
#     # HG = Data(x=torch.FloatTensor(embs), edge_index=torch.LongTensor(edge_index))
#     batch_sampler = HypergraphBatchSampler(len(hypergraph_list), args.GNNs_batch_size)
#     # HG = DataLoader(hypergraph_list, batch_size=32, shuffle=False)
#     HG = DataLoader(hypergraph_list, batch_sampler=batch_sampler, shuffle=False)
#     return HG
# endregion


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

    args.GNNs_num_features = embeddings.shape[1]

    model = SetGNN(args).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.GNNs_lr, weight_decay=args.GNNs_wd
    )

    for epoch in range(args.GNNs_epochs):

        model.train()
        optimizer.zero_grad()

        loss_cl = 0.0

        for batch_idx, batch in enumerate(HG):
            batch_sub = copy.deepcopy(batch)
            batch_sub.norm = torch.ones((batch_sub.x.shape[0], 1))

            batch1 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)
            batch2 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)

            X_V1, X_H1 = model.forward(batch1)
            X_V2, X_H2 = model.forward(batch2)

            batch_loss_V = contrastive_loss_node(X_V1, X_V2, args.GNNs_tau)
            batch_loss_H = contrastive_loss_node(X_H1, X_H2, args.GNNs_tau)

            loss_cl = (batch_loss_V + batch_loss_H) * 0.5

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Batch ID: {batch_idx:03d}, "
                    f"Loss: {loss_cl:.4f}"
                )
            loss_cl.backward()
            optimizer.step()

    model.eval()
    emb_V, emb_H = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(HG):
            batch.norm = torch.ones((batch.x.shape[0], 1))
            batch = batch.to(device)
            X_V, X_H = model.forward(batch)
            emb_V.append(X_V)
            emb_H.append(X_H)
    save_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    save_path = osp.join(save_dir, f"{args.GNNs_model_name}.pickle")
    save_pickle(
        {
            "Nodes": torch.cat(emb_V, dim=0).cpu().detach().numpy(),
            "Hyperedges": torch.cat(emb_H, dim=0).cpu().detach().numpy(),
        },
        save_path,
    )
    print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("")
