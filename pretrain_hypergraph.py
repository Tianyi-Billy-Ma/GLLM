from json import load
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


def reverse_edge_index(edge_index):
    edge_index = edge_index.numpy()
    edge_index = np.stack([edge_index[1, :], edge_index[0, :]], axis=0)
    return edge_index


def reverse_bipartite_data(hg):
    x_s, x_t = hg.x_t, hg.x_s  # reverse here
    edge_index = reverse_edge_index(hg.edge_index)
    assert edge_index[0, :].max() + 1 == x_s.shape[0], "Nodes mismatch"
    assert edge_index[1, :].max() + 1 == x_t.shape[0], "Hyperedges mismatch"
    edge_index = torch.LongTensor(edge_index)
    return BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)


def construct_hypergraph(args, tables, node_embeddings, hyperedge_embeddings):
    hypergraph_list = []
    num_nodes = 0
    num_he = 0
    for t_idx, (key, val) in enumerate(tables.items()):
        edge_index = generate_edge_index(val)
        nidx = edge_index[0, :].max() + 1
        x_s = torch.FloatTensor(node_embeddings[num_nodes : num_nodes + nidx])
        num_nodes += x_s.shape[0]
        edge_index = torch.LongTensor(edge_index)
        # TODO:  support initialization for hyperedge attribute features.
        eidx = edge_index[1, :].max() + 1
        x_t = torch.FloatTensor(hyperedge_embeddings[num_he : num_he + eidx])
        num_he += x_t.shape[0]
        hg = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
        hg = reverse_bipartite_data(hg) if args.GNNs_reverse_HG else hg
        hg.x = hg.x_s
        hypergraph_list.append(hg)
    if not args.GNNs_reverse_HG:
        assert (
            node_embeddings
            == torch.cat([hg.x_s for hg in hypergraph_list], dim=0).numpy()
        ).all(), "Nodes mismatch"
        assert (
            hyperedge_embeddings
            == torch.cat([hg.x_t for hg in hypergraph_list], dim=0).numpy()
        ).all(), "Hyperedges mismatch"
    else:
        assert (
            node_embeddings
            == torch.cat([hg.x_t for hg in hypergraph_list], dim=0).numpy()
        ).all(), "Nodes mismatch"
        assert (
            hyperedge_embeddings
            == torch.cat([hg.x_s for hg in hypergraph_list], dim=0).numpy()
        ).all(), "Hyperedges mismatch"
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
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    table_info_path = osp.join(pretrain_dir, "info.pickle")
    node_embedding_path = osp.join(pretrain_dir, "embeddings_node.pickle")
    hyperedge_embedding_path = osp.join(pretrain_dir, "embeddings_hyperedge.pickle")
    node_embeddings = load_pickle(node_embedding_path)
    hyperedge_embeddings = load_pickle(hyperedge_embedding_path)
    tables = load_pickle(table_info_path)["tables"]
    # Construct hypergraph
    HG = construct_hypergraph(args, tables, node_embeddings, hyperedge_embeddings)

    assert (
        node_embeddings.shape[1] == hyperedge_embeddings.shape[1]
    ), "Node and hyperedge embeddings have different dimensions"
    args.GNNs_num_features = node_embeddings.shape[1]

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
    emb_V, emb_E = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(HG):
            batch.norm = torch.ones((batch.x.shape[0], 1))
            batch = batch.to(device)
            X_V, X_E = model.forward(batch)
            emb_V.append(X_V)
            emb_E.append(X_E)
    save_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_node_path = osp.join(save_dir, f"embeddings_node.pickle")
    save_hyperedge_path = osp.join(save_dir, f"embeddings_hyperedge.pickle")
    emb_V = torch.cat(emb_V, dim=0).cpu().detach().numpy()
    emb_E = torch.cat(emb_E, dim=0).cpu().detach().numpy()
    if args.GNNs_reverse_HG:
        emb_V, emb_E = emb_E, emb_V
    save_pickle(emb_V, save_node_path)
    save_pickle(emb_E, save_hyperedge_path)
    print("")


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("")
