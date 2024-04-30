from collections import defaultdict
from json import load
from math import e
import os
import os.path as osp
import numpy as np
from src import (
    load_pickle,
    save_json,
    aug,
    contrastive_loss_node,
    BipartiteData,
)
from arguments import parse_args
import torch

import copy
from torch_geometric.data import DataLoader
from torch_scatter import scatter
from models.GNNs.allset import SetGNN
from src.load_data import save_pickle

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
    hypergraph_edge_index = np.array(
        [[node_idx, num_rows + num_cols] for node_idx in range(num_cols * num_rows)]
    )
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


def construct_hypergraph(
    args,
    tables,
    node_embeddings,
    hyperedge_embeddings,
    title_embeddings,
    summary_embeddings,
):
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
        eidx = edge_index[1, :].max() + 1  # The last eidx is the table node.
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


def main(args):
    # Load data saved by pretrain_embedding.py
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    table_info_path = osp.join(pretrain_dir, "plaintext_tables.pickle")
    mapping_path = osp.join(pretrain_dir, "mapping.pickle")
    node_embedding_path = osp.join(pretrain_dir, "embeddings_node.pickle")
    hyperedge_embedding_path = osp.join(pretrain_dir, "embeddings_hyperedge.pickle")
    nodes = load_pickle(node_embedding_path)
    hyperedges = load_pickle(hyperedge_embedding_path)

    nids, node_embeddings = nodes["ids"], nodes["embeddings"]
    eids, hyperedge_embeddings = hyperedges["ids"], hyperedges["embeddings"]

    input_data = load_pickle(table_info_path)
    tables = input_data["tables"]

    mappings = load_pickle(mapping_path)
    nid2tid, eid2tid = mappings["nid2tid"], mappings["eid2tid"]

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
    save_node_path = osp.join(save_dir, "embeddings_node.pickle")
    save_hyperedge_path = osp.join(save_dir, "embeddings_hyperedge.pickle")

    emb_V = torch.cat(emb_V, dim=0).cpu().detach().numpy()
    emb_E = torch.cat(emb_E, dim=0).cpu().detach().numpy()
    if args.GNNs_reverse_HG:
        emb_V, emb_E = emb_E, emb_V

    assert emb_V.shape[0] == len(nids), "Node embeddings mismatch"
    assert emb_E.shape[0] == len(eids), "Hyperedge embeddings mismatch"

    save_pickle({"ids": nids, "embeddings": emb_V}, save_node_path)
    save_pickle({"ids": eids, "embeddings": emb_E}, save_hyperedge_path)

    tid2eids = defaultdict(list)

    for eid, tid in eid2tid.items():
        tid2eids[tid].append(eid)

    save_pickle(tid2eids, osp.join(save_dir, "tid2eids.pickle"))

    print("")


def generate_table_embeddings(args):
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    summary_embedding_path = osp.join(pretrain_dir, "embeddings_summary.pickle")
    title_emnbedding_path = osp.join(pretrain_dir, "embeddings_title.pickle")
    mapping_path = osp.join(pretrain_dir, "mapping.pickle")

    summaries = load_pickle(summary_embedding_path)
    titles = load_pickle(title_emnbedding_path)

    title_ids, title_embeddings = (
        titles["ids"],
        titles["embeddings"],
    )
    summary_ids, summary_embeddings = (
        summaries["ids"],
        summaries["embeddings"],
    )

    mappings = load_pickle(mapping_path)
    nid2tid, eid2tid = mappings["nid2tid"], mappings["eid2tid"]

    node_embedding_path = osp.join(GNNs_dir, "embeddings_node.pickle")
    hyperedge_embedding_path = osp.join(GNNs_dir, "embeddings_hyperedge.pickle")

    nodes = load_pickle(node_embedding_path)
    hyperedges = load_pickle(hyperedge_embedding_path)

    nids, node_embeddings = nodes["ids"], nodes["embeddings"]
    eids, hyperedge_embeddings = hyperedges["ids"], hyperedges["embeddings"]

    hyperedge_embeddings = torch.FloatTensor(hyperedge_embeddings)
    edge_index = np.stack(
        [
            np.arange(hyperedge_embeddings.shape[0]),
            [int(tid.split("_")[-1]) for tid in eid2tid.values()],
        ],
        0,
    )

    edge_index = torch.LongTensor(edge_index)

    table_embeddings = scatter(
        hyperedge_embeddings[edge_index[0]],
        edge_index[1],
        dim=0,
        reduce="mean",
    )
    table_embeddings = table_embeddings.numpy()
    
    return True


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    generate_table_embeddings(args)
    print("")
