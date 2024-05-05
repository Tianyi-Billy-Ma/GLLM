from collections import defaultdict
from json import load
from math import e
import os
import os.path as osp
from matplotlib import table
import numpy as np

from arguments import parse_args
import torch

import copy
from torch_geometric.loader import DataLoader
from torch_scatter import scatter
from models.GNNs.allset import SetGNN, Encoder
from models.LLMs.models import load_retriever
from src.load_data import save_pickle, load_pickle
from src.helper import BipartiteData, hypergraph_collcate_fn
from src.contrastive import ContrastiveLoss, contrastive_loss_node
from src.augmentation import aug
from src.preprocess import add_special_token, generate_tokens_and_token_ids
from src.normalize_text import normalize


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
    hg_edge_index = np.array([[node_idx, 0] for node_idx in range(num_cols * num_rows)])
    col_edge_index[:, 1] += num_rows  # shift the index of column hyperedge index
    hg_edge_index[:, 1] += num_rows + num_cols  # shift the index of table node
    edge_index = np.concatenate(
        [row_edge_index, col_edge_index, hg_edge_index], axis=0
    ).T  # (2, c_num_edges)
    return edge_index, num_cols, num_rows


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
    node_passages,
    hyperedge_passages,
    table_passages,
    tokenizer,
    title_embeddings=None,
    summary_embeddings=None,
):
    hypergraph_list = []
    num_nodes = 0
    num_he = 0

    for t_idx, (key, val) in enumerate(tables.items()):
        edge_index, num_cols, num_rows = generate_edge_index(val)
        nidx = edge_index[0, :].max() + 1

        # passage_tokens = []
        xs_ids = []
        xt_ids = []
        for nid in range(num_nodes, num_nodes + nidx):
            passage = node_passages[f"node_{nid}"]
            passage = normalize(passage)

            tokens, token_ids = generate_tokens_and_token_ids(
                passage, tokenizer, args.LLMs_passage_max_token_length
            )

            # passage_tokens.append(tokens)
            xs_ids.append(token_ids)
        # x_s = tokenizer.convert_tokens_to_ids(emb)
        xs_ids = torch.LongTensor(xs_ids)

        assert num_cols * num_rows == xs_ids.shape[0], "Nodes mismatch"

        num_nodes += xs_ids.shape[0]

        edge_index = torch.LongTensor(edge_index)

        # The last eidx is the table node.
        # Note here we didn't add 1 cause the last eidx is for table node.
        eidx = edge_index[1, :].max()

        for eid in range(num_he, num_he + eidx):
            passage = hyperedge_passages[f"hyperedge_{eid}"]
            passage = normalize(passage)
            tokens, token_ids = generate_tokens_and_token_ids(
                passage, tokenizer, args.LLMs_passage_max_token_length
            )
            xt_ids.append(token_ids)

        xt_ids = torch.LongTensor(xt_ids)
        assert num_cols + num_rows == xt_ids.shape[0], "Hyperedges mismatch"

        num_he += num_cols + num_rows

        # Add Table Embedding to x_t
        table_passage = table_passages[f"table_{t_idx}"]
        tokens, token_ids = generate_tokens_and_token_ids(
            table_passage, tokenizer, args.LLms_passage_max_token_length
        )

        table_token_ids = torch.LongTensor(token_ids)
        xt_ids = torch.cat(tensors=[xt_ids, table_token_ids.unsqueeze(0)], dim=0)

        hg = BipartiteData(x_s=xs_ids, x_t=xt_ids, edge_index=edge_index)
        hg = reverse_bipartite_data(hg) if args.GNNs_reverse_HG else hg

        hg.num_nodes = hg.x_s.shape[0]
        hg.num_hyperedges = hg.x_t.shape[0]

        # The edge_index for generate global view of the hypergraphs
        # Here we consider the sub-hypergraph embeddings,
        # i.e., eidx = num_cols + num_rows in the origianl hypergraph
        # Or nidx = num_cols + num_rows in the reversed hypergraph
        # Because the last eidx is for table node.
        # And this can be view as a self-loop
        hg.edge_index_H = num_cols + num_rows
        hg.t_idx = t_idx

        hypergraph_list.append(hg)

    HG = DataLoader(
        hypergraph_list,
        batch_size=args.GNNs_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=hypergraph_collcate_fn,
    )
    print([hg.edge_index_H for hg in hypergraph_list][:64])
    return HG


def main(args):
    # Load data saved by pretrain_embedding.py
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")

    data = load_pickle(osp.join(pretrain_dir, "plaintext_data.pickle"))
    tables = data["tables"]
    passages = data["passages"]

    node_passages, hyperedge_passages = passages["nodes"], passages["hyperedges"]
    table_passages = passages["tables"]

    mappings = data["mappings"]
    # table_embeddings = torch.FloatTensor(table_embeddings).to(device)
    contrastive_loss = ContrastiveLoss(args.GNNs_tau)

    _, tokenizer = load_retriever(args.LLMs_pretrain_model)
    tokenizer = add_special_token(tokenizer)

    # Construct hypergraph
    HG = construct_hypergraph(
        args,
        tables,
        node_passages,
        hyperedge_passages,
        table_passages,
        tokenizer=tokenizer,
    )

    # model = SetGNN(args).to(device)
    model = Encoder(args).to(device)
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
            batch_sub = batch_sub.to(device)

            batch1 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)
            batch2 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)

            X_V1, X_E1, X_G1 = model.forward(batch1)
            X_V2, X_E2, X_G2 = model.forward(batch2)

            if args.GNNs_reverse_HG:
                batch_loss = contrastive_loss_node(X_V1, X_V2, args.GNNs_tau)
            else:
                batch_loss = contrastive_loss_node(X_E1, X_E2, args.GNNs_tau)
            # batch_loss_V = contrastive_loss_node(G_V1, G_V2, args.GNNs_tau)
            # batch_loss_H = contrastive_loss_node(G_E1, G_E2, args.GNNs_tau)

            loss_cl = batch_loss

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Batch ID: {batch_idx:03d}, "
                    f"Loss: {loss_cl:.4f}"
                )
            loss_cl.backward()
            optimizer.step()

    model.eval()
    emb_V, emb_E, emb_G = [], [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(HG):
            batch.norm = torch.ones((batch.x.shape[0], 1))
            batch = batch.to(device)
            X_V, X_E, X_G = model.forward(batch)
            emb_V.append(X_V)
            emb_E.append(X_E)
            emb_G.append(X_G)

    save_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    os.makedirs(save_dir, exist_ok=True)

    emb_V = torch.cat(emb_V, dim=0).cpu().detach().numpy()
    emb_E = torch.cat(emb_E, dim=0).cpu().detach().numpy()
    emb_G = torch.cat(emb_G, dim=0).cpu().detach().numpy()

    emb_V, emb_E = emb_E, emb_V if args.GNNs_reverse_HG else emb_V, emb_E

    assert emb_V.shape[0] == len(nids), "Node embeddings mismatch"
    assert emb_E.shape[0] == len(eids), "Hyperedge embeddings mismatch"

    save_pickle(
        {
            "nodes": {"ids": nids, "embeddings": emb_V},
            "hyperedges": {"ids": nids, "embeddings": emb_E},
            "tables": {"ids": tids, "embeddings": emb_G},
        },
        osp.join(save_dir, "embeddings.pickle"),
    )

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
    args.GNNs_reverse_HG = True
    args.GNNs_aug = "mask"
    args.GNNs_tau = 0.07
    main(args)
    # generate_table_embeddings(args)

    # x_s = torch.randn(10, 5)
    # x_t = torch.randn(6, 5)
    # edge_index = torch.LongTensor(
    #     [[1, 2, 3, 5, 5, 1, 2, 3, 4, 9, 2, 1], [1, 2, 3, 5, 5, 1, 2, 3, 4, 2, 2, 1]]
    # )
    # V_mapping = torch.LongTensor(
    #     [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
    # )
    # E_mapping = torch.LongTensor([[0, 1, 2, 3, 4, 5], [0, 0, 0, 0, 0, 0]])

    # hg = BipartiteData(x_s=x_s, x_t=x_t, edge_index=edge_index)
    # hg.V_mapping = V_mapping
    # hg.E_mapping = E_mapping

    # data_list = [hg, hg, hg]
    # loader = DataLoader(data_list, batch_size=2)
    # batch = next(iter(loader))

    print("")
