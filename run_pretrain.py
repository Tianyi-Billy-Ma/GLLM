from collections import defaultdict
import os.path as osp
import os
import numpy as np
import torch
import copy
from arguments import parse_args
from experiment_retrieve_table import load_retriever
from src.contrastive import ContrastiveLoss, contrastive_loss_node
from src.load_data import load_data, load_pickle, save_pickle
from src.preprocess import (
    generate_unique_tables,
    generate_hyperedges_plaintext_from_tables,
    generate_node_plaintext_from_tables,
    generate_plaintext_from_table,
    construct_hypergraph,
)
from src.preprocess import (
    START_ROW_TAG,
    END_ROW_TAG,
    START_COL_TAG,
    END_COL_TAG,
    START_NODE_TAG,
    END_NODE_TAG,
    START_TITLE_TAG,
    END_TITLE_TAG,
    START_SUMMARY_TAG,
    END_SUMMARY_TAG,
    NULL_TAG,
)

from src.preprocess import add_special_token
from models.GNNs.allset import Encoder, SetGNN
from src.augmentation import aug
from torch_scatter import scatter

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128mb"
# device = torch.device("cpu")


def main(args):
    # ************************* Load Data *************************
    data = load_data(args)

    # ************************* Preprocess Data *************************

    unique_tables, qas, tname2tid, qid2tid, qid2qname = generate_unique_tables(data)

    tables = {}

    for t_idx, (key, table) in enumerate(unique_tables.items()):
        raw_str = load_pickle(
            osp.join(
                args.raw_data_dir, args.dname, "summary", f"summary_{t_idx}.pickle"
            )
        )
        raw_split = raw_str.split("\n")
        assert len(raw_split) == 4
        table["title"] = raw_split[1][14:-2]
        table["summary"] = raw_split[2][16:-1]
        tables[tname2tid[table["name"]]] = table

    dict_nodes_plaintext = generate_node_plaintext_from_tables(args, tables)
    dict_hyperedges_plaintext = generate_hyperedges_plaintext_from_tables(args, tables)

    nodes, hyperedges = {}, {}

    nid2tid, eid2tid = {}, {}

    for tid, cell in dict_nodes_plaintext.items():
        idx = len(nid2tid.keys())
        for i, value in zip(range(idx, idx + len(cell)), cell):
            nid = f"node_{i}"
            nid2tid[nid] = tid
            if not args.LLMs_pretrain_include_tags:
                nodes[nid] = value.replace(START_NODE_TAG, "").replace(END_NODE_TAG, "")
            nodes[nid] = value
    assert sum([len(cell) for _, cell in dict_nodes_plaintext.items()]) == len(
        nodes.keys()
    ), "Number of nodes mismatch"
    assert sum([len(cell) for _, cell in dict_nodes_plaintext.items()]) == len(
        nid2tid.keys()
    ), "Number of nodes mismatch"
    # t2n_mappings.extend([key] * len(value))
    for tid, cell in dict_hyperedges_plaintext.items():
        idx = len(eid2tid.keys())
        for i, value in zip(range(idx, idx + len(cell)), cell):
            if value.startswith(START_ROW_TAG) and value.endswith(END_ROW_TAG):
                eid2tid[f"row_{i}"] = tid
                if args.LLMs_pretrain_include_tags:
                    hyperedges[f"row_{i}"] = value[5:-5]
                else:
                    hyperedges[f"row_{i}"] = value
            elif value.startswith(START_COL_TAG) and value.endswith(END_COL_TAG):
                eid2tid[f"col_{i}"] = tid
                if args.LLMs_pretrain_include_tags:
                    hyperedges[f"col_{i}"] = value[5:-5]
                else:
                    hyperedges[f"col_{i}"] = value
            else:
                raise ValueError("Invalid hyperedge")
    assert sum([len(cell) for _, cell in dict_hyperedges_plaintext.items()]) == len(
        hyperedges.keys()
    ), "Number of hyperedges mismatch"
    assert sum([len(cell) for _, cell in dict_hyperedges_plaintext.items()]) == len(
        eid2tid.keys()
    ), "Number of hyperedges mismatch"

    titles, summaries = {}, {}
    titleid2tid, summaryid2tid = {}, {}
    for idx, (tid, table) in enumerate(tables.items()):
        titles[f"title_{idx}"] = table["title"]
        titleid2tid[f"title_{idx}"] = tid
        summaries[f"summary_{idx}"] = table["summary"]
        summaryid2tid[f"summary_{idx}"] = tid

    passage_dict = {
        "nodes": nodes,
        "hyperedges": hyperedges,
        "titles": titles,
        "summaries": summaries,
        "tables": {
            tid: generate_plaintext_from_table(table, args)
            for tid, table in tables.items()
        },
    }

    tid2nid, tid2eid = defaultdict(list), defaultdict(list)

    for nid, tid in nid2tid.items():
        tid2nid[tid].append(nid)
    for eid, tid in eid2tid.items():
        tid2eid[tid].append(eid)

    # ************************* Generate Hypergraph Embeddings *************************

    model, tokenizer = load_retriever(args.LLMs_pretrain_model)

    if args.LLMs_pretrain_include_tags:
        tokenizer = add_special_token(tokenizer)
        model.resize_token_embeddings(len(tokenizer))

    args.LLMs_pretrain_vocab_size = len(tokenizer)
    args.pad_token_id = tokenizer.pad_token_id

    HG = construct_hypergraph(
        args, tables, passage_dict, model, tokenizer, [tid2nid, tid2eid]
    )

    save_pickle(HG, osp.join(args.processed_data_dir, args.dname, "GNNs", "HG.pickle"))

    # HG = load_pickle(osp.join(args.processed_data_dir, args.dname, "GNNs", "HG.pickle"))

    # ************************* Model Training *************************

    # model = Encoder(args).to(device)
    model = SetGNN(args).to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.GNNs_lr, weight_decay=args.GNNs_wd
    )

    contrastive_loss_node_fn = ContrastiveLoss(args.GNNs_tau)
    # contrastive_loss_node_fn = contrastive_loss_node

    for epoch in range(args.GNNs_epochs):
        model.train()

        for batch_idx, batch in enumerate(HG):
            batch_sub = copy.deepcopy(batch)

            batch_sub.norm = torch.ones((batch_sub.x_s.shape[0], 1))
            batch1 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)
            batch2 = aug(batch_sub, args.GNNs_aug, args.GNNs_aug_ratio).to(device)

            emb_V1, emb_E1 = model(batch1)
            emb_V2, emb_E2 = model(batch2)

            batch_loss1 = contrastive_loss_node_fn(emb_V1, emb_V2, args.GNNs_tau)
            batch_loss2 = contrastive_loss_node_fn(emb_E1, emb_E2, args.GNNs_tau)

            loss_cl = (batch_loss1 + batch_loss2) * 0.5

            if epoch % args.display_step == 0 and args.display_step > 0:
                print(
                    f"Epoch: {epoch:02d}, "
                    f"Batch ID: {batch_idx:03d}, "
                    f"Loss: {loss_cl:.4f}"
                )
            loss_cl.backward()
            optimizer.step()

    # torch.save(
    #     model.embed_layer.state_dict(), osp.join(args.GNNs_dir, "checkpoint.pth")
    # )

    model.eval()

    emb_V, emb_E = [], []
    with torch.no_grad():
        for batch_idx, batch in enumerate(HG):
            batch = batch.to(device)
            emb_V_batch, emb_E_batch = model(batch)
            emb_V.append(emb_V_batch)
            emb_E.append(emb_E_batch)

    emb_V = torch.cat(emb_V, dim=0).cpu().detach()
    emb_E = torch.cat(emb_E, dim=0).cpu().detach()

    # emb_V, emb_E = emb_E, emb_V if args.GNNs_reverse_HG else emb_V, emb_E

    edge_index = []
    for tid, nids in tid2nid.items():
        for nid in nids:
            edge_index.append([int(nid.split("_")[-1]), int(tid.split("_")[-1])])
    edge_index = torch.LongTensor(edge_index).T

    emb_G_V = scatter(emb_V, edge_index[1, :], dim=0, reduce="mean").numpy()

    save_dir = osp.join(args.root_dir, "output", "GNNs", "table_from_node")
    os.makedirs(save_dir, exist_ok=True)
    save_pickle(
        {"ids": list(tid2nid.keys()), "embedding": emb_G_V},
        osp.join(save_dir, "tables.pickle"),
    )

    edge_index = []
    for tid, eids in tid2eid.items():
        for eid in eids:
            edge_index.append([int(eid.split("_")[-1]), int(tid.split("_")[-1])])

    edge_index = torch.LongTensor(edge_index).T
    emb_G_E = scatter(emb_E, edge_index[1, :], dim=0, reduce="mean").numpy()
    save_dir = osp.join(args.root_dir, "output", "GNNs", "table_from_he")
    os.makedirs(save_dir, exist_ok=True)
    save_pickle(
        {"ids": list(tid2nid.keys()), "embedding": emb_G_E},
        osp.join(save_dir, f"tables.pickle"),
    )


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_pretrain_include_tags = False
    args.GNNs_reverse_HG = False
    args.GNNs_aug = "edge"
    main(args)
