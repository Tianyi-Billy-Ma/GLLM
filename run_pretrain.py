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
from src.helper import seed_everything
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
    seed_everything(args.seed)
    data = load_data(args)

    # ************************* Preprocess Data *************************

    if args.reprocess_dataset:
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
        dict_hyperedges_plaintext = generate_hyperedges_plaintext_from_tables(
            args, tables
        )

        nodes, hyperedges = {}, {}

        nid2tid, eid2tid = {}, {}

        for tid, cell in dict_nodes_plaintext.items():
            idx = len(nid2tid.keys())
            for i, value in zip(range(idx, idx + len(cell)), cell):
                nid = f"node_{i}"
                nid2tid[nid] = tid
                if not args.LLMs_pretrain_include_tags:
                    value = value.replace(START_NODE_TAG, "").replace(END_NODE_TAG, "")
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
                    eid = f"row_{i}"
                elif value.startswith(START_COL_TAG) and value.endswith(END_COL_TAG):
                    eid = f"col_{i}"
                else:
                    raise ValueError("Invalid hyperedge")
                eid2tid[eid] = tid
                if not args.LLMs_pretrain_include_tags:
                    value = value.replace(START_ROW_TAG, "").replace(END_ROW_TAG, "")
                    value = value.replace(START_COL_TAG, "").replace(END_COL_TAG, "")
                hyperedges[eid] = value
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

        tid2nid = {
            tid: [nid for nid, _tid in nid2tid.items() if _tid == tid]
            for tid in set(nid2tid.values())
        }
        tid2eid = {
            tid: [eid for eid, _tid in eid2tid.items() if _tid == tid]
            for tid in set(eid2tid.values())
        }

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

        save_pickle(
            HG, osp.join(args.processed_data_dir, args.dname, "GNNs", "HG.pickle")
        )
    else:
        HG = load_pickle(
            osp.join(args.processed_data_dir, args.dname, "GNNs", "HG.pickle")
        )

    # ************************* Model Training *************************

    # model = Encoder(args).to(device)
    if args.GNNs_pretrain_emb:
        model = Encoder(args)
    else:
        model = SetGNN(args)
    model = model.to(device)

    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.GNNs_lr, weight_decay=args.GNNs_wd
    )

    contrastive_loss_node_fn = ContrastiveLoss(args.GNNs_tau)
    # contrastive_loss_node_fn = contrastive_loss_node

    for epoch in range(args.GNNs_epochs):
        model.train()
        # model.eval()
        for batch_idx, batch in enumerate(HG):
            batch_sub = copy.deepcopy(batch)

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

    edge_index = []
    for tid, eids in tid2eid.items():
        for eid in eids:
            edge_index.append([int(eid.split("_")[-1]), int(tid.split("_")[-1])])

    edge_index = torch.LongTensor(edge_index).T
    emb_G_E = scatter(emb_E, edge_index[1, :], dim=0, reduce="mean").numpy()

    # ************************* Save Embeddings *************************

    id_mappings = [list(tid2nid.keys()), list(tid2eid.keys())]
    embs = [emb_G_V, emb_G_E]
    save_dirs = [
        osp.join(args.root_dir, "output", "GNNs", "table_from_node}"),
        osp.join(args.root_dir, "output", "GNNs", f"table_from_he"),
    ]

    for ids, save_dir, emb in zip(id_mappings, save_dirs, embs):
        os.makedirs(save_dir, exist_ok=True)
        save_pickle(
            {"ids": ids, "embedding": emb},
            osp.join(save_dir, "tables.pickle"),
        )

    # ************************* Evaluation *************************
    from experiment_retrieve_table import Retriever
    from src.preprocess import process_questions

    questions = process_questions(args)

    data = load_data(args)
    _, _, _, qid2tid, _ = generate_unique_tables(data)

    # save_dirs = [
    #     "/media/mtybilly/My Passport1/Program/GLLM/output/LLMs/dict_summary_title_contriever-msmarco"
    # ]

    for save_dir in save_dirs:
        args.LLMs_retriever_input_path = save_dir
        retriever = Retriever(args)
        retriever.setup_retriever()

        predictions = {}

        qids = list(questions.keys())
        passages = [questions[qid] for qid in qids]
        documents = retriever.search_table(passages, 10)
        for qid, document in zip(qids, documents):
            predictions[qid] = document

        num_questions = len(questions.keys())
        res = {1: 0, 3: 0, 5: 0, 10: 0}

        for qid, prediction in predictions.items():
            gt = qid2tid[qid].split("_")[-1]
            if isinstance(prediction, list):
                prefix_res = [name.split("_")[-1] for name in prediction]
            for k in res.keys():
                if gt in prefix_res[:k]:
                    res[k] += 1

        res = {k: round(100 * v / num_questions, 2) for k, v in res.items()}

        print(f"{res}")


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_pretrain_include_tags = False
    args.GNNs_reverse_HG = False
    args.reprocess_dataset = False
    args.GNNs_aug = "edge"
    args.LLMs_save_or_load_index = False
    args.reprocess_dataset = True
    main(args)
