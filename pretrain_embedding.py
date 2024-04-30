from imp import load_compiled
from math import trunc
from numpy import add, save
from tqdm import tqdm
from pyparsing import col
from sentence_transformers import SentenceTransformer
import ast
import os, os.path as osp
from sympy import hyper
import torch
import json
from src import (
    load_data,
    load_pickle,
    save_pickle,
    generate_node_plaintext_within_tables,
    generate_hyperedges_plaintext_from_tables,
    normalize,
)
from src.preprocess import add_special_token
from arguments import parse_args
from models.LLMs.models import load_retriever
import logging

from src.preprocess import add_special_token

logger = logging.getLogger(__name__)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_unique_tables(dataset):
    qas = {}
    tname2tid, qid2tid, qid2qname = {}, {}, {}
    tables = {}
    for id, row in enumerate(dataset):
        table = row["table"]
        tname = row["table"]["name"]
        row_id, q, a = row["id"], row["question"], row["answers"]
        if tname in tname2tid.keys():
            tid = tname2tid[tname]
            assert len(tables[tid]["rows"]) == len(
                table["rows"]
            ), "Table with same name has different number of rows"
            assert (
                tables[tid]["header"] == table["header"]
            ), "Table with same name has different header"
        else:
            tid = f"table_{len(tables.keys())}"
            tname2tid[tname] = tid
            tables[tid] = table
        qid = f"question_{id}"
        qas[qid] = {
            "question": q,
            "answer": a,
        }
        qid2tid[qid] = tid
        qid2qname[qid] = row_id
    return tables, qas, tname2tid, qid2tid, qid2qname


def generate_embeddings(args, passages, model, tokenizer):
    model = model.to(device)
    model.eval()
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    total = 0
    with torch.no_grad():
        for idx, (passage_id, passage) in tqdm(
            enumerate(passages.items()), total=len(passages), desc="Encoding passages"
        ):
            batch_ids.append(passage_id)
            text = passage

            if args.LLMs_lowercase:
                text = text.lower()
            if args.LLMs_normalize_text:
                text = normalize(text)
            batch_text.append(text)

            if (
                len(batch_text) == args.LLMs_pretrain_batch_size
                or idx == len(passages) - 1
            ):
                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.LLMs_passage_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch)
                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)
                # logger.info(f"Encoded passages {total}/{len(passages)}")
                batch_text, batch_ids = [], []
    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def main(args):
    # unique_ts: unique tables in the dataset.  unique_ts[table_name] = table
    # q2t_mappings: mapping from question id to table name. q2t_mappings[id] = table_name
    # qas: question-answer pairs. qas[id] = {"question": q, "answers": a}
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    load_path = osp.join(pretrain_dir, "plaintext_tables.pickle")
    mapping_path = osp.join(pretrain_dir, "mapping.pickle")
    node_plaintext_path = osp.join(pretrain_dir, "plaintext_node.pickle")
    hyperedge_plaintext_path = osp.join(pretrain_dir, "plaintext_hyperedge.pickle")
    title_plaintext_path = osp.join(pretrain_dir, "plaintext_title.pickle")
    summary_plaintext_path = osp.join(pretrain_dir, "plaintext_summary.pickle")
    if (
        osp.exists(load_path)
        and osp.exists(mapping_path)
        and osp.exists(node_plaintext_path)
        and osp.exists(hyperedge_plaintext_path)
        and osp.exists(title_plaintext_path)
        and osp.exists(summary_plaintext_path)
        and not args.LLMs_pretrain_reload
    ):
        data = load_pickle(load_path)
        tables, qas = (
            data["tables"],
            data["qas"],
        )
        mappings = load_pickle(mapping_path)
        tname2tid, qid2tid, qid2qname, nid2tid, eid2tid, titleid2tid, summaryid2tid = (
            mappings["tname2tid"],
            mappings["qid2tid"],
            mappings["qid2qname"],
            mappings["nid2tid"],
            mappings["eid2tid"],
            mappings["titleid2tid"],
            mappings["summaryid2tid"],
        )
        nodes = load_pickle(node_plaintext_path)
        hyperedges = load_pickle(hyperedge_plaintext_path)
        titles = load_pickle(title_plaintext_path)
        summaries = load_pickle(summary_plaintext_path)
    else:
        dataset = load_data(args)
        unique_tables, qas, tname2tid, qid2tid, qid2qname = generate_unique_tables(
            dataset
        )
        tables = {}
        for idx, (key, table) in enumerate(unique_tables.items()):
            raw_str = load_pickle(
                osp.join(
                    args.raw_data_dir, args.dname, "summary", f"summary_{idx}.pickle"
                )
            )
            raw_split = raw_str.split("\n")
            assert len(raw_split) == 4
            if args.LLMs_pretrain_include_tags:
                table["title"] = "[TITLE] " + raw_split[1][14:-2] + " [/TITLE]"
                table["summary"] = "[SUMMARY] " + raw_split[2][16:-1] + " [/SUMMARY]"
            else:
                table["title"] = raw_split[1][14:-2]
                table["summary"] = raw_split[2][16:-1]
            tables[tname2tid[table["name"]]] = table

        dict_nodes_plaintext = generate_node_plaintext_within_tables(
            tables, args.LLMs_lowercase, args.LLMs_normalize_text
        )
        dict_hyperedges_plaintext = generate_hyperedges_plaintext_from_tables(
            tables, args.LLMs_lowercase, args.LLMs_normalize_text
        )
        nodes, hyperedges = {}, {}
        nid2tid, eid2tid = {}, {}
        for tid, cell in dict_nodes_plaintext.items():
            idx = len(nid2tid.keys())
            for i, value in zip(range(idx, idx + len(cell)), cell):
                nid = f"node_{i}"
                nid2tid[nid] = tid
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
                if value.startswith("[ROW]") and value.endswith("[/ROW]"):
                    eid2tid[f"row_{i}"] = tid
                    if args.LLMs_pretrain_include_tags:
                        hyperedges[f"row_{i}"] = value[5:-5]
                    else:
                        hyperedges[f"row_{i}"] = value
                elif value.startswith("[COL]") and value.endswith("[/COL]"):
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

        save_pickle(
            {
                "tables": tables,
                "qas": qas,
            },
            load_path,
        )
        save_pickle(
            {
                "tname2tid": tname2tid,
                "qid2tid": qid2tid,
                "qid2qname": qid2qname,
                "nid2tid": nid2tid,
                "eid2tid": eid2tid,
                "titleid2tid": titleid2tid,
                "summaryid2tid": summaryid2tid,
            },
            mapping_path,
        )
        save_pickle(nodes, node_plaintext_path)
        save_pickle(hyperedges, hyperedge_plaintext_path)
        save_pickle(titles, title_plaintext_path)
        save_pickle(summaries, summary_plaintext_path)
        # passages = nodes + hyperedges
        # save_pickle(passages, passage_plaintext_path)
    # plain_text = nodes + hyperedges
    # assert len(plain_text) == len(nid2tid.keys()), "Number of plain text mismatch"
    model, tokenizer = load_retriever(args.LLMs_pretrain_model)
    tokenizer = add_special_token(tokenizer)

    model.resize_token_embeddings(len(tokenizer))
    model = model.to(device)

    node_path = osp.join(pretrain_dir, "embeddings_node.pickle")
    hyperedge_path = osp.join(pretrain_dir, "embeddings_hyperedge.pickle")
    title_path = osp.join(pretrain_dir, "embeddings_title.pickle")
    summary_path = osp.join(pretrain_dir, "embeddings_summary.pickle")

    plaintexts = [nodes, hyperedges, titles, summaries]
    save_paths = [
        node_path,
        hyperedge_path,
        title_path,
        summary_path,
    ]
    for plaintext, save_path in zip(plaintexts, save_paths):
        ids, embeddings = generate_embeddings(args, plaintext, model, tokenizer)
        assert len(ids) == embeddings.shape[0], "Number of ids and embeddings mismatch"
        assert len(ids) == len(plaintext.keys()), "Number of ids and plaintext mismatch"
        save_pickle({"ids": ids, "embeddings": embeddings}, save_path)


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_pretrain_reload = True
    main(args)
    print("")
