from audioop import reverse
from collections import defaultdict
from string import Template
from typing import Mapping
from matplotlib import table
import pandas as pd
import re
import numpy as np
import torch
from sympy import hyper

from tqdm import tqdm

from torch_geometric.loader import DataLoader
from .helper import BipartiteData
from .normalize_text import normalize

from .load_data import load_json, load_pickle
import os
import os.path as osp
import json


START_NODE_TAG = "[NODE]"
END_NODE_TAG = "[/NODE]"
START_ROW_TAG = "[ROW]"
END_ROW_TAG = "[/ROW]"
START_COL_TAG = "[COL]"
END_COL_TAG = "[/COL]"
START_TITLE_TAG = "[TITLE]"
END_TITLE_TAG = "[/TITLE]"
START_SUMMARY_TAG = "[SUMMARY]"
END_SUMMARY_TAG = "[/SUMMARY]"
NULL_TAG = "[NULL]"

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


def generate_node_plaintext_from_tables(args, tables):
    lowercase, normalize_text = args.LLMs_lowercase, args.LLMs_normalize_text

    def generate_node_plaintext_from_table(table):
        header = table["header"]
        rows = table["rows"]
        res = []
        for row in rows:
            for i, cell in enumerate(row):
                if "dict" in args.LLMs_table_plaintext_format:
                    text = f"{header[i]}: {cell if cell else NULL_TAG}"
                elif "html" in args.LLMs_table_plaintext_format:
                    text = f"<{header[i]}>{cell if cell else NULL_TAG}"
                elif "sentence" in args.LLMs_table_plaintext_format:
                    text = f"{header[i]} is {cell if cell else NULL_TAG}"
                else:
                    raise ValueError(
                        "Node format should be either dict, html, or sentence"
                    )
                if lowercase:
                    text = text.lower()
                if normalize_text:
                    text = normalize(text)
                text = START_NODE_TAG + text + END_NODE_TAG
                res.append(text)
        assert len(res) == len(header) * len(rows), "Number of nodes mismatch"
        return res

    return {tname: generate_node_plaintext_from_table(t) for tname, t in tables.items()}


def generate_hyperedges_plaintext_from_tables(
    args,
    tables,
):
    lowercase, normalize_text = args.LLMs_lowercase, args.LLMs_normalize_text

    def generate_hyperedges_plaintext_from_table(table):
        passages = []
        headers = table["header"]
        rows = table["rows"]
        for row in rows:
            if "dict" in args.LLMs_table_plaintext_format:
                m = "; ".join(
                    [
                        f"{headers[idx]}: {cell if cell else NULL_TAG}"
                        for idx, cell in enumerate(row)
                    ]
                )
            elif "html" in args.LLMs_table_plaintext_format:
                m = " ".join(
                    [
                        f"<{headers[idx]}>{cell if cell else NULL_TAG}"
                        for idx, cell in enumerate(row)
                    ]
                )
            elif "sentence" in args.LLMs_table_plaintext_format:
                m = ". ".join(
                    [
                        f"{headers[idx]} is {cell if cell else NULL_TAG}"
                        for idx, cell in enumerate(row)
                    ]
                )
            else:
                raise ValueError(
                    "Hyperedges format should be either dict, html, or sentence"
                )
            if lowercase:
                m = m.lower()
            if normalize_text:
                m = normalize(m)
            m = START_ROW_TAG + m + END_ROW_TAG
            passages.append(m)
        for idx, header in enumerate(headers):
            if "dict" in args.LLMs_table_plaintext_format:
                m = "; ".join(
                    [
                        f"{header} is {row[idx] if row[idx] else NULL_TAG}"
                        for row in rows
                    ]
                )
            elif "html" in args.LLMs_table_plaintext_format:
                m = " ".join(
                    [f"<{header}>{row[idx] if row[idx] else NULL_TAG}" for row in rows]
                )
            elif "sentence" in args.LLMs_table_plaintext_format:
                m = ". ".join(
                    [
                        f"{header} is {row[idx] if row[idx] else NULL_TAG}"
                        for row in rows
                    ]
                )
            else:
                raise ValueError(
                    "Hyperedges format should be either dict, html, or sentence"
                )
            if lowercase:
                m = m.lower()
            if normalize_text:
                m = normalize(m)
            m = START_COL_TAG + m + END_COL_TAG
            passages.append(m)
        return passages

    return {
        tname: generate_hyperedges_plaintext_from_table(t)
        for tname, t in tables.items()
    }


def generate_plaintext_from_table(table, args=None):
    header = table["header"]
    rows = table["rows"]
    res = ""
    if args == None:
        res = (
            "| "
            + " | ".join(
                [normalize(head).replace("\n", " ") for head in table["header"]]
            )
            + "|\n"
        )
        res += "| " + " | ".join(["---" for _ in table["header"]]) + "|\n"
        for row in table["rows"]:
            res += (
                "| "
                + " |".join(normalize(element).replace("\n", " ") for element in row)
                + "|\n"
            )
    else:
        if "md" in args.LLMs_table_plaintext_format:
            res = generate_plaintext_from_table(table)

        elif "dict" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    "; ".join([f"{header[i]}: {cell}" for i, cell in enumerate(row)])
                    + "\n"
                )
        elif "html" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    " ".join([f"<{header[i]}>{cell}" for i, cell in enumerate(row)])
                    + "\n"
                )
        elif "sentence" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    ". ".join([f"{header[i]} is {cell}" for i, cell in enumerate(row)])
                    + "\n"
                )

        if "summary" in args.LLMs_table_plaintext_format:
            if (
                args.LLMs_pretrain_include_tags
                and (not table["summary"].startswith(START_SUMMARY_TAG))
                and (not table["summary"].endswith(END_SUMMARY_TAG))
            ):
                res = (
                    START_SUMMARY_TAG + table["summary"] + END_SUMMARY_TAG + "\n" + res
                )
            elif (
                not args.LLMs_pretrain_include_tags
                and table["summary"].startswith(START_SUMMARY_TAG)
                and table["summary"].endswith(END_SUMMARY_TAG)
            ):
                res = (
                    "Table Summary:"
                    + table["summary"]
                    .replace(START_SUMMARY_TAG, "")
                    .replace(END_SUMMARY_TAG, "")
                    + "\n"
                    + res
                )
            else:
                raise ValueError(
                    "Table summary should be enclosed with [SUMMARY] and [/SUMMARY] or not enclosed with [SUMMARY] and [/SUMMARY]"
                )
        if "title" in args.LLMs_table_plaintext_format:
            if args.LLMs_pretrain_include_tags and (
                not table["title"].startswith(START_TITLE_TAG)
                and not table["title"].endswith(END_TITLE_TAG)
            ):
                res = START_TITLE_TAG + table["title"] + END_TITLE_TAG + "\n" + res
            elif (
                not args.LLMs_pretrain_include_tags
                and table["title"].startswith(START_TITLE_TAG)
                and table["title"].endswith(END_TITLE_TAG)
            ):
                res = (
                    "Table Title:"
                    + table["title"]
                    .replace(START_TITLE_TAG, "")
                    .replace(END_TITLE_TAG, "")
                    + "\n"
                    + res
                )
            else:
                raise ValueError(
                    "Table title should be enclosed with [TITLE] and [/TITLE] or not enclosed with [TITLE] and [/TITLE]"
                )
    return res


def generate_tokens_and_token_ids(passage, tokenizer, max_length):
    PADDING_TAG = tokenizer.pad_token
    tokens = tokenizer.tokenize(passage)[:max_length]
    while len(tokens) < max_length:
        tokens.append(PADDING_TAG)
    ids = tokenizer.convert_tokens_to_ids(tokens)
    return tokens, ids


def add_special_token(tokenizer):
    tokenizer.add_special_tokens(
        special_tokens_dict={
            "additional_special_tokens": [
                START_NODE_TAG,
                END_NODE_TAG,
                START_ROW_TAG,
                END_ROW_TAG,
                START_COL_TAG,
                END_COL_TAG,
                START_TITLE_TAG,
                END_TITLE_TAG,
                START_SUMMARY_TAG,
                END_SUMMARY_TAG,
                NULL_TAG,
            ]
        }
    )
    return tokenizer


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
    return edge_index, num_rows, num_cols


def construct_hypergraph(args, tables, passage_dict, model, tokenizer, mappings):
    hypergraph_list = []

    node_passages, hyperperedge_passages = (
        passage_dict["nodes"],
        passage_dict["hyperedges"],
    )

    nids = [nid for nid, _ in node_passages.items()]
    eids = [eid for eid, _ in hyperperedge_passages.items()]
    tids = [tid for tid, _ in tables.items()]

    tid2nids, tid2eids = mappings  # table id to node ids, table id to hyperedge ids
    if args.GNNs_pretrain_emb:
        table_passages = passage_dict["tables"]
        _, node_embeddings = generate_embeddings(args, node_passages, model, tokenizer)
        _, hyperedge_embeddings = generate_embeddings(
            args, hyperperedge_passages, model, tokenizer
        )
        _, table_embeddings = generate_embeddings(
            args, table_passages, model, tokenizer
        )

    for t_idx, (key, val) in enumerate(tables.items()):
        tid = f"table_{t_idx}"

        edge_index, num_rows, num_cols = generate_edge_index(val)

        if args.GNNs_pretrain_emb:
            x_s, x_t = [], []
            x_s = np.array(
                [
                    node_embeddings[i]
                    for i, nid in enumerate(nids)
                    if nid in tid2nids[tid]
                ]
            )
            x_t = np.array(
                [
                    hyperedge_embeddings[i]
                    for i, eid in enumerate(eids)
                    if eid in tid2eids[tid]
                ]
                + [table_embeddings[t_idx]]
            )
            x_s = torch.FloatTensor(x_s)
            x_t = torch.FloatTensor(x_t)

        else:
            xs_ids, xt_ids = [], []
            curr_nids = tid2nids[tid]
            curr_eids = tid2eids[tid]
            for id in curr_nids + curr_eids:
                assert (
                    id.startswith("node")
                    or id.startswith("row")
                    or id.startswith("col")
                )
                passage = (
                    node_passages[id]
                    if id.startswith("node")
                    else hyperperedge_passages[id]
                )
                passage = normalize(passage)
                # _, token_ids = generate_tokens_and_token_ids(
                #     passage, tokenizer, args.LLMs_passage_max_token_length
                # )
                token_ids = tokenizer.encode(
                    passage,
                    max_length=args.LLMs_passage_max_token_length,
                    truncation=True,
                    padding="max_length",
                )
                if id.startswith("node"):
                    xs_ids.append(token_ids)
                else:
                    xt_ids.append(token_ids)
            assert len(xs_ids) == num_rows * num_cols, "Number of nodes mismatch"
            assert len(xt_ids) == num_rows + num_cols, "Number of hyperedges mismatch"

            xs_ids = torch.LongTensor(xs_ids)
            xt_ids = torch.LongTensor(xt_ids)

            ### Add Table ids to x_t

            table_passage = table_passages[tid]
            # _, token_ids = generate_tokens_and_token_ids(
            #     table_passage, tokenizer, args.LLMs_passage_max_token_length
            # )
            token_ids = tokenizer.encode(
                passage,
                max_length=args.LLMs_passage_max_token_length,
                truncation=True,
                padding="max_length",
            )
            token_ids = torch.LongTensor(token_ids)
            xt_ids = torch.cat([xt_ids, token_ids.unsqueeze(0)], dim=0)

            x_s, x_t = xs_ids, xt_ids

        edge_index = torch.LongTensor(edge_index)

        assert edge_index[0, :].max() + 1 == x_s.shape[0], "Nodes mismatch"
        assert edge_index[1, :].max() + 1 == x_t.shape[0], "Hyperedges mismatch"

        hg = BipartiteData(
            x_s=x_s,
            x_t=x_t,
            edge_index=edge_index,
        )
        hg = reverse_bipartite_data(hg) if args.GNNs_reverse_HG else hg
        hg.num_nodes = hg.x_s.shape[0]
        hg.num_hyperedges = hg.x_t.shape[0]
        hypergraph_list.append(hg)

    HG = DataLoader(
        hypergraph_list, batch_size=args.GNNs_batch_size, shuffle=True, drop_last=False
    )
    return HG


def generate_embeddings(args, passages, model=None, tokenizer=None):
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
                    max_length=args.LLMs_passage_max_length,
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


def process_questions(args):
    raw_question_dir = osp.join(args.raw_data_dir, args.dname, "rephrase")
    files = os.listdir(raw_question_dir)

    rephrased_questions = {}
    for file in files:
        qid = file.split(".")[0]
        if f"{qid}.json" in files:
            response = load_json(osp.join(raw_question_dir, f"{qid}.json"))
        elif f"{qid}.pickle" in files:
            response = load_pickle(osp.join(raw_question_dir, f"{qid}.pickle"))
            element1 = response.split("\n")[1]
            element2 = response.split("\n")[2]
            if '"' in element1 and '"' in element2:
                while element2[-1] != '"':
                    element2 = element2[:-1]

                def replace_special_char(element):
                    indices = [i for i in range(len(element)) if element[i] == '"']
                    indices = indices[:3] + [indices[-1]]
                    return "".join(
                        [
                            char
                            if (char != '"') or (char == '"' and i in indices)
                            else "'"
                            for i, char in enumerate(element)
                        ]
                    )

                element1 = replace_special_char(element1)
                element2 = replace_special_char(element2)
                modified_response = "{" + element1 + element2 + "}"
            else:

                def add_special_char(v1):
                    index1 = v1.lower().find("original")
                    if index1 == -1:
                        index1 = v1.lower().find("rephrased")
                    index2 = v1.lower().find("question")
                    index3 = v1.find(":")
                    index4 = len(v1)
                    assert index1 != -1 and index2 != -1 and index3 != -1, "Error"
                    return (
                        v1[:index1]
                        + '"'
                        + v1[index1:index3]
                        + '":"'
                        + v1[index3 + 1 : index4]
                        + '"'
                    )

                element1 = add_special_char(element1)
                element2 = add_special_char(element2)
                modified_response = "{" + element1 + "," + element2 + "}"
            response = json.loads(modified_response)
        else:
            print(f"File {qid} not found.")
            continue
        if (
            "rephrased question" in response.keys()
            and "original question" in response.keys()
        ):
            rephrased_question = response["rephrased question"]
        elif (
            "rephrased_question" in response.keys()
            and "original_question" in response.keys()
        ):
            rephrased_question = response["rephrased_question"]
        elif (
            "original_question" in response.keys()
            and "accurate_question" in response.keys()
        ):
            rephrased_question = response["accurate_question"]
        elif (
            "Origianl question" in response.keys()
            and "Rephrased question" in response.keys()
        ):
            rephrased_question = response["Rephrased question"]
        elif (
            "Original Question" in response.keys()
            and "Rephrased Question" in response.keys()
        ):
            rephrased_question = response["Rephrased Question"]
        else:
            print(f"File {qid} is not in good format")
            break
        rephrased_questions[qid] = rephrased_question

    return rephrased_questions
