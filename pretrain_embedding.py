from src import load_data, load_pickle, save_pickle
from arguments import parse_args
from sentence_transformers import SentenceTransformer
import os, os.path as osp


def generate_unique_tables(dataset):
    qt_mappings = {}
    unique_ts = {}
    qas = {}
    for row in dataset:
        t = row["table"]
        tname = t["name"]
        id = row["id"]
        q = row["question"]
        a = row["answers"]
        if tname in unique_ts.keys():
            assert len(unique_ts[tname]["rows"]) == len(
                t["rows"]
            ), "Table with same name has different number of rows"
            assert (
                unique_ts[tname]["header"] == t["header"]
            ), "Table with same name has different header"
        else:
            unique_ts[tname] = t
        qas[id] = {
            "question": q,
            "answers": a,
        }
        qt_mappings[id] = tname
    return unique_ts, qt_mappings, qas


def generate_node_plaintext_within_tables(tables):
    def generate_node_plaintext_from_table(table):
        header = table["header"]
        rows = table["rows"]
        return [f"{header[i]} is {cell}" for row in rows for i, cell in enumerate(row)]

    return {tname: generate_node_plaintext_from_table(t) for tname, t in tables.items()}


if __name__ == "__main__":
    args = parse_args()
    dataset = load_data(args)
    unique_ts, q2t_mappings, qas = generate_unique_tables(dataset)
    dict_nodes_plaintext = generate_node_plaintext_within_tables(unique_ts)
    nodes = []
    t2n_mappings = []
    for key, value in dict_nodes_plaintext.items():
        nodes.extend(value)
        t2n_mappings.extend([key] * len(value))
    model = SentenceTransformer(args.pretrain_model)
    sentence_embeddings = model.encode(nodes)
    save_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    save_path = osp.join(save_dir, f"{args.pretrain_model}.pickle")
    save_pickle(
        {
            "t2n_mappings": t2n_mappings,
            "q2t_mappings": q2t_mappings,
            "qas": qas,
            "embeddings": sentence_embeddings,
            "tables": unique_ts,
        },
        save_path,
    )
    print("")
