from datasets import Dataset as ds
import os
import os.path as osp
import logging
import argparse
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModel
import pickle
import numpy as np

logger = logging.getLogger(__name__)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

raw_arff_dnames = [
    "bank",
    "creditg",
    "blood",
    "jungle",
    "calhousing",
    "income",
    "car",
    "heart",
    "diabetes",
]


def load_passages(path):
    passages = []
    data = load_pickle(path)
    for idx, passage in enumerate(data):
        passages.append({"id": str(idx), "text": passage})
    return passages


def load_pickle(pickle_path):
    with open(pickle_path, mode="rb") as f:
        data = pickle.load(f)
    return data


def save_pickle(data, pickle_path):
    with open(pickle_path, mode="wb") as f:
        pickle.dump(data, f)


def byte_to_string_columns(data):
    for col, dtype in data.dtypes.items():
        if dtype == object:  # Only process byte object columns.
            data[col] = data[col].apply(lambda x: x.decode("utf-8"))
    return data


def load_raw_data(args):
    logging.info(f"Loading raw dataset {args.dname}")
    dname = args.dname
    raw_data_dir = args.raw_data_dir
    if dname == "wikitablequestions":
        dataset = ds.load_dataset("wikitablequestions", trust_remote_code=True)
        from datasets import concatenate_datasets

        dataset = concatenate_datasets(
            [dataset["train"], dataset["validation"], dataset["test"]]
        )
    else:
        logger.error(f"Dataset {args.dname} not supported")

    return dataset


def load_data(args, format=None):
    format = args.format if format is None else format
    processed_data_dir = args.processed_data_dir
    data_path = osp.join(processed_data_dir, args.dname, format)
    if format == "pandas":
        dataset = load_raw_data(args)
    elif osp.isdir(data_path) and not args.reprocess:
        dataset = ds.load_from_disk(data_path)
    else:
        logger.info(
            f"Dataset {args.dname} in {format} format not found. Downloading and saving"
        )
        dataset = download_and_save(args, format)
    return dataset


def split_data(dataset, args):
    ds = dataset.train_test_split(test_size=args.test_prop, seed=args.seed)
    return ds["train"], ds["test"]


def download_and_save(args, format):
    dataset = load_raw_data(args)
    if isinstance(dataset, pd.DataFrame):
        dataset = ds.from_pandas(dataset)
    save2disk(dataset, args, format)
    return dataset


def dataset2list(dataset):
    # formater = MessageGenerator(args)
    # messages = [formater.substitute(message) for _, message in dataset.iterrows()]
    # dataset = ds.from_dict(
    #         {
    #             "message": messages,
    #             "label": dataset["label"].to_list(),
    #             "id": dataset.index,
    #         }
    #     )
    assert isinstance(dataset, dataset.Dataset), "Input must be a dataset.Dataset"

    return dataset


def save2disk(dataset, args, format=None):
    format = args.format if format is None else format
    data_dir = osp.join(args.processed_data_dir, args.dname)
    os.makedirs(data_dir, exist_ok=True)
    save_path = osp.join(data_dir, format)
    dataset.save_to_disk(save_path)
    logger.info(f"Dataset {args.dname} saved to {save_path}")


def generate_emb(args, dataset):
    model_name = args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    all_ids, all_embeddings = [], []
    batch_idx, batch_text = [], []
    with torch.no_grad():
        for idx, row in enumerate(dataset):
            batch_idx.append(row["id"])
            batch_text.append(row["message"])
            if len(batch_text) == args.batch_size or idx == len(dataset) - 1:
                encoded_batch = tokenizer(
                    batch_text,
                    return_tensors="pt",
                    max_length=args.emb_length,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {idx: val.cuda() for idx, val in encoded_batch.items()}
                embeddings = model(**encoded_batch)
                if isinstance(embeddings, dict):
                    embeddings = embeddings["pooler_output"]
                embeddings = embeddings.cpu()
                all_ids.extend(batch_idx)
                all_embeddings.append(embeddings)
                batch_idx, batch_text = [], []
                logger.info(f"Encoded {idx} samples")
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    save_dir = osp.join(args.processed_data_dir, args.dname, "embeddings")
    os.makedirs(save_dir, exist_ok=True)
    save_file = osp.join(save_dir, f"{args.format}.pickle")
    with open(save_file, mode="wb") as f:
        pickle.dump((all_ids, all_embeddings), f)
    return all_ids, all_embeddings


def stat_tables(dataset):
    num_tabs = len(dataset)
    num_cols = [len(d["table"]["header"]) for d in dataset]
    num_rows = [len(d["table"]["rows"]) for d in dataset]

    avg_cols, avg_rows = np.mean(num_cols), np.mean(num_rows)
    min_cols, min_rows = np.min(num_cols), np.min(num_rows)
    max_cols, max_rows = np.max(num_cols), np.max(num_rows)

    print(f"Number of tables: {num_tabs}")
    print(f"Average number of cols/rows: {avg_cols:.2f}/{avg_rows:.2f}")
    print(f"Max number of cols/rows: {max_cols:.2f}/{max_rows:.2f}")
    print(f"Min number of cols/rows: {min_cols:.2f}/{min_rows:.2f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    root_dir = os.getcwd()
    data_dir = osp.join(root_dir, "data")
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--raw_data_dir", default=osp.join(data_dir, "raw"), type=str)
    parser.add_argument(
        "--processed_data_dir", default=osp.join(data_dir, "processed"), type=str
    )
    parser.add_argument("--dname", default="bank", type=str)
    parser.add_argument("--format", default="list", type=str)
    parser.add_argument("--reprocess", action="store_true")
    parser.add_argument(
        "--model_name_or_path", default="facebook/contriever-msmarco", type=str
    )
    parser.add_argument("--batch_size", default=512, type=int)
    parser.add_argument("--emb_length", default=512, type=int)

    args = parser.parse_args()
    args.reprocess = True

    # all_ids, all_embeddings = load_emb(args)
    print("")
