import os
import os.path as osp

from arguments import parse_args
from src.load_data import load_json, load_pickle
from src.evaluation import f1_score, acc_score
import re


def main(args):
    output_dir = osp.join(
        args.root_dir,
        "output",
        "RAGTableQA_Meta-Llama-3-8B-Instruct_2024-04-27-04-49-57",
    )
    files = os.listdir(output_dir)
    gts, predictions = [], []
    for file in files:
        file_path = osp.join(output_dir, file)
        data = load_json(file_path)
        if isinstance(data, dict):
            gt, prediction = data["ground_truth"], data["prediction"]
            gts.append(gt)
            predictions.append(prediction.replace("\n", ""))
        else:
            for qa in data:
                gt, prediction = qa["ground_truth"], qa["prediction"]
                gts.append(gt)
                predictions.append(prediction.replace("\n", ""))
    metric = {
        "f1_score": f1_score(predictions, gts),
        "accuracy": acc_score(predictions, gts),
    }
    print(metric)
    return metric


def evaluate_retriever(args, fname):
    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")

    output_dir = osp.join(
        args.root_dir,
        "output",
        fname,
    )

    predictions = load_json(osp.join(output_dir, "predictions.json"))
    data = load_pickle(osp.join(pretrain_dir, "plaintext_tables.pickle"))
    tables, qas = data["tables"], data["qas"]
    mappings = load_pickle(osp.join(pretrain_dir, "mapping.pickle"))
    qid2tid = mappings["qid2tid"]

    num_questions = len(qas.keys())

    res = {
        "title": {1: 0, 3: 0, 5: 0, 10: 0},
        "summary": {1: 0, 3: 0, 5: 0, 10: 0},
        "table": {1: 0, 3: 0, 5: 0, 10: 0},
    }

    for qid, prediction in predictions.items():
        gt = qid2tid[qid].split("_")[-1]
        if isinstance(prediction, list):
            prefix = "table"
            prefix_res = [name.split("_")[-1] for name in prediction]
            for k in res[prefix].keys():
                if gt in prefix_res[:k]:
                    res[prefix][k] += 1
        else:
            for prefix in ["title", "summary", "table"]:
                if prefix in prediction.keys():
                    prefix_res = [name.split("_")[-1] for name in prediction[prefix]]
                    for k in res[prefix].keys():
                        if gt in prefix_res[:k]:
                            res[prefix][k] += 1

    for prefix in res.keys():
        prefix_acc = res[prefix]
        res[prefix] = {
            k: round(100 * v / num_questions, 2) for k, v in prefix_acc.items()
        }
    return res


if __name__ == "__main__":
    args = parse_args()
    fnames = os.listdir(osp.join(args.root_dir, "output"))
    for table_format in ["md", "dict", "html", "sentence"]:
        for sub in ["", "_summary", "_summary_title", "_title"]:
            table_plaintext_format = table_format + sub
            pattern = re.compile(
                r"^"
                + re.escape(
                    "RAGTable_" + table_plaintext_format + "_Meta-Llama-3-8B-Instruct"
                )
            )
            fname = [fname for fname in fnames if pattern.match(fname)][0]
            res = evaluate_retriever(args, fname)
            print("Table format:", table_plaintext_format)
            print(res)
