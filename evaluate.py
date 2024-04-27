import os
import os.path as osp

from arguments import parse_args
from src.load_data import load_json
from src.evaluation import f1_score, acc_score


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


if __name__ == "__main__":
    args = parse_args()
    main(args)
