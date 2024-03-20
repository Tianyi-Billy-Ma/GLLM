import argparse, os
import os.path as osp


def parse_args():
    parser = argparse.ArgumentParser()

    root_dir = os.getcwd()
    data_dir = osp.join(root_dir, "data")
    raw_data_dir = osp.join(data_dir, "raw")
    processed_data_dir = osp.join(data_dir, "processed")
    LLMs_dir = osp.join(root_dir, "models", "LLMs")
    GNNs_dir = osp.join(root_dir, "models", "GNNs")

    parser.add_argument("--root_dir", default=root_dir, type=str)
    parser.add_argument("--seed", default=3, type=int)
    parser.add_argument("--cuda", default=0, type=int)
    # args for task
    parser.add_argument("--task_mode", default="classification", type=str)
    parser.add_argument("--test_prop", default=0.2, type=float)
    # args for LLMs
    parser.add_argument("--LLMs_dir", default=LLMs_dir)
    parser.add_argument("--LLMs_model_name", default="llama-2-7b", type=str)
    # args for GNNs
    parser.add_argument("--GNNs_model_name", default="AllSetTransformer", type=str)
    parser.add_argument("-GNNs_epochs", default=300, type=int)
    parser.add_argument("-GNNs_lr", default=0.001, type=float)
    parser.add_argument("-GNNs_wd", default=0.0, type=float)
    parser.add_argument("--GNNs_aug", default="mask", type=str)
    parser.add_argument("--GNNs_aug_ratio", default=0.2, type=float)
    parser.add_argument("--GNNs_tau", default=0.2, type=float)
    parser.add_argument("--GNNs_dir", default=GNNs_dir)
    parser.add_argument("--GNNs_name", default="GCN", type=str)
    parser.add_argument("--GNNs_num_layers", default=2, type=int)
    parser.add_argument("--GNNs_dropout", default=0.5, type=float)
    parser.add_argument("--GNNs_aggregate", default="mean", type=str)
    parser.add_argument("--GNNs_normalization", default="ln", type=str)
    parser.add_argument("--GNNs_input_norm", action="store_false")
    parser.add_argument("--GNNs_GPR", action="store_false")
    parser.add_argument("--GNNs_LearnMask", default=False, type=bool)
    parser.add_argument("--GNNs_PMA", default=True, type=bool)
    parser.add_argument("--GNNs_MLP_hidden", default=256, type=int)
    parser.add_argument("--GNNs_num_MLP_layers", default=2, type=int)
    parser.add_argument("--GNNs_heads", default=4, type=int)
    parser.add_argument("--GNNs_classifier_hidden", default=256, type=int)
    parser.add_argument("--GNNs_num_classifier_layers", default=2, type=int)
    parser.add_argument("--GNNs_batch_size", default=64, type=int)
    # GNNs Placeholder
    parser.add_argument("--GNNs_num_features", type=int)
    parser.add_argument("--GNNs_num_classes", type=int)

    # args for Dataset
    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--raw_data_dir", default=raw_data_dir, type=str)
    parser.add_argument("--processed_data_dir", default=processed_data_dir, type=str)
    parser.add_argument("--dname", default="wikitablequestions", type=str)
    parser.add_argument("--format", default="table", type=str)
    parser.add_argument("--reprocess", action="store_true")

    # args for Pretrain
    parser.add_argument("--pretrain_model", default="all-mpnet-base-v2", type=str)

    # args for display
    parser.add_argument("--display_step", default=1, type=int)

    args = parser.parse_args()

    return args
