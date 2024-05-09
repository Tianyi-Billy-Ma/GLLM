import argparse, os
import os.path as osp

from src import preprocess


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
    parser.add_argument("--task_mode", default="RAGTable", type=str)
    parser.add_argument("--test_prop", default=0.2, type=float)
    # args for save
    parser.add_argument("--save_output", action="store_true")
    # args for LLMs
    parser.add_argument("--LLMs_dir", default=LLMs_dir)
    parser.add_argument(
        "--LLMs_model_name", default="meta-llama/Meta-Llama-3-8B-Instruct", type=str
    )
    parser.add_argument(
        "--LLMs_retriever_model_name", default="facebook/contriever-msmarco", type=str
    )
    parser.add_argument(
        "--LLMs_generator_model_name",
        default="meta-llama/Meta-Llama-3-8B-Instruct",
        type=str,
    )
    parser.add_argument("--LLMs_max_new_tokens", default=15, type=int)
    parser.add_argument(
        "--LLMs_world_size",
        default=1,
        type=int,
        help="world size to use multiple GPUs.",
    )
    parser.add_argument("--LLMs_no_fp16", action="store_true", help="inference in fp32")
    parser.add_argument("--LLMs_dtype", default="half", type=str)
    parser.add_argument("--LLMs_projection_size", type=int, default=768)
    parser.add_argument(
        "--LLMs_n_subquantizers",
        type=int,
        default=0,
        help="Number of subquantizer used for vector quantization, if 0 flat index is used",
    )
    parser.add_argument(
        "--LLMs_n_bits", type=int, default=8, help="Number of bits per subquantizer"
    )
    parser.add_argument("--LLMs_save_or_load_index", action="store_false")
    parser.add_argument("--LLMs_reload_index", action="store_true")
    parser.add_argument("--LLMs_indexing_batch_size", type=int, default=1000000)
    parser.add_argument("--LLMs_lowercase", action="store_true")
    parser.add_argument("--LLMs_normalize_text", action="store_true")
    parser.add_argument("--LLMs_passage_batch_size", default=64, type=int)
    parser.add_argument("--LLMs_passage_max_length", default=512, type=int)
    parser.add_argument("--LLMs_passage_max_token_length", default=512, type=int)
    parser.add_argument("--LLMs_question_batch_size", default=1, type=int)
    parser.add_argument("--LLMs_question_maxlength", default=512, type=int)
    parser.add_argument("--LLMs_num_docs", default=100, type=int)
    # args for Pretrain
    parser.add_argument(
        "--LLMs_pretrain_model", default="facebook/contriever-msmarco", type=str
    )
    parser.add_argument("--LLMs_pretrain_batch_size", default=64, type=int)

    parser.add_argument("--LLMs_pretrain_include_tags", action="store_true")
    parser.add_argument("--LLMs_pretrain_vocab_size", type=int)
    # args for retriever
    parser.add_argument("--LLMs_retriever_include_tags", action="store_true")
    parser.add_argument(
        "--LLMs_retriever_input_path",
        default=osp.join(processed_data_dir, "wikitablequestions"),
    )
    # args for baseline
    parser.add_argument("--LLMs_table_plaintext_format", default="dict", type=str)
    parser.add_argument("--GNNs_table_embedding_format", default="all", type=str)
    # args for GNNs
    parser.add_argument("--GNNs_model_name", default="AllSetTransformer", type=str)
    parser.add_argument("-GNNs_epochs", default=5, type=int)
    parser.add_argument("-GNNs_lr", default=0.001, type=float)
    parser.add_argument("-GNNs_wd", default=0.0, type=float)
    parser.add_argument("--GNNs_aug", default="mask", type=str)
    parser.add_argument("--GNNs_aug_ratio", default=0.2, type=float)
    parser.add_argument("--GNNs_tau", default=0.07, type=float)
    parser.add_argument("--GNNs_dir", default=GNNs_dir)
    parser.add_argument("--GNNs_num_layers", default=1, type=int)
    parser.add_argument("--GNNs_dropout", default=0.0, type=float)
    parser.add_argument("--GNNs_hidden_dim", default=768, type=int)
    parser.add_argument("--GNNs_MLP_hidden", default=768, type=int)
    parser.add_argument("--GNNs_layernorm_eps", default=1e-12, type=float)
    parser.add_argument("--GNNs_pre_norm", action="store_true")
    parser.add_argument("--GNNs_num_heads", default=8, type=int)
    parser.add_argument("--GNNs_batch_size", default=32, type=int)
    parser.add_argument("--GNNs_gated_proj", action="store_true")
    parser.add_argument("--GNNs_pretrain_emb", action="store_true")
    parser.add_argument("-GNNs_activation_fn", default="relu")
    # GNNs Placeholder
    parser.add_argument("--GNNs_num_features", type=int)
    parser.add_argument("--GNNs_num_classes", type=int)
    parser.add_argument(
        "--GNNs_reverse_HG", action="store_false", help="reverse hypergraph"
    )
    # args for Dataset
    parser.add_argument("--reprocess_dataset", default=True, type=bool)
    parser.add_argument("--data_dir", default=data_dir, type=str)
    parser.add_argument("--raw_data_dir", default=raw_data_dir, type=str)
    parser.add_argument("--processed_data_dir", default=processed_data_dir, type=str)
    parser.add_argument("--dname", default="wikitablequestions", type=str)
    parser.add_argument("--format", default="table", type=str)
    parser.add_argument("--reprocess", action="store_true")

    # args for display
    parser.add_argument("--display_step", default=1, type=int)

    args = parser.parse_args()

    return args
