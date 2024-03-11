from src import load_pickle, load_data
import os, os.path as osp
from arguments import parse_args


def construct_hypergraph(args, tables, embs):
    pass


if __name__ == "__main__":
    args = parse_args()

    pickle_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", f"{args.pretrain_model}.pickle"
    )
    dict_tables = load_pickle(pickle_path)
    t2n_mappings, q2t_mappings, qas, embeddings, tables = (
        dict_tables["t2n_mappings"],
        dict_tables["q2t_mappings"],
        dict_tables["qas"],
        dict_tables["embeddings"],
        dict_tables["tables"]
    )
    
    print("")
