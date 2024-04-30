from multiprocessing.pool import ThreadPool
import os
import os.path as osp
import time

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from vllm import LLM
from arguments import parse_args
from models.LLMs.models import Retriever
from src.load_data import load_pickle
import torch

from run import run

num_cores = 8


def main(args):
    ### Step 1: Load data

    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    raw_dir = osp.join(args.raw_data_dir, args.dname)

    if args.LLMs_rephrase_question:
        rephrase_questions_path = osp.join(
            pretrain_dir, "plaintext_rephrased_questions.pickle"
        )
        rephrase_questions = load_pickle(rephrase_questions_path)

    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    if args.save_output:
        save_model_name = args.LLMs_generator_model_name.split("/")[-1]
        output_dir = osp.join(
            args.root_dir,
            "output",
            f"{args.task_mode}_{save_model_name}_{curr_time}",
        )
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = None

    data = load_pickle(osp.join(pretrain_dir, "plaintext_tables.pickle"))
    tables, qas = data["tables"], data["qas"]

    retriever = Retriever(args)
    retriever.setup_retriever()
    # res = retriever.search_document(query1)

    # tokenzier = add_special_token(tokenzier)
    torch.cuda.empty_cache()
    model = LLM(
        model=args.LLMs_generator_model_name,
        download_dir=args.LLMs_dir,
        dtype=args.LLMs_dtype,
        tensor_parallel_size=args.LLMs_world_size,
    )

    batch_qas = [{} for _ in range(num_cores)]

    for idx, (qid, qa) in enumerate(qas.items()):
        work_id = idx % num_cores
        batch_qas[work_id][qid] = qa

    with ThreadPoolExecutor(max_workers=num_cores) as executor:
        futures = []
        for work_id in range(num_cores):
            futures.append(
                executor.submit(
                    run,
                    args,
                    batch_qas[work_id],
                    rephrase_questions,
                    model,
                    retriever,
                    output_dir,
                )
            )
        for future in futures:
            future.result()


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    args.LLMs_rephrase_question = True
    args.save_output = True
    args.LLMs_question_batch_size = 128
    main(args)
    print("")
