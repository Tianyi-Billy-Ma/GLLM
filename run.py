import os
import os.path as osp
import time

from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from arguments import parse_args
from models.LLMs.models import Retriever
from src.load_data import load_pickle
from src.prompt import generate_prompt
from src.preprocess import add_special_token
from src.load_data import load_json, save_json
from src.normalize_text import normalize
from tqdm import tqdm
import re
import torch


def run(args, qas, rephrase_questions, model, retriever, output_dir=None):
    assert (output_dir and args.save_output) or (
        not output_dir and not args.save_output
    )
    num_questions = len(qas.items())
    prompts = []
    questions, ground_truths = [], []
    qids = []

    # for idx, (qid, qa) in tqdm(enumerate(qas.items())):
    for idx, (qid, qa) in enumerate(qas.items()):
        ground_truth = (
            qa["answer"][0] if isinstance(qa["answer"], list) else qa["answer"]
        )
        question = (
            rephrase_questions[qid] if args.LLMs_rephrase_question else qa["question"]
        )
        qids.append(qid)
        questions.append(question)
        ground_truths.append(ground_truth)

        documents = retriever.search_document(question, 15)
        evidence = documents["hyperedge"]
        prompt = generate_prompt(args, question, evidence)
        prompts.append(prompt)

        # if len(questions) == args.LLMs_question_batch_size or idx == num_questions - 1:
        #     documents = retriever.search_document_batch(questions, 15)
        #     evidences = [document["hyperedge"] for document in documents]
        #     prompts = [
        #         generate_prompt(args, question, evidence)
        #         for question, evidence in zip(questions, evidences)
        #     ]
        if len(prompts) == args.LLMs_question_batch_size or idx == num_questions - 1:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=100, stop=["</answer>"]
            )
            response = model.generate(prompts, sampling_params)
            predictions = [normalize(result.outputs[0].text) for result in response]
            if args.save_output:
                for qid, prompt, question, gt, pred in zip(
                    qids, prompts, questions, ground_truths, predictions
                ):
                    save_json(
                        {
                            "prompt": prompt,
                            "question": question,
                            "ground_truth": gt,
                            "prediction": pred,
                        },
                        osp.join(output_dir, f"{qid}.json"),
                    )
            prompts, ground_truths, questions, qids = [], [], [], []
        if idx % 1000 == 0:
            print(f"Finished {idx}/{num_questions} questions")


1


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

    run(args, qas, rephrase_questions, model, retriever, output_dir)
    # model.set_tokenizer(tokenzier)


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    args.LLMs_rephrase_question = True
    args.save_output = False
    args.LLMs_question_batch_size = 64
    main(args)
    print("")

    # args = parse_args()
    # main(args)
    # print("")
