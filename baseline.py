from json import load
import os
import os.path as osp
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from arguments import parse_args
from models.LLMs.models import Retriever
from src.load_data import load_pickle
from src.helper import PROMPT_DICT, TASK_DICT
from src.preprocess import add_special_token
from src.load_data import load_json
from tqdm import tqdm
import re


def generate_prompt(question, evidences):
    # instruction = TASK_DICT["TableQA"]
    instruction = TASK_DICT["TableQA"]
    if type(evidences) == dict:
        documents = "\n".join([doc for _, doc in evidences.items()])
    else:
        documents = "\n".join(evidences)
    prompt = PROMPT_DICT["TableQA"]
    return prompt.format_map(
        {"instruction": instruction, "documents": documents, "question": question}
    )


def main(args):
    ### Step 1: Load data

    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    raw_dir = osp.join(args.raw_data_dir, args.dname)

    rephrase_files = [
        file.split(".")[0] for file in os.listdir(osp.join(raw_dir, "rephrase"))
    ]
    data = load_pickle(osp.join(pretrain_dir, "plaintext_tables.pickle"))

    tables, qas = data["tables"], data["qas"]

    tid2eids = load_pickle(osp.join(GNNs_dir, "tid2eids.pickle"))
    plaintext_hyperedges = load_pickle(
        osp.join(pretrain_dir, "plaintext_hyperedge.pickle")
    )
    mappings = load_pickle(osp.join(pretrain_dir, "mapping.pickle"))

    qid2tid = mappings["qid2tid"]

    model = LLM(
        model=args.LLMs_generator_model_name,
        download_dir=args.LLMs_dir,
        dtype=args.LLMs_dtype,
        tensor_parallel_size=args.LLMs_world_size,
    )
    # model.set_tokenizer(tokenzier)

    num_questions = len(qas.items())
    correct = 0
    prompts = []
    questions, ground_truths = [], []
    for idx, (qid, qa) in tqdm(enumerate(qas.items())):
        tid = qid2tid[qid]
        eids = tid2eids[tid]
        evidences = {eid: plaintext_hyperedges[eid] for eid in eids}

        question, ground_truth = qa["question"], qa["answer"][0]
        questions.append(question)
        ground_truths.append(ground_truth)

        prompt = generate_prompt(question, evidences)
        prompts.append(prompt)
        if len(prompts) == 64:
            sampling_params = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=30)
            response = model.generate(prompts, sampling_params)
            answers = [result.outputs[0].text for result in response]
            correct += sum(
                (ground_truth in answer)
                for ground_truth, answer in zip(ground_truths, answers)
            )
            prompts, questions, ground_truth = [], [], []
        if idx % 1000 == 0:
            print(f"Finished {idx}/{num_questions} questions")
    print("Accuracy: ", correct / num_questions)


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    args.LLMs_rephrase_question = True
    main(args)
    print("")

    # args = parse_args()
    # main(args)
    # print("")
