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


def main(args):
    ### Step 1: Load data

    pretrain_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    GNNs_dir = osp.join(args.processed_data_dir, args.dname, "GNNs")
    raw_dir = osp.join(args.raw_data_dir, args.dname)
    rephrase_files = [
        file.split(".")[0] for file in os.listdir(osp.join(raw_dir, "rephrase"))
    ]
    curr_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.gmtime())

    if args.save_output:
        output_dir = osp.join(
            args.root_dir,
            "output",
            f"{args.task_mode}_{args.LLMs_generator_model_name.split("/")[-1]}_{curr_time}",
        )
        os.makedirs(output_dir, exist_ok=True)

    data = load_pickle(osp.join(pretrain_dir, "plaintext_tables.pickle"))
    tables, qas = data["tables"], data["qas"]
    mappings = load_pickle(osp.join(pretrain_dir, "mapping.pickle"))
    qid2tid = mappings["qid2tid"]

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
    # model.set_tokenizer(tokenzier)

    num_questions = len(qas.items())
    correct = 0
    prompts = []
    questions, ground_truths = [], []
    for idx, (qid, qa) in tqdm(enumerate(qas.items())):
        question, ground_truth = qa["question"], qa["answer"][0]

        questions.append(question)
        ground_truths.append(ground_truth)
        table = tables[qid2tid[qid]]

        evidences = (
            "| "
            + " |".join(
                [normalize(head).replace("\n", " ") for head in table["header"]]
            )
            + "\n"
        )
        for row in table["rows"]:
            evidences += (
                "| "
                + " |".join(normalize(element).replace("\n", " ") for element in row)
                + "\n"
            )

        prompt = generate_prompt(args, question, evidences)
        prompts.append(prompt)
        if len(prompts) == args.LLMs_question_batch_size or idx == num_questions - 1:
            sampling_params = SamplingParams(
                temperature=0.0, top_p=1.0, max_tokens=100, stop=["</answer>"]
            )
            response = model.generate(prompts, sampling_params)
            predictions = [normalize(result.outputs[0].text) for result in response]
            if args.save_output:
                save_json(
                    [
                        
                        {
                            "prompt": prompt,
                            "question": question,
                            "ground_truth": ground_truth,
                            "prediction": prediction,
                        }
                        for prompt, question, ground_truth, prediction in zip(
                            prompts, questions, ground_truths, predictions
                        )
                    ],
                    osp.join(output_dir, f"{idx}.json"),
                )
            print(f"{prompts[0]}\n Response: {predictions[0]}\n GT:{ground_truths[0]}")
            prompts = []
            ground_truths = []
            questions = []
        if idx % 1000 == 0:
            print(f"Finished {idx}/{num_questions} questions")
    print("Accuracy: ", correct / num_questions)


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    args.LLMs_rephrase_question = True
    args.save_output = True
    main(args)
    print("")

    # args = parse_args()
    # main(args)
    # print("")
