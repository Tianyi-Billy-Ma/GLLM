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

query1 = "what was the last year where this team was a part of the usl a-league?"
query2 = "how many total points did the bombers score against the bc lions?"
query3 = "which is deeper, lake tuz or lake palas tuzla?"
query4 = "how many people stayed at least 3 years in office?"
query5 = "how many times did an italian cyclist win a round?"
query6 = "which is the fidef rst city listed alphabetically?"
query7 = "which model has the most in service?"


def generate_prompt(question, documents):
    Instruction = TASK_DICT["TableQA"]
    evidences = "\n".join(documents)
    prompt = (
        f"##{Instruction}\n\n## Input:\n\n{question}\n\n## Evidences: {evidences}\n\n"
    )
    return prompt


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

    retriever = Retriever(args)
    retriever.setup_retriever()
    # res = retriever.search_document(query1)

    tokenzier = AutoTokenizer.from_pretrained(
        args.LLMs_generator_model_name, padding_side="left"
    )
    tokenzier = add_special_token(tokenzier)
    model = LLM(
        model=args.LLMs_generator_model_name,
        download_dir=args.LLMs_dir,
        dtype=args.LLms_dtype,
        tensor_parallel_size=args.LLMs_world_size,
    )
    model.resize_token_embeddings(len(tokenzier))

    for idx, (qid, qa) in tqdm(enumerate(qas.items())):
        question, answer = qa["question"], qa["answer"]
        if qid in rephrase_files:
            rephrase_dict = load_json(osp.join(pretrain_dir, "rephrase", f"{qid}.json"))
            question = rephrase_dict["rephrased question"]
        documents = retriever.search_document(question)["hyperedge"]

        sampling_params = SamplingParams(
            temperature=0.0, top_p=1.0, max_tokens=args.LLMs_max_new_tokens
        )
        prompts = generate_prompt(question, documents)
        response = model.generate(prompts, sampling_params)
        print(response)


if __name__ == "__main__":
    args = parse_args()
    args.LLMs_reload_index = True
    main(args)
    print("")

    # args = parse_args()
    # main(args)
    # print("")
