import os
import os.path as osp
from transformers import AutoTokenizer, AutoModel
from vllm import LLM
from arguments import parse_args
from models.LLMs.models import Retriever
from src.load_data import load_pickle
from src.helper import PROMPT_DICT

query1 = "what was the last year where this team was a part of the usl a-league?"
query2 = "how many total points did the bombers score against the bc lions?"
query3 = "which is deeper, lake tuz or lake palas tuzla?"
query4 = "how many people stayed at least 3 years in office?"
query5 = "how many times did an italian cyclist win a round?"
query6 = "which is the fidef rst city listed alphabetically?"
query7 = "which model has the most in service?"


def generate():
    pass


def main(args):
    retriever = Retriever(args)
    retriever.setup_retriever()

    # tokenizer = AutoTokenizer.from_pretrained(args.LLMs_generator_model_name)
    # generator = LLM(
    #     model=args.LLMs_generator_model_name,
    #     download_dir=args.LLMs_dir,
    #     dtype=args.LLms_no_fp16,
    #     tensor_parallel_size=1,
    # )

    # table_path = osp.join(
    #     args.processed_data_dir, args.dname, "pretrain", "info.pickle"
    # )
    # input_data = load_pickle(table_path)
    # tables, qas, tname2tid, qid2tid, qid2qname = (
    #     input_data["tables"],
    #     input_data["qas"],
    #     input_data["tname2tid"],
    #     input_data["qid2tid"],
    #     input_data["qid2qname"],
    # )

    # for idx, (qid, question) in enumerate(qas.items()):
    #     results = {}
    #     prompt = PROMPT_DICT["default"].format(instruction=question)

    return True


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("")
    # args = parse_args()
    # main(args)
    # print("")
