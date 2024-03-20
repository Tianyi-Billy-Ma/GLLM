from transformers import AutoTokenizer, AutoModel
from vllm import LLM
import json


def main(args):

    model_name = "facebook/contribert-msmarco"

    tokenizer = AutoTokenizer.from_pretrained("facebook/contriever-msmarco")
    model = LLM(model=model_name, download_dir=args.LLMs_dir)

    # model =


if __name__ == "__main__":

    
    # args = parse_args()
    # main(args)
    # print("")
