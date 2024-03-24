from transformers import AutoTokenizer, AutoModel
from vllm import LLM
import json
from arguments import parse_args
from models.LLMs.models import Retriever


def main(args):

    retriever = Retriever(args)
    retriever.setup_retriever()
    res = retriever.search_document(
        query="what was the last year where this team was a part of the usl a-league?"
    )
    print(res)


if __name__ == "__main__":

    args = parse_args()
    main(args)
    print("")
    # args = parse_args()
    # main(args)
    # print("")
