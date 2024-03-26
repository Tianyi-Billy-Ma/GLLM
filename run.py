from transformers import AutoTokenizer, AutoModel
from vllm import LLM
import json
from arguments import parse_args
from models.LLMs.models import Retriever


query1 = "what was the last year where this team was a part of the usl a-league?"
query2 = "how many total points did the bombers score against the bc lions?"
query3 = "which is deeper, lake tuz or lake palas tuzla?"
query4 = "how many people stayed at least 3 years in office?"
query5 = "how many times did an italian cyclist win a round?"
query6 = "which is the first city listed alphabetically?"
query7 = "which model has the most in service?"


def main(args):
    retriever = Retriever(args)
    retriever.setup_retriever()
    res = retriever.search_document(query=query6)
    print(res)


if __name__ == "__main__":
    args = parse_args()
    main(args)
    print("")
    # args = parse_args()
    # main(args)
    # print("")
