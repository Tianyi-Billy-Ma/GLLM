from transformers import AutoTokenizer, RagRetriever, RagModel
import transformers
import torch
from datasets import load_dataset
from langchain import HuggingFacePipeline
import os

device = "cuda:0" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    llm = HuggingFacePipeline.from_model_id("facebook/rag-token-nq", task="", device=device)
    print("")
