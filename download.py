from vllm import LLM
import os
import os.path as osp

root_dir = os.getcwd()
model_dir = osp.join(root_dir, "models", "LLMs")
model = LLM(
        model="meta-llama/Meta-Llama-3-8B-Instruct",
        download_dir=model_dir,
        dtype="half",
        tensor_parallel_size=1,
    )