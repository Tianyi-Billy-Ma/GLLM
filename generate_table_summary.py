from json import load
import os
import os.path as osp

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI

from src import load_pickle, save_pickle
from arguments import parse_args
import time
import numpy as np
import json


def main(args):
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    template = """Question: Please provide a detail title and a brief summary for the following table {table}. 
    Please answer in the format of a dictionary with the following keys: title and summary. 
    """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    pickle_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "all-mpnet-base-v2_info.pickle"
    )

    dict_tables = load_pickle(pickle_path)
    tables = dict_tables["tables"]
    save_dir = osp.join(args.raw_data_dir, args.dname, "table")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        files = os.listdir(save_dir)
        saved_file_index = [file.split("_")[-1].split(".")[0] for file in files]
    for idx, (key, table) in enumerate(tables.items()):
        if str(idx) in saved_file_index:
            continue
        table = {"header": table["header"], "rows": table["rows"]}
        try:
            response = llm_chain.run(table=str(table))
        except Exception as e:
            if e.http_status == 400:
                print(f"Error in generating summary for table {idx}. Error: {e}")
                print("tries to remove some rows ")
                table = {"header": table["header"], "rows": table["rows"][:100]}
                response = llm_chain.run(table=str(table))
        save_pickle(
            response,
            osp.join(save_dir, f"summary_{idx}.pickle"),
        )
        rest = np.random.choice([1, 2, 3, 4])
        time.sleep(rest)
        if idx % 100 == 0:
            print(f"Generated summaries for {idx} tables.")
    return tables


def load_raw(args):
    load_dir = osp.join(args.raw_data_dir, args.dname, "table")
    files = os.listdir(load_dir)
    data = {}
    for idx, file in enumerate(files):
        raw = load_pickle(osp.join(load_dir, file))
        raw = json.loads(raw)
        data[f"table_{idx}"] = raw
    return data


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    data = load_raw(args)
    print("")
