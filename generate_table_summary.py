from json import load
import os
import os.path as osp

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain_openai import ChatOpenAI

from src import load_pickle, save_pickle, save_json
from arguments import parse_args
import time
import numpy as np
import json


def format_table(table):
    headers = table["header"]
    rows = table["rows"]

    text = "|"
    text += "|".join(headers) + "\n"
    for row in rows:
        text += "|"
        text += "|".join(row) + "\n"
    return text


def generate_summary(args):
    llm = OpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    template = """Question: Please provide a detail title and a brief summary for the following table {table}. 
    Please answer in the format of a dictionary with the following keys: title and summary. 
    """

    prompt = PromptTemplate.from_template(template)

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    pickle_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "info.pickle"
    )

    dict_tables = load_pickle(pickle_path)
    tables = dict_tables["tables"]
    save_dir = osp.join(args.raw_data_dir, args.dname, "summary")
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


def rephrase_questions(args):
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.3)

    # template = """Question: Please provide a detail title and a brief summary for the following table {table}.
    # Please answer in the format of a dictionary with the following keys: title and summary.
    # """

    template = """Question: Please check whether the question {question} is accurate for table {table} with title {title}. 
    If not, please rephrase the question so that it is accurate to the table.
    For example, the question: how many people stayed at least 3 years in office? is not accurate for table: 
    "header": [ "", "Name", "Took office", "Left office", "Party", "Notes/Events" ], 
    "rows": [ [ "11", "William McCreery", "March 4, 1803", "March 3, 1809", "Democratic Republican", "" ], [ "12", "Alexander McKim", "March 4, 1809", "March 3, 1815", "Democratic Republican", "" ], [ "13", "William Pinkney", "March 4, 1815", "April 18, 1816", "Democratic Republican", "Resigned to accept position as Minister Plenipotentiary to Russia" ], [ "14", "Peter Little", "September 2, 1816", "March 3, 1823", "Democratic Republican", "" ], [ "14", "Peter Little", "March 4, 1823", "March 3, 1825", "Jacksonian DR", "" ], [ "14", "Peter Little", "March 4, 1825", "March 3, 1829", "Adams", "" ], [ "15", "Benjamin C. Howard", "March 4, 1829", "March 3, 1833", "Jacksonian", "" ] ]
    Because the question is too broad, and the more accurate question would be: how many people stayed at least 3 years in office as mayors of Baltimore from 1803 to 1833.
    Another example, the question: "Which lake in Turkey is deeper, Lake Tuz or Lake Palas Tuzla?"
    Please answer in the dictionary format with the following keys: origianl question, need to rephrase? and rephrased question.
    """

    prompt = PromptTemplate(
        template=template, input_variables=["table", "title", "question"]
    )

    llm_chain = LLMChain(llm=llm, prompt=prompt)

    pickle_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "info.pickle"
    )

    data = load_pickle(pickle_path)

    tables, qas, tname2tid, qid2tid, qid2qname, n2tid = (
        data["tables"],
        data["qas"],
        data["tname2tid"],
        data["qid2tid"],
        data["qid2qname"],
        data["n2tid"],
    )

    save_dir = osp.join(args.raw_data_dir, args.dname, "rephrase")
    if not osp.exists(save_dir):
        os.makedirs(save_dir)
        saved_qid = []
    else:
        files = os.listdir(save_dir)
        saved_qid = [file.split(".")[0] for file in files]
    for idx, (qid, qa) in enumerate(qas.items()):
        if qid in saved_qid:
            continue
        table = tables[qid2tid[qid]]
        title = table["title"][6:-6]  # remove the <title> tag
        tabletext = format_table(table)
        question = qa["question"]
        response = llm_chain.run(table=tabletext, title=title, question=question)

        try:
            response = llm_chain.run(table=str(table), title=title, question=question)
        except Exception as e:
            if e.http_status == 400:
                print(f"Error in generating summary for table {qid}. Error: {e}")
                print("tries to remove some rows ")
                table = {"header": table["header"], "rows": table["rows"][:100]}
                response = llm_chain.run(
                    table=str(table), title=title, question=question
                )
        try:
            response = json.loads(response)
        except Exception as e:
            print(f"Error in generating summary for table {qid}. Error: {e}")
            continue
        save_json(
            response,
            osp.join(save_dir, f"{qid}.json"),
        )
        rest = np.random.choice([1, 2, 3, 4])
        time.sleep(rest)
        if idx % 100 == 0:
            print(f"Generated response for {idx} tables.")
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
    # data = load_raw(args)
    rephrase_questions(args)
    print("")
