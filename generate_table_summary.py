from json import load
from multiprocessing.forkserver import MAXFDS_TO_SEND
import os
import os.path as osp

from src import load_pickle, save_pickle, save_json, load_json
from arguments import parse_args
import time
import numpy as np
import json

from openai import OpenAI


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
    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    template = """Question: Please provide a detail title and a brief summary for the following table {table}. 
    Please answer in the format of a dictionary with the following keys: title and summary. 
    """

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
        try:
            tabletext = format_table(table)
            user_input = template.format_map({"table": tabletext})
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=200,
            )
        except Exception as e:
            if e.http_status == 400:
                print(f"Error in generating summary for table {idx}. Error: {e}")
                print("tries to remove some rows ")
                table = {"header": table["header"], "rows": table["rows"][:100]}
                tabletext = format_table(table)
                user_input = template.format_map({"table": tabletext})
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": user_input}],
                    max_tokens=200,
                )
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
    # template = """Question: Please provide a detail title and a brief summary for the following table {table}.
    # Please answer in the format of a dictionary with the following keys: title and summary.
    # """

    template = """## Instruction: Please check whether the question is accurate for table. 
    If not, please rephrase the question so that it is accurate to the table.
    For example, the question: how many people stayed at least 3 years in office? is not accurate for table: 
    "header": [ "", "Name", "Took office", "Left office", "Party", "Notes/Events" ], 
    "rows": [ [ "11", "William McCreery", "March 4, 1803", "March 3, 1809", "Democratic Republican", "" ], [ "12", "Alexander McKim", "March 4, 1809", "March 3, 1815", "Democratic Republican", "" ], [ "13", "William Pinkney", "March 4, 1815", "April 18, 1816", "Democratic Republican", "Resigned to accept position as Minister Plenipotentiary to Russia" ], [ "14", "Peter Little", "September 2, 1816", "March 3, 1823", "Democratic Republican", "" ], [ "14", "Peter Little", "March 4, 1823", "March 3, 1825", "Jacksonian DR", "" ], [ "14", "Peter Little", "March 4, 1825", "March 3, 1829", "Adams", "" ], [ "15", "Benjamin C. Howard", "March 4, 1829", "March 3, 1833", "Jacksonian", "" ] ]
    Because the question is too broad, and the more accurate question would be: how many people stayed at least 3 years in office as mayors of Baltimore from 1803 to 1833.
    
    ## Question: {question}
    
    ## Table:
    
    Title: {title}
    {table}
    
    Please answer in the dictionary format with the following keys: origianl question and rephrased question.
    """

    client = OpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
    )

    table_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "plaintext_tables.pickle"
    )

    data = load_pickle(table_path)

    tables, qas = (
        data["tables"],
        data["qas"],
    )

    qid2tid = load_pickle(
        osp.join(args.processed_data_dir, args.dname, "pretrain", "mapping.pickle")
    )["qid2tid"]

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
        try:
            user_input = template.format_map(
                {"question": question, "table": tabletext, "title": title}
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
                max_tokens=200,
            )
        except Exception as e:
            print(e)
            print(f"Error in generating summary for table {qid}. Error: {e}")
            print("tries to remove some rows ")
            table = {"header": table["header"], "rows": table["rows"][:100]}
            tabletext = format_table(table)
            user_input = template.format_map(
                {"question": question, "table": tabletext, "title": title}
            )
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": user_input}],
            )
        try:
            response = response.choices[0].message.content
            response = json.loads(response)
            save_json(
                response,
                osp.join(save_dir, f"{qid}.json"),
            )
        except Exception as e:
            print(f"Error in generating summary for table {qid}. Error: {e}")
            save_pickle(response, osp.join(save_dir, f"{qid}.pickle"))
            continue
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


def process_questions(args):
    raw_question_dir = osp.join(args.raw_data_dir, args.dname, "rephrase")
    save_dir = osp.join(args.processed_data_dir, args.dname, "pretrain")
    save_path = osp.join(save_dir, "plaintext_rephrased_questions.pickle")
    files = os.listdir(raw_question_dir)

    table_path = osp.join(
        args.processed_data_dir, args.dname, "pretrain", "plaintext_tables.pickle"
    )

    data = load_pickle(table_path)

    tables, qas = (
        data["tables"],
        data["qas"],
    )
    rephrased_questions = {}
    for idx, (qid, qa) in enumerate(qas.items()):
        if f"{qid}.json" in files:
            response = load_json(osp.join(raw_question_dir, f"{qid}.json"))
        elif f"{qid}.pickle" in files:
            response = load_pickle(osp.join(raw_question_dir, f"{qid}.pickle"))
            element1 = response.split("\n")[1]
            element2 = response.split("\n")[2]
            if '"' in element1 and '"' in element2:
                while element2[-1] != '"':
                    element2 = element2[:-1]

                def replace_special_char(element):
                    indices = [i for i in range(len(element)) if element[i] == '"']
                    indices = indices[:3] + [indices[-1]]
                    return "".join(
                        [
                            char
                            if (char != '"') or (char == '"' and i in indices)
                            else "'"
                            for i, char in enumerate(element)
                        ]
                    )

                element1 = replace_special_char(element1)
                element2 = replace_special_char(element2)
                modified_response = "{" + element1 + element2 + "}"
            else:

                def add_special_char(v1):
                    index1 = v1.lower().find("original")
                    if index1 == -1:
                        index1 = v1.lower().find("rephrased")
                    index2 = v1.lower().find("question")
                    index3 = v1.find(":")
                    index4 = len(v1)
                    assert index1 != -1 and index2 != -1 and index3 != -1, "Error"
                    return (
                        v1[:index1]
                        + '"'
                        + v1[index1:index3]
                        + '":"'
                        + v1[index3 + 1 : index4]
                        + '"'
                    )

                element1 = add_special_char(element1)
                element2 = add_special_char(element2)
                modified_response = "{" + element1 + "," + element2 + "}"
            response = json.loads(modified_response)
        else:
            print(f"File {qid} not found.")
            continue
        if (
            "rephrased question" in response.keys()
            and "original question" in response.keys()
        ):
            rephrased_question = response["rephrased question"]
        elif (
            "rephrased_question" in response.keys()
            and "original_question" in response.keys()
        ):
            rephrased_question = response["rephrased_question"]
        elif (
            "original_question" in response.keys()
            and "accurate_question" in response.keys()
        ):
            rephrased_question = response["accurate_question"]
        elif (
            "Origianl question" in response.keys()
            and "Rephrased question" in response.keys()
        ):
            rephrased_question = response["Rephrased question"]
        elif (
            "Original Question" in response.keys()
            and "Rephrased Question" in response.keys()
        ):
            rephrased_question = response["Rephrased Question"]
        else:
            print(f"File {qid} is not in good format")
            break
        rephrased_questions[qid] = rephrased_question

    save_pickle(rephrased_questions, save_path)


if __name__ == "__main__":
    args = parse_args()
    # main(args)
    # data = load_raw(args)
    # rephrase_questions(args)
    process_questions(args)
    print("")
