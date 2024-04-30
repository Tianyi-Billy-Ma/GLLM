from collections import defaultdict
from string import Template
from typing import Mapping
import pandas as pd
import re
from .normalize_text import normalize


def generate_node_plaintext_within_tables(tables, lowercase=True, normalize_text=True):
    def generate_node_plaintext_from_table(table):
        header = table["header"]
        rows = table["rows"]
        res = []
        for row in rows:
            for i, cell in enumerate(row):
                text = f"{header[i]} is {cell if cell else '[NULL]'}"
                if lowercase:
                    text = text.lower()
                if normalize_text:
                    text = normalize(text)
                text = "[NODE] " + text + " [/NODE]"
                res.append(text)
        return res

    return {tname: generate_node_plaintext_from_table(t) for tname, t in tables.items()}


def generate_hyperedges_plaintext_from_tables(
    tables, lowercase=True, normalize_text=True
):
    def generate_hyperedges_plaintext_from_table(table):
        passages = []
        headers = table["header"]
        rows = table["rows"]
        for row in rows:
            m = "; ".join(
                [
                    f"{headers[idx]} is {cell if cell else '[NULL]'}"
                    for idx, cell in enumerate(row)
                ]
            )
            if lowercase:
                m = m.lower()
            if normalize_text:
                m = normalize(m)
            m = "[ROW] " + m + " [/ROW]"
            passages.append(m)
        for idx, header in enumerate(headers):
            m = "; ".join(
                [f"{header} is {row[idx] if row[idx] else '[NULL]'}" for row in rows]
            )
            if lowercase:
                m = m.lower()
            if normalize_text:
                m = normalize(m)
            m = "[COL] " + m + " [/COL]"
            passages.append(m)
        return passages

    return {
        tname: generate_hyperedges_plaintext_from_table(t)
        for tname, t in tables.items()
    }


def generate_plaintext_from_table(table, args=None):
    header = table["header"]
    rows = table["rows"]
    res = ""
    if args == None:
        res = (
            "| "
            + " | ".join(
                [normalize(head).replace("\n", " ") for head in table["header"]]
            )
            + "|\n"
        )
        res += "| " + " | ".join(["---" for _ in table["header"]]) + "|\n"
        for row in table["rows"]:
            res += (
                "| "
                + " |".join(normalize(element).replace("\n", " ") for element in row)
                + "|\n"
            )
    else:
        if "md" in args.LLMs_table_plaintext_format:
            res = generate_plaintext_from_table(table)

        elif "dict" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    "; ".join([f"{header[i]}: {cell}" for i, cell in enumerate(row)])
                    + "\n"
                )
        elif "html" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    " ".join([f"<{header[i]}>{cell}" for i, cell in enumerate(row)])
                    + "\n"
                )
        elif "sentence" in args.LLMs_table_plaintext_format:
            for row in rows:
                res += (
                    ". ".join([f"{header[i]} is {cell}" for i, cell in enumerate(row)])
                    + "\n"
                )

        if "summary" in args.LLMs_table_plaintext_format:
            if (
                args.LLMs_pretrain_include_tags
                and (not table["summary"].startswith("[SUMMARY]"))
                and (not table["summary"].endswith("[/SUMMARY]"))
            ):
                res = "[SUMMARY]" + table["summary"] + "[/SUMMARY]" + "\n" + res
            elif (
                not args.LLMs_pretrain_include_tags
                and table["summary"].startswith("[SUMMARY]")
                and table["summary"].endswith("[/SUMMARY]")
            ):
                res = (
                    "Table Summary:"
                    + table["summary"]
                    .replace("[SUMMARY]", "")
                    .replace("[/SUMMARY]", "")
                    + "\n"
                    + res
                )
            else:
                raise ValueError(
                    "Table summary should be enclosed with [SUMMARY] and [/SUMMARY] or not enclosed with [SUMMARY] and [/SUMMARY]"
                )
        if "title" in args.LLMs_table_plaintext_format:
            if args.LLMs_pretrain_include_tags and (
                not table["title"].startswith("[TITLE]")
                and not table["title"].endswith("[/TITLE]")
            ):
                res = "[TITLE]" + table["title"] + "[/TITLE]" + "\n" + res
            elif (
                not args.LLMs_pretrain_include_tags
                and table["title"].startswith("[TITLE]")
                and table["title"].endswith("[/TITLE]")
            ):
                res = (
                    "Table Title:"
                    + table["title"].replace("[TITLE]", "").replace("[/TITLE]", "")
                    + "\n"
                    + res
                )
            else:
                raise ValueError(
                    "Table title should be enclosed with [TITLE] and [/TITLE] or not enclosed with [TITLE] and [/TITLE]"
                )
    return res


node_token_names = ["[NODE]", "[/NODE]"]
row_token_names = ["[ROW]", "[/ROW]"]
col_token_names = ["[COL]", "[/COL]"]
title_token_names = ["[TITLE]", "[/TITLE]"]
summary_token_names = ["[SUMMARY]", "[/SUMMARY]"]
null_token_name = "[NULL]"


def add_special_token(tokenizer):
    tokenizer.add_special_tokens(
        special_tokens_dict={
            "additional_special_tokens": node_token_names
            + row_token_names
            + col_token_names
            + title_token_names
            + summary_token_names
            + [null_token_name]
        }
    )
    return tokenizer
