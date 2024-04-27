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


def generate_plaintext_from_table(tables):
    for table in tables:
        header = table["header"]
        rows = table["rows"]
    return "\n".join(
        [f"{header[i]}: {cell}" for row in rows for i, cell in enumerate(row)]
    )


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
