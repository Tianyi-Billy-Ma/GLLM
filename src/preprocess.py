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
                text = "[NST] " + text + " [NED]"
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
            m = "[RST] " + m + " [RED]"
            passages.append(m)
        for idx, header in enumerate(headers):
            m = "; ".join(
                [f"{header} is {row[idx] if row[idx] else '[NULL]'}" for row in rows]
            )
            if lowercase:
                m = m.lower()
            if normalize_text:
                m = normalize(m)
            m = "[CST] " + m + " [CED]"
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
