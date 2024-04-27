import numpy as np
import re
from collections import Counter
import string


def exact_match_score(y_pred, y_true, normalize=False):
    if normalize:
        res = normalize_answer(y_pred) == normalize_answer(y_true)
    else:
        res = y_pred == y_true
    return res


def acc_score(y_preds, labels):
    assert len(y_preds) == len(
        labels
    ), "Length of predictions and labels should be same."
    match_count = 0
    for y_pred, y_true in zip(y_preds, labels):
        if y_true in y_pred:
            match_count += 1
    return round(100 * (match_count / len(y_preds)), 2)


def f1_score(decoded_preds, decoded_labels):
    res = []
    for pred, label in zip(decoded_preds, decoded_labels):
        res.append(match(pred, label, True))
    num_same = sum(res)
    precision = 1.0 * num_same / len(decoded_preds)
    recall = 1.0 * num_same / len(decoded_labels)
    f1 = (2 * precision * recall) / (precision + recall)
    return round(f1 * 100, 2)


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def find_entity_tags(sentence):
    entity_regex = r"(.+?)(?=\s<|$)"
    tag_regex = r"<(.+?)>"
    entity_names = re.findall(entity_regex, sentence)
    tags = re.findall(tag_regex, sentence)

    results = {}
    for entity, tag in zip(entity_names, tags):
        if "<" in entity:
            results[entity.split("> ")[1]] = tag
        else:
            results[entity] = tag
    return results


def match(prediction, ground_truth, normalize=False):
    if normalize:
        prediction = normalize_answer(prediction)
        ground_truth = normalize_answer(ground_truth)
    if ground_truth in prediction:
        return 1
    else:
        return 0
