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
    for y_pred, label in zip(y_preds, labels):
        y_true = label[0]
        if y_pred == y_true:
            match_count += 1
    return 100 * (match_count / len(y_preds))


def f1_score(decoded_preds, decoded_labels):
    res = []
    for prediction, answers in zip(decoded_preds, decoded_labels):
        if type(prediction) == list and type(answers) == list:
            assert len(answers) > 0, "No valid answers found."
            res.append(np.max([qa_f1_score(prediction, gt) for gt in answers]))
        else:
            res.append(np.max([qa_f1_score(prediction, gt) for gt in answers]))
    return 100 * np.mean(res)


def qa_f1_score(y_pred, y_true):
    prediction_tokens = normalize_answer(y_pred).split()
    ground_truth_tokens = normalize_answer(y_true).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


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


def match(prediction, ground_truth):
    for gt in ground_truth:
        if gt in prediction:
            return 1
    return 0
