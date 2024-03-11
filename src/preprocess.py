from collections import defaultdict
from string import Template
from typing import Mapping
import pandas as pd
import re

########################################################################################################################
# bank
########################################################################################################################
# Use description from: https://archive.ics.uci.edu/ml/datasets/bank+marketing
# and https://www.openml.org/search?type=data&sort=runs&id=1461&status=active
bank_feature_names = [
    ("age", "age"),
    ("job", "type of job"),
    ("marital", "marital status"),
    ("education", "education"),
    ("default", "has credit in default?"),
    ("balance", "average yearly balance, in euros"),
    ("housing", "has housing loan?"),
    ("loan", "has personal loan?"),
    ("contact", "contact communication type"),
    ("day", "last contact day of the month"),
    ("month", "last contact month of year"),
    ("duration", "last contact duration, in seconds"),
    (
        "campagin",
        "number of contacts performed during this campaign and for this client",
    ),
    (
        "pdays",
        "number of days that passed by after the client was last contacted from a previous campaign",
    ),
    (
        "previous",
        "number of contacts performed before this campaign and for this client",
    ),
    ("poutcome", "outcome of the previous marketing campaign"),
]
template_config_bank = {
    "pre": {
        "age": lambda x: f"{int(x)}",
        "balance": lambda x: f"{int(x)}",
        "day": lambda x: f"{int(x)}",
        "duration": lambda x: f"{int(x)}",
        "campaign": lambda x: f"{int(x)}",
        "pdays": lambda x: (
            f"{int(x)}" if x != -1 else "client was not previously contacted"
        ),
        "previous": lambda x: f"{int(x)}",
    }
}
template_bank = " ".join(
    ["The " + v + " is ${" + k + "}." for k, v in bank_feature_names]
)
template_bank_list = "\n".join(
    ["- " + v + ": ${" + k + "}" for k, v in bank_feature_names]
)
template_bank_list_values = "\n".join(["${" + k + "}" for k, v in bank_feature_names])
bank_permutation = [14, 11, 4, 15, 8, 16, 12, 1, 6, 5, 13, 7, 9, 3, 10, 2]
template_bank_list_permuted = "\n".join(
    [
        "- " + x[1] + ": ${" + bank_feature_names[bank_permutation[i] - 1][0] + "}"
        for i, x in enumerate(bank_feature_names)
    ]
)
template_config_bank_list = template_config_bank
template_config_bank_list_permuted = template_config_bank
template_config_bank_list_values = template_config_bank_list


class MessageGenerator(Template):
    def __init__(
        self,
        args,
        prefix=None,
        suffix=None,
        default=None,
        pre=None,
        post=None,
        fns=None,
    ):
        template = eval(f"template_{args.dname}_{args.format}")
        template_config = eval(f"template_config_{args.dname}_{args.format}")
        super().__init__(template)

        self.pre = pre if pre is not None else {}
        self.post = post if post is not None else {}

        self.prefix = defaultdict(lambda: "", prefix if prefix is not None else {})
        self.suffix = defaultdict(lambda: "", suffix if suffix is not None else {})
        self.default = defaultdict(lambda: "", default if default is not None else {})

        self.fns = fns if fns is not None else []

        self.format_timestamp = lambda dt: (dt.to_pydatetime()).strftime("%B %-d, %Y")
        self.format_timedelta = lambda td: str((td.to_pytimedelta()).days)

    def clean_message(self, note):
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove all leading and trailing whitespaces
        note = re.sub(r"^[ \t]+", "", note)
        note = re.sub(r"\n[ \t]+", "\n", note)
        note = re.sub(r"[ \t]$", "", note)
        note = re.sub(r"[ \t]\n", "\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Remove repeated dots and the end of the line
        note = re.sub(r"\.+$", ".", note)
        note = re.sub(r"\.+\n", ".\n", note)
        # Remove whitespaces before colon at the end of the line
        note = re.sub(r"\s*\.$", ".", note)
        note = re.sub(r"\s*\.\n", ".\n", note)
        # Template remove all repeated whitespaces and more than double newlines
        note = re.sub(r"[ \t]+", " ", note)
        note = re.sub("\n\n\n+", "\n\n", note)
        # Remove repetitive whitespace colon sequences
        # Ignore for ... in creditg dataset
        if "... " not in note:
            note = re.sub(r"(\s*\.)+ +", ". ", note)

        return note

    def substitute(self, mapping, **kwds):
        if isinstance(mapping, pd.Series):
            mapping = mapping.to_dict()
        assert type(mapping) is dict, "The mapping must be a dictionary."
        # Pre-formatting
        mapping = {
            k: self.pre[k](mapping[k]) if k in self.pre.keys() else mapping[k]
            for k in mapping.keys()
        }
        # Remove empty string or None
        mapping = {
            k: mapping[k]
            for k in mapping.keys()
            if (mapping[k] != "" and not pd.isna(mapping[k]))
        }

        # Format special datatypes
        for k in mapping.keys():
            if isinstance(mapping[k], pd.Timestamp):
                mapping[k] = self.format_timestamp(mapping[k])
            if isinstance(mapping[k], pd.Timedelta):
                mapping[k] = self.format_timedelta(mapping[k])

        # For existing groups add prefixes and suffixes
        for k in mapping.keys():
            mapping[k] = self.prefix[k] + str(mapping[k]) + self.suffix[k]

        # Post-formatting
        mapping = {
            k: self.post[k](mapping[k]) if k in self.post.keys() else mapping[k]
            for k in mapping.keys()
        }

        # For non existing keys add defaults
        for _, _, k, _ in self.pattern.findall(self.template):
            if k not in mapping.keys():
                mapping[k] = self.default[k]

        text = super().substitute(mapping)

        for fn in self.fns:
            text = fn(text)

        return self.clean_message(text)
