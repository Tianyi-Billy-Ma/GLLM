PROMPT = {
    "meta-llama/Meta-Llama-3-8B-Instruct": (
        """
<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n
{instruction}<|eot_id|><|start_header_id|>user<|end_header_id|>\n
{user_input}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\nResponse:<answer>
"""
    ),
    "meta-llama/Llama-2-7b-hf": (
        """
<s>[INST]<<SYS>>
Instruction: {instruction}
<</SYS>>

{user_input}

[/INST]

Response:
"""
    ),
}

TASK_DICT = {
    "RAGTableQA": "Answer the question between <question> tags in <answer> tags without explaination according the documents provided in <context> tags."
    "In <context> tags, element in [ROW] tags is a row in a table and element in [COL] tags is a column in a table.",
    "TableQA": "Answer the question between <question> tags in <answer> tags without explaination according the table provided in <context> tags.",
    "Summary_Relevent": "Given the question between <question> tags and a summary of table in <context> tags,"
    "rate whether the detailed table is helpful to the question in <question> tags in <answer> tags, from 1 (lowest) - 5 (highest).\n\n"
    "The detailed criterion is as follows:\n"
    "5: The detailed table is mostly relevant to the question, providing all the necessary information to answer the question.\n"
    "4: The detailed table is highly relevant to the question, while there can be some minor risk that it is irrelevant to the question.\n"
    "3: The detailed table might be relevant to the question, but some major risk that it is irrelevant to the question.\n"
    "2: The detailed table is highly possbile to irrelevant to the question.\n"
    "1: The detailed table is barely on-topic or completely irrelevant to the question.\n"
    "\nExamples:\n"
    "Question: Who is the current president of the United States as of 2023?\n"
    "Table Summary: The table contains the list of presidents of the United States from 2020 to 2024.\n"
    "Response: 5\n"
    "Explanation: The table provides the information about the president of the United States from 2020 to 2024, and as 2023 is in the range, the table is mostly relevant to the question.\n\n"
    "Question: What is the longest lake in United States?\n"
    "Table Summary: The table contains the basic information about lakes in the world.\n"
    "Response: 4\n"
    "Explanation: The table is highly relevant to the question, as it contains the basic information about lakes in the world."
    "However, there is a risk that the length of the lakes in not provided in the table\n\n"
    "Question: What is the capital of France?\n"
    "Table Summary: The table contains the list of capital cities of Asian countries.\n"
    "Response: 1\n"
    "Explanation: The table is irrelevant to the question, as it contains the list of capital cities of Asian countries, not European countries.\n\n",
}


QUESTION_PROMPT = {
    "RAGTableQA": "<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>\n\n",
    "TableQA": "<context>\n{context}\n</context>\n\n<question>\n{question}\n</question>\n\n",
}


def generate_prompt(args, question, context):
    model = args.LLMs_generator_model_name
    task = args.task_mode
    assert model in PROMPT, f"Model {model} not supported"
    assert task in TASK_DICT, f"Task {task} not supported"
    assert task in QUESTION_PROMPT, f"Task {task} not supported"
    instruction = TASK_DICT[task]
    question = QUESTION_PROMPT[task].format(
        question=question, context="\n".join([value for key, value in context.items()])
    )
    prompt = PROMPT[model].format(instruction=instruction, user_input=question)
    return prompt
