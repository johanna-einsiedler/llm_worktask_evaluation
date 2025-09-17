import json
import os
import subprocess
from typing import (
    Any,
    Callable,
    Literal,
    ParamSpec,
    Protocol,
    TypeAlias,
    cast,
    overload,
    runtime_checkable,
)

from inspect_ai import Task, eval, task
from inspect_ai.dataset import (
    FieldSpec,
    MemoryDataset,
    Sample,
    csv_dataset,
    json_dataset,
)
from inspect_ai.model import (
    CachePolicy,
    ChatMessageAssistant,
    ChatMessageSystem,
    ChatMessageUser,
    GenerateConfigArgs,
)
from inspect_ai.scorer import (
    Score,
    Scorer,
    Target,
    accuracy,
    model_graded_fact,
    score,
    scorer,
    stderr,
)
from inspect_ai.solver import (
    Generate,
    Solver,
    TaskState,
    chain,
    generate,
    # prompt_template,
    self_critique,
    solver,
    system_message,
    user_message,
)
from inspect_ai.util import resource
import pandas as pd
import regex as re
from take_test import (
    collect_overall_scores,
    copy_answer_key,
    run_evaluation,
    save_answer_json,
    save_answer_key,
    save_evaluation,
)
from typing_extensions import Unpack

system_prompt_template = (
    """You are an expert worker within the domain of {occupation}. Complete the following exam."""
)
# with open("test_strings/exam.txt", "r", encoding="utf-8") as file:
#     test_exam = file.read()

# with open("test_strings/answer_key.txt", "r", encoding="utf-8") as file:
#     answer_key = file.read()
# with open("test_strings/grading.txt", "r", encoding="utf-8") as file:
#     grading_script = file.read()


# dataset1 = csv_dataset(
#     "basic/business_and_financial_operations_occupations_exams.csv",
#     FieldSpec(
#         input="exam",
#         id="task_id",
#         metadata=["occupation", "answer_key", "grading", "task_id"],
#     ),
# )


@solver
def custom_user_message(template: str, **params: Any) -> Solver:
    """Solver which inserts a user message into the conversation.

    User message template containing any number of optional `params`.
    for substitution using the `str.format()` method. All values
    contained in sample `metadata` and `store` are also automatically
    included in the `params`.

    Args:
      template: Template for user message.
      **params: Parameters to fill into the template.

    Returns:
      A solver that inserts the parameterised user message.
    """
    # read template
    content = resource(template)

    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # kwargs = state.metadata | state.store._data | params
        state.messages.append(ChatMessageUser(content=content))
        return state

    return solve


@scorer(metrics=[accuracy(), stderr()])
def runScoringScripts() -> Scorer:
    async def score(state: TaskState, target: Target) -> Score:
        # get model answer
        answer = state.output.completion
        model = str(state.model)
        task_id = str(state.metadata["task_id"]).replace(".", "_")
        path = os.getcwd()
        save_answer_json(answer, task_id, path, model)
        copy_answer_key(task_id, path)
        run_evaluation(task_id, path, model)
        score = collect_overall_scores(task_id, path, model)
        return Score(value=score)

    return score


@solver
def setup():
    async def solve(state, generate):
        task_id = str(state.metadata["task_id"]).replace(".", "_")
        answer_key = state.metadata["answer_key"]
        grading_script = state.metadata["grading"]
        path = os.getcwd()
        save_answer_key(answer_key, task_id, path)
        save_evaluation(grading_script, task_id, path)

        return state

    return solve


@task
def exams():
    return Task(
        dataset=dataset,
        setup=setup(),
        solver=[
            system_message(system_prompt_template),
            custom_user_message("{input}"),
            generate(),
        ],
        scorer=runScoringScripts(),
        # metadata=folder_path,
    )


def read_in_data(occupations_file_names, prefix, suffix):
    df_all = pd.DataFrame()
    for occ in occupations_file_names:
        df = pd.read_csv(prefix + occ + suffix)
        df["occupation_group"] = occ
        df_all = pd.concat([df_all, df], ignore_index=True)
    print(df_all.shape[0], " tasks in the data")
    return df_all


if __name__ == "__main__":
    models = [
        "anthropic/claude-3-7-sonnet-20250219",
        "openai/o3-2025-04-16",
        "openai-api/deepseek/deepseek-reasoner",
        "google/gemini-1.5-flash",
        "google/gemini-2.0-flash",
        "anthropic/claude-3-7-sonnet-20250219",
        "openai/gpt-4o",
        "openai/gpt-3.5-turbo-0125",
        "openai-api/deepseek/deepseek-chat",
        "google/gemini-2.5-pro-preview-03-25",
        "anthropic/claude-3-5-sonnet-20240620",
    ]
    folder_path = "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/exams/basic/gemini-2.5-pro/"
    csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]
    dataframes = []
    # # Read each CSV file into a list of DataFrames
    for csv_file in csv_files:
        print(csv_file)
        df_g = pd.read_csv(os.path.join(folder_path, csv_file))
        df_g["occupation_group"] = csv_file.split("_exam")[0]
        dataframes.append(df_g)
    combined_df = pd.concat(dataframes, ignore_index=True)[0:2]
    print(combined_df.head())

    combined_df.to_csv(os.path.join(folder_path, "all_exams.csv"))

    os.chdir(folder_path)
    print(os.getcwd())
    dataset = csv_dataset(
        "all_exams.csv",
        FieldSpec(
            input="exam",
            id="task_id",
            metadata=["occupation", "answer_key", "grading", "task_id"],
        ),
    )
    dataset = dataset.filter(lambda sample: sample.input != "")
    dataset = dataset.filter(lambda sample: sample.input != "Exam not valid")
    eval(exams(), model=models, temperature=0)
