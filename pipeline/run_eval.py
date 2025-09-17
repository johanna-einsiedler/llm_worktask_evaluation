import copy
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
from inspect_ai.analysis.beta import evals_df, samples_df

# Optionally wrap into a new in-memory dataset container
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
import numpy as np
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
        dataset=sub_dataset,
        setup=setup(),
        solver=[
            system_message(system_prompt_template),
            custom_user_message("{input}"),
            generate(),
        ],
        scorer=runScoringScripts(),
        # metadata=folder_path,
    )


if __name__ == "__main__":
    models = [
        "anthropic/claude-3-7-sonnet-20250219",
        "openai/o3-2025-04-16",
        "openai-api/deepseek/deepseek-reasoner",
        "google/gemini-1.5-flash",
        "google/gemini-2.0-flash",
        "openai/gpt-4o",
        "openai/gpt-3.5-turbo-0125",
        "openai-api/deepseek/deepseek-chat",
        "google/gemini-2.5-flash",
        "anthropic/claude-3-5-sonnet-20240620",
        "anthropic/claude-3-haiku-20240307",
        "google/gemini-2.5-pro",
        "anthropic/claude-sonnet-4-20250514",
    ]
    generating_model = "claude-3-7-sonnet-20250219"  # "o3-2025-04-16"
    folder_path = f"../data/exams/advanced/{generating_model}/"

    # occ = "management_occupations"
    occ = "computer_and_mathematical_occupations"
    # occ = "business_and_financial_operations_occupations"
    os.chdir(folder_path + "/")
    print(os.getcwd())

    df = pd.read_csv(occ + "_exams.csv")
    # check for NaN values in the 'exam' column and filter them out - inspect can't handle them
    df = df[df["exam"].notna()]
    df.to_csv((occ + "_exams.csv"))
    print(df.head())
    dataset = csv_dataset(
        occ + "_exams.csv",
        FieldSpec(
            input="exam",
            id="task_id",
            metadata=["occupation", "answer_key", "grading", "task_id"],
        ),
    )

    dataset = dataset.filter(lambda sample: sample.input != "")
    dataset = dataset.filter(lambda sample: sample.input != "Exam not valid")
    samples_list = list(dataset)  # Fully materialize as list of Sample objects
    copied_list = copy.deepcopy(samples_list)  # Deep copy the Sample list

    for model in models:
        print(model)
        results_path = f"../../../../logs/advanced/{generating_model}/{occ}/"

        if os.path.isdir(results_path) and os.listdir(results_path):
            test_results = samples_df(results_path)
            ids_taken = test_results.loc[test_results["model_usage"].str.contains(model), "id"]
            print("Exams already evaluated: " + str(len(ids_taken.unique())))
            # sub_dataset = dataset.load()
            sub_dataset = MemoryDataset(copied_list)
            sub_dataset = sub_dataset.filter(lambda sample: sample.id not in ids_taken.tolist())
        else:
            print("No previous test results found — using full dataset.")
            sub_dataset = MemoryDataset(copied_list)

            eval(
                exams(),
                model=model,
                temperature=0,
                log_dir=f"../../../../logs/advanced/{generating_model}/" + occ + "/",
            )
        # except:
        #     print("no more exams left")
        #     continue
