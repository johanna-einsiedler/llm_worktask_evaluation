#### Note uses python 3.9 environment (newenv)
import ast
import json
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Annotated, TypedDict

import anthropic
import numpy as np
import pandas as pd
import regex as re
from dotenv import find_dotenv, load_dotenv
from IPython.display import Image, display
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # or your equivalent import
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from openai import OpenAI
from typing_extensions import TypedDict

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
from query_agents import *

#######################################
# Helper functions


def safe_eval(value, default=[]):
    """
    Safely evaluates a string as a Python literal.

    Parameters:
        value (any): The value to evaluate. Typically a string representing a Python literal.
        default (any): The default value to return if evaluation fails or value is NaN. Defaults to an empty list.

    Returns:
        any: The evaluated Python object, or the default value if evaluation fails.
    """
    if pd.isna(value):
        return default
    try:
        return ast.literal_eval(value)
    except:
        return default


def join_items(items, conj="and"):
    """
    Joins a list of strings into a human-readable string with commas and a conjunction.

    Parameters:
        items (list of str): The list of string items to join.
        conj (str): The conjunction to use before the last item. Defaults to 'and'.

    Returns:
        str: A string of items joined by commas and the conjunction.

    Examples:
        join_items(['apples']) -> "apples"
        join_items(['apples', 'oranges']) -> "apples and oranges"
        join_items(['apples', 'bananas', 'oranges']) -> "apples, bananas and oranges"
    """
    if len(items) == 1:
        return items[0]
    if len(items) > 1:
        return ", ".join(items[:-1]) + f" {conj} " + items[-1]
    return ""


def build_system_prompt(
    occupation,
    task_description,
    task_id,
    required_tools,
    required_materials,
    level,
    education,
    template,
):
    """
    Constructs a system prompt by filling a template with task-specific context and constraints.

    Args:
        occupation (str): The occupation relevant to the task (e.g., "graphic designer").
        task_description (str): A brief description of the task to be completed.
        task_id (str): A unique identifier for the task.
        required_tools (list): A list of tools the candidate has access to (e.g., ["Photoshop", "Excel"]).
        required_materials (list): A list of digital materials the candidate may use (e.g., ["PDF", "video"]).
        level (str): The difficulty level or expected skill level for the task.
        template (str): A string template containing placeholders for all relevant fields.

    Returns:
        str: A fully formatted system prompt string incorporating all inputs and constraints.
    """

    # Tools
    if required_tools:
        tools_instructions = (
            f"- The candidate has access to a computer with the following tools: "
            f"{join_items(required_tools, conj='and')}"
        )
    else:
        tools_instructions = (
            "- The candidate does not have access to any special tools."
        )
    # Materials
    if required_materials:
        materials_instructions = (
            f"- The candidate can also be given digital materials such as "
            f"{join_items(required_materials, conj='or')} that must be used for the test."
        )
    else:
        materials_instructions = (
            "- The candidate does not have access to any additional digital materials."
        )

    return template.format(
        occupation=occupation,
        task_description=task_description,
        task_id=task_id,
        tools_instructions=tools_instructions,
        materials_instructions=materials_instructions,
        level=level,
        education=education,
    )


def extract_and_save_python_script(
    script_text: str, folder: str, filename: str = "task_evaluation.py"
):
    """Finds Python code enclosed in triple backticks ```python ...``` and saves it to file.
    useful for extractign grading script
    """
    match = re.search(r"```python(.*?)```", script_text, re.DOTALL)
    if not match:
        raise ValueError("No ```python ... ``` code block found in the grading text.")
    code = match.group(1).strip()

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


def extract_and_save_json(
    json_text: str, folder: str, filename: str = "answer_key.json"
):
    """
    Finds JSON enclosed in triple backticks ```json ...``` and saves it to a file.
    """
    match = re.search(r"```json(.*?)```", json_text, re.DOTALL)
    if not match:
        raise ValueError("No ```json ... ``` block found in the evaluation text.")
    json_str = match.group(1).strip()

    data = json.loads(json_str)  # parse the JSON
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    return data


#################
# Define Exam State class
#####################


class ExamState(TypedDict):
    occupation: str
    task_id: str
    task_description: str
    exam_author_model: str

    # Tools and materials
    tools: str
    materials: str
    # level for exam (basic or dvanacned)
    level: str
    exam: dict
    system_prompt: str
    overview: str
    instructions: str
    materials_evaluator: str
    materials_candidate: str
    submission: str
    evaluation: str
    grading: str
    answer_key: str
    errors: list
    # Boolean flags for validation checks
    check_real_materials: bool
    check_no_internet: bool
    check_candidate_materials: bool
    check_consistency: bool
    alter_target: str
    # Key grade and count how many times below threshold
    key_grade_threshold: float
    key_grade: float
    check_overall_makes_sense: bool
    explanation_overall_makes_sense: str
    metadata: dict
    education: str
    check_answer_key: bool
    check_answer_coverage: bool
    check_feasible: bool
    explanation_feasible: str
    sequence: list
    counter: int


###################################
# Nodes
####################################


def node_system_prompt(state: ExamState) -> ExamState:
    """
    Reads a system prompt template from file, populates it using task-specific details from the state,
    and updates the state with the generated system prompt.

    Args:
        state (ExamState): A dictionary-like object containing exam-related fields such as
                           'occupation', 'task_description', 'task_id', 'tools', 'materials', and 'level'.

    Returns:
        ExamState: The updated state with an added 'system_prompt' key containing the compiled prompt string.
    """

    with open("../prompts/exam_generation_prompts/system_prompt.txt", "r") as file:
        system_prompt_template = file.read()
    print("compiling system prompt")
    state["system_prompt"] = build_system_prompt(
        state["occupation"],
        state["task_description"],
        state["task_id"],
        state["tools"],
        state["materials"],
        state["level"],
        state["education"],
        template=system_prompt_template,
    )
    state["sequence"].append("system_prompt")
    state["counter"] = state["counter"] + 1

    os.makedirs(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/",
        exist_ok=True,
    )

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_system_prompt.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["system_prompt"]))
    return state


def node_overview(state: ExamState) -> ExamState:
    """
    Generates an overview of the exam by querying a language model using the system prompt and a predefined overview prompt.

    Args:
        state (ExamState): A dictionary-like object containing fields such as 'system_prompt',
                           'exam_author_model', and a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'overview': the generated exam overview text.
            - 'metadata["overview"]': usage metadata returned by the model.
    """
    with open("../prompts/exam_generation_prompts/prompt_overview.txt", "r") as file:
        prompt_overview = file.read()
    print("creating exam overview")

    content, metadata = query_agent(
        state["system_prompt"], prompt_overview, state["exam_author_model"]
    )
    state["overview"] = content
    state["metadata"]["overview"] = metadata
    state["sequence"].append("overview")
    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_overview.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_overview))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_overview.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["overview"]))
    return state


def node_instructions(state: ExamState) -> ExamState:
    """
    Generates detailed exam instructions by querying a language model with a formatted instruction prompt.

    The prompt is constructed using a template that incorporates the previously generated exam overview.
    The result and associated metadata are stored in the state.

    Args:
        state (ExamState): A dictionary-like object containing keys such as 'overview', 'system_prompt',
                           'exam_author_model', and a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'instructions': the generated exam instructions text.
            - 'metadata["instructions"]': usage metadata returned by the model.
    """
    print("creating exam instructions")
    with open(
        "../prompts/exam_generation_prompts/prompt_instructions.txt", "r"
    ) as file:
        prompt_template_instructions = file.read()
    prompt = prompt_template_instructions.format(overview=state["overview"])
    content, metadata = query_agent(
        state["system_prompt"], prompt, state["exam_author_model"]
    )
    state["instructions"] = content
    state["metadata"]["instructions"] = metadata
    state["sequence"].append("instructions")

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_instructions.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_instructions.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["instructions"]))
    return state


def node_materials(state: ExamState) -> ExamState:
    """
    Generates exam materials using a language model, based on the exam overview and instructions.

    The function formats a prompt using a predefined template, queries the model, and updates the state with:
    - The full generated materials content.
    - A specific candidate-facing subset extracted from the response (if available).
    - Related usage metadata.

    If the candidate-facing materials cannot be extracted, it logs the issue and tracks failures.

    Args:
        state (ExamState): A dictionary-like object containing 'overview', 'instructions', 'system_prompt',
                           'exam_author_model', and a 'metadata' dictionary. Also uses 'failed_candidate_materials' for error tracking.

    Returns:
        ExamState: The updated state with:
            - 'materials_all': the complete generated materials content.
            - 'materials_candidate': extracted materials for the candidate or a fallback message.
            - 'metadata["materials"]': usage metadata from the model.
            - 'failed_candidate_materials': incremented if extraction fails.
    """
    print("creating exam materials")
    with open("../prompts/exam_generation_prompts/prompt_materials.txt", "r") as file:
        prompt_template_materials = file.read()

    prompt = prompt_template_materials.format(
        answer_overview=state["overview"], answer_instructions=state["instructions"]
    )
    content, metadata = query_agent(
        state["system_prompt"], prompt, state["exam_author_model"]
    )
    state["metadata"]["materials"] = metadata
    try:
        match = re.search(
            r"^(.*?)<MATERIALS_FOR_CANDIDATE>(.*?)</MATERIALS_FOR_CANDIDATE>(.*)$",
            content,
            re.DOTALL,
        )

        if match:
            state["materials_evaluator"] = match.group(1) + match.group(
                3
            )  # outside text
            state["materials_candidate"] = match.group(2)  # inside text
            state["check_candidate_materials"] = True

    except:
        if state["materials_evaluator"] == "No material required":
            state["materials_candidate"] = "No material required"
            state["check_candidate_materials"] = True

        else:
            state["materials_candidate"] = "Not extracted"
            state["check_candidate_materials"] = False

            print("materials candidate was not able to be extracted")
    state["sequence"].append("materials" + str(state["check_candidate_materials"]))

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_materials.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_materials_candidate.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["materials_candidate"]))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_materials_evaluator.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["materials_evaluator"]))
    return state


def node_check_materials_fake_image(state: ExamState) -> ExamState:
    """
    Performs a sanity check to detect whether the exam materials include potentially fake or AI-generated images.

    The function loads a prompt from file and queries a language model using the current instructions and materials.
    Based on the model's response ("Y" for yes), it updates a flag in the state indicating whether the materials
    are considered authentic.

    Args:
        state (ExamState): A dictionary-like object containing 'instructions', 'materials_all',
                           'exam_author_model', and a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'check_real_materials': Boolean flag indicating the model's assessment of the materials' authenticity.
            - 'metadata["check_materials"]': usage metadata from the model.
    """
    print("checking for fake images")
    with open("../prompts/sanity_check_prompts/prompt_fake_images.txt", "r") as file:
        prompt_check_fake_image = file.read()
    content, metadata = query_agent(
        prompt_check_fake_image,
        state["instructions"] + state["materials_candidate"],
        state["exam_author_model"],
    )
    state["metadata"]["check_materials"] = metadata
    if content == "Y":
        state["check_real_materials"] = False
    else:
        state["check_real_materials"] = True
    state["sequence"].append(
        "check real materials " + str(state["check_real_materials"])
    )
    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_materials_fake_image.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_check_fake_image))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_materials_fake_image.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["check_real_materials"]))

    return state


def node_check_materials_fake_website(state: ExamState) -> ExamState:
    """
    Performs a sanity check to detect whether the exam materials reference fake or unreliable websites.

    The function loads a prompt from file and queries a language model using the current instructions and materials.
    Based on the model's response ("Y" for yes), it updates a flag in the state indicating whether the materials
    contain potentially fake website references.

    Args:
        state (ExamState): A dictionary-like object containing 'instructions', 'materials_all',
                           'exam_author_model', and a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'check_no_internet': Boolean flag indicating whether the materials contain legitimate websites.
            - 'metadata["check_website"]': usage metadata from the model.
    """

    print("checking for fake websites")
    with open("../prompts/sanity_check_prompts/prompt_fake_websites.txt", "r") as file:
        prompt_check_fake_website = file.read()

    content, metadata = query_agent(
        prompt_check_fake_website,
        state["instructions"] + state["materials_candidate"],
        state["exam_author_model"],
    )
    state["metadata"]["check_website"] = metadata

    if content == "Y":
        state["check_no_internet"] = False
    else:
        state["check_no_internet"] = True
    state["sequence"].append("check no internet " + str(state["check_no_internet"]))

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_materials_fake_website.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_check_fake_website))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_materials_fake_website.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["check_no_internet"]))
    return state


def node_check_consistency(state: ExamState) -> ExamState:
    """
    Checks whether exam instructions and materials are consistent and aligned.
    Uses a language model to return structured JSON feedback.

    Expected model outputs:


    Updates the state with:

          - metadata["check_instructions_materials"]: model usage metadata
      - alter_target: str, if inconsistent (instructions or materials)
    """
    print("Checking consistency between instructions and materials...")
    with open("../prompts/sanity_check_prompts/prompt_consistency.txt", "r") as file:
        prompt_check_consistency = file.read()

    # Call the exam author model
    content, metadata = query_agent(
        prompt_check_consistency,
        state["instructions"] + state["materials_candidate"],
        state["exam_author_model"],
    )
    # Parse JSON safely

    state["metadata"]["check_consistent"] = metadata
    response = content.strip()

    try:
        result = json.loads(response)
        state["check_consistent"] = bool(result.get("consistent", False))
        state["alter_target"] = str(result.get("alter", ""))
    except:
        try:
            text = re.search(r"```json(.*?)```", response, re.DOTALL).group(1).strip()
            result = json.loads(text)
            state["check_consistent"] = bool(result.get("consistent", False))
            state["alter_target"] = str(result.get("alter", ""))

        except AttributeError:
            # If no JSON block found, mark sense-check as False
            # and store the raw response for debugging.
            state["check_consistent"] = False
            state["alter_target"] = ""

        except json.JSONDecodeError:
            # If the LLM's response isn't valid JSON, mark sense-check as False
            # and store the raw response for debugging.
            state["check_consistent"] = False
            state["alter_target"] = ""

    state["metadata"]["check_consistency"] = metadata

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_consistency.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_check_consistency))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_consistency.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["check_consistency"]) + str(state["alter_target"]))
    return state


def node_submission(state: ExamState) -> ExamState:
    """
    Generates submission requirements for the exam using a language model.

    A submission prompt is formatted with the previously generated overview, instructions, and materials.
    The result and its metadata are stored in the state.

    Args:
        state (ExamState): A dictionary-like object containing keys such as 'overview', 'instructions',
                           'materials_all', 'system_prompt', 'exam_author_model', and a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'submission': the generated submission requirements text.
            - 'metadata["submission"]': usage metadata returned by the model.
    """
    print("creating exam submission requirements    ")
    with open("../prompts/exam_generation_prompts/prompt_submission.txt", "r") as file:
        prompt_template_submission = file.read()
    prompt = prompt_template_submission.format(
        answer_overview=state["overview"],
        answer_instructions=state["instructions"],
        answer_materials=state["materials_candidate"],
    )
    content, metadata = query_agent(
        state["system_prompt"], prompt, state["exam_author_model"]
    )
    state["submission"] = content
    state["metadata"]["submission"] = metadata
    state["sequence"].append("submission")
    state["exam"] = (
        state["instructions"] + state["materials_candidate"] + state["submission"]
    )

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_submission.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_submission.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["submission"]))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_exam.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["exam"]))
    return state


def node_evaluation(state: ExamState) -> ExamState:
    """
    Generates evaluation criteria and guidance for assessing the exam using a language model.

    A prompt is formatted using the exam's overview, instructions, materials, and submission requirements.
    The generated evaluation material and its metadata are stored in the state. The function also increments
    a counter to track the number of answer key generations.

    Args:
        state (ExamState): A dictionary-like object containing keys such as 'overview', 'instructions',
                           'materials_all', 'submission', 'system_prompt', 'exam_author_model', and
                           'answer_key_count', along with a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'evaluation': the generated evaluation material.
            - 'metadata["evaluation"]': usage metadata from the model.
            - 'answer_key_count': incremented by 1.
    """

    print("creating exam evaluation material")
    with open("../prompts/exam_generation_prompts/prompt_evaluation.txt", "r") as file:
        prompt_template_evaluation = file.read()
    prompt = prompt_template_evaluation.format(
        answer_overview=state["overview"],
        # answer_instructions=state["instructions"],
        # answer_materials=state["materials_all"],
        # answer_submission=state["submission"],
        exam=state["exam"],
        materials_evaluator=state["evaluation"],
    )

    content, metadata = query_agent(
        state["system_prompt"], prompt, state["exam_author_model"]
    )
    state["evaluation"] = content
    state["metadata"]["evaluation"] = metadata
    state["sequence"].append("evaluation")

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_evaluation.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_evaluation.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["evaluation"]))
    return state


def node_grading(state: ExamState) -> ExamState:
    """
    Generates a grading script or rubric for evaluating exam submissions using a language model.

    The prompt is constructed with previously generated components: overview, instructions, materials,
    submission requirements, and evaluation criteria. The resulting grading script and its metadata are
    added to the state.

    Args:
        state (ExamState): A dictionary-like object containing keys such as 'overview', 'instructions',
                           'materials_all', 'submission', 'evaluation', 'system_prompt', and
                           'exam_author_model', as well as a nested 'metadata' dictionary.

    Returns:
        ExamState: The updated state with:
            - 'grading': the generated grading script or rubric.
            - 'metadata["grading"]': usage metadata from the model.
    """
    print("generating grading script")
    # Note, I modified the prompt so that files are passed as argument
    with open("../prompts/exam_generation_prompts/prompt_grading.txt", "r") as file:
        prompt_template_grading = file.read()
    prompt = prompt_template_grading.format(
        overview=state["overview"],
        exam=state["exam"],
        materials_evaluator=state["materials_evaluator"],
        evaluation=state["evaluation"],
    )

    content, metadata = query_agent(
        state["system_prompt"], prompt, state["exam_author_model"]
    )
    state["grading"] = content
    state["metadata"]["grading"] = metadata
    state["sequence"].append("grading")

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_grading.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_grading.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["grading"]))
    return state


def node_save_eval_and_answer(state: ExamState) -> ExamState:
    """
    1) Saves the Python grading script from state["grading"] into `task_evaluation.py`
    2) Saves the answer key JSON from state["evaluation"] into `answer_key.json`
    """
    print("saving answer key")
    task_id = state["task_id"]
    path = "../data/exams/" + state["level"] + "/" + state["exam_author_model"] + "/"
    folder = task_id.replace(".", "_")
    # if state["level"] == "advanced":
    #     path = path + "/advanced"
    # if state["level"] == "moderate":
    #     path = path + "/moderate/"
    try:
        # 1. Save the Python grading script
        script = extract_and_save_python_script(
            script_text=state["grading"],
            folder=path + folder,
            filename="task_evaluation.py",
        )
        # 2. Save the answer key
        key = extract_and_save_json(
            json_text=state["evaluation"],
            folder=path + folder,
            filename="answer_key.json",
        )
        state["answer_key"] = key
        print(f"Grading script and answer key saved successfully for task {task_id}.")
    except Exception as exc:
        err_msg = f"Error saving assets for {task_id}: {exc}"
        print(err_msg)
        state["errors"].append(err_msg)
    if len(state["errors"]) != 0:
        print(state["errors"])
    state["sequence"].append("save_eval_and_answer")
    return state


def node_cleanup(state: ExamState) -> ExamState:
    """
    Removes duplicate materials, eliminates unnecessary repetition,
    and resolves inconsistencies in the exam text.

    The function:
      - Deduplicates entries in 'materials_all' (exact duplicates).
      - Uses a language model to detect and reduce unnecessary repetition
        as well as resolve inconsistencies in exam text.
      - Updates 'materials_all' and 'instructions' with cleaned versions.
      - Stores usage metadata for auditing.

    Args:
        state (ExamState): A dictionary-like object containing:
                           - 'instructions'
                           - 'materials_all'
                           - 'exam_author_model'
                           - 'metadata'

    Returns:
        ExamState: The updated state with:
            - 'materials_all' cleaned (deduplicated + no unnecessary repetition)
            - 'instructions' revised for consistency
            - 'metadata["check_inconsistencies"]' containing model usage metadata
    """

    print("checking for inconsistencies and duplicates")
    # Step 1: Remove exact duplicate lines from materials
    deduped_materials = list(dict.fromkeys(state["exam"].splitlines()))
    deduped_materials_text = "\n".join(deduped_materials)

    # Step 2: Load prompt for repetition & inconsistency cleanup
    with open("../prompts/exam_generation_prompts/prompt_cleanup.txt", "r") as file:
        prompt_cleanup = file.read()

    # Step 3: Query the model to resolve inconsistencies and remove unnecessary repetition
    content, metadata = query_agent(
        prompt_cleanup, deduped_materials_text, state["exam_author_model"]
    )

    # Step 4: Update state
    state["exam"] = content
    state["metadata"]["check_inconsistencies"] = metadata
    state["sequence"].append("removed inconsistencies")

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_cleanup.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_cleanup))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_cleanup.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["exam"]))
    return state


def node_check_answer_key(state: ExamState) -> ExamState:
    """
    Runs a subprocess to evaluate the answer key for the exam and retrieves the grading results from a JSON file.

    This function:
    1. Constructs the path based on the task ID and exam model.
    2. Runs an external Python script (`task_evaluation.py`) to generate the answer key.
    3. Loads and parses the results from a JSON file (`test_results.json`) to extract the overall score.
    4. Updates the state with the grade and any errors encountered during the process.

    If the script fails or the required files are not found, it captures the errors and stores them in the state.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'task_id': The identifier for the task.
            - 'exam_author_model': The model of the exam author.
            - 'level': The level of the exam ('advanced' or other).
            - 'errors': A list where error messages will be stored.
            - A `key_grade` entry where the overall score will be stored.

    Returns:
        ExamState: The updated state with:
            - 'key_grade': The overall score from the answer key (or NaN if there was an error).
            - 'errors': A list containing error messages encountered during the process.
    """
    print("checking answer key")
    errors = []
    task_id = state["task_id"]
    subfolder = task_id.replace(".", "_")
    path = (
        "../data/exams/"
        + state["level"]
        + "/"
        + state["exam_author_model"]
        + "/"
        + task_id.replace(".", "_")
    )

    try:
        result = subprocess.run(
            ["python", "task_evaluation.py", "answer_key.json", "answer_key.json"],
            cwd=path,
            check=True,  # Raise an exception if the command fails
            stderr=subprocess.PIPE,  # Capture stderr
            stdout=subprocess.PIPE,  # Capture stdout (if needed)
        )
        print("Script executed successfully.")

        # If the script runs successfully, append None to errors list (no errors)
        errors.append(None)
        # Now get answer key grade
        try:
            # Load the JSON file
            with open(path + "test_results.json", "r") as f:
                data = json.load(f)

            # Extract the overall_score
            overall_score = data.get("overall_score", None)
            state["key_grade"] = np.round(overall_score)
            return state

        except FileNotFoundError:
            print(path)
            print(f"Error: The file {path}test_results.json was not found.")
            errors.append("no overall score found")
            state["errors"].append(errors)
            state["key_grade"] = np.nan
            return state
        except json.JSONDecodeError:
            print(f"Error: The file {path}test_results.json is not a valid JSON file.")
            errors.append("not json file")
            state["errors"].append(errors)
            state["key_grade"] = np.nan
            return state
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            state["errors"].append(errors)
            errors.append(str(e))
            state["key_grade"] = np.nan
            return state

    except subprocess.CalledProcessError as e:
        # Capture and store the error output in the errors list
        print(f"Error: Script failed with return code {e.returncode}")
        print(f"Error Output:\n{e.stderr.decode('utf-8')}")
        errors.append(
            e.stderr.decode("utf-8")
        )  # Append the error message to the errors list
        state["errors"].append(errors)
        state["key_grade"] = np.nan
        return state
    except FileNotFoundError:
        error_message = "Error: The script or directory was not found. Check the path."
        print(error_message)
        print(path)
        errors.append(error_message)  # Append the error message to the errors list
        state["errors"].append(errors)
        state["key_grade"] = np.nan
        return state
    except Exception as e:
        # Capture and store any unexpected error
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        errors.append(error_message)  # Append the error message to the errors list
        state["errors"].append(errors)
        state["key_grade"] = np.nan

    if state["key_grade"] >= state["key_grade_threshold"]:
        state["check_answer_key"] = True
    else:
        state["check_answer_key"] = False
    print("check answer key", state["check_answer_key"], state["key_grade"])
    state["sequence"].append("check_answer_key " + str(state["key_grade"]))

    if len(state["errors"]) != 0:
        print(state["errors"])
    return state


def node_check_answer_coverage(state: ExamState) -> ExamState:
    """
    Checks what share of answers in the JSON answer key are contained
    in the exam instructions.

    The function:
      - Loads the answer key (assumed JSON string in state["answer_key"]).
      - Compares each answer with the exam instructions text.
      - Computes the share of answers that appear in the instructions.
      - Stores this percentage in state["metadata"]["answer_coverage"].

    Args:
        state (ExamState): A dictionary-like object containing:
                           - 'instructions' (str): Exam instructions text.
                           - 'answer_key' (str): JSON string with answers.
                           - 'metadata' (dict): For storing results.

    Returns:
        ExamState: The updated state with:
            - 'metadata["answer_coverage"]': dict containing coverage statistics
    """
    print("checking answer contamination")
    task_id = state["task_id"]
    path = "../data/exams/" + state["level"] + "/" + state["exam_author_model"] + "/"
    folder = task_id.replace(".", "_")
    instructions_text = state["exam"].lower()
    # Parse the answer key JSON
    try:
        answer_key = json.loads(path + folder + "answer_key.json")
    except json.JSONDecodeError:
        state["answer_coverage"] = 0.0
        return state

    # Extract answers (assuming dict of {question: answer})
    answers = list(answer_key.values())
    total_answers = len(answers)

    if total_answers == 0:
        state["answer_coverage"] = 0.0
        return state

    # Count how many answers appear in instructions
    contained_count = sum(1 for ans in answers if str(ans).lower() in instructions_text)

    coverage_share = contained_count / total_answers

    # Store results
    if coverage_share < 0.3:
        state["check_answer_coverage"] = True
    else:
        state["check_answer_coverage"] = False
    state["sequence"].append(
        "check_answer_coverage "
        + str(state["check_answer_coverage"])
        + str(coverage_share)
    )
    state["counter"] = state["counter"] + 1

    return state


def node_overall_makes_sense(state: ExamState) -> ExamState:
    """
    Performs a sanity check to evaluate whether the generated exam makes logical sense overall.

    The function:
    1. Loads system and user prompts from files.
    2. Queries a language model to evaluate whether the overall exam makes sense.
    3. Parses the response, attempting to interpret it as JSON.
    4. Based on the response, updates the state with:
        - A boolean flag `check_overall_makes_sense` to indicate whether the exam makes sense.
        - An explanation `explanation_overall_makes_sense` if available.

    If the response is not valid JSON, the function marks the exam as not making sense and stores the raw response for debugging.

    Args:
        state (ExamState): A dictionary-like object containing:
            - System prompts, user messages, and metadata for checking exam quality.
            - Any previous responses or checks relevant to the overall exam quality.

    Returns:
        ExamState: The updated state with:
            - 'check_overall_makes_sense': Boolean flag indicating whether the exam makes logical sense.
            - 'explanation_overall_makes_sense': Explanation of the evaluation or a raw response if parsing failed.
            - 'metadata["check_sense"]': Metadata related to this sanity check.
    """
    print("checking if exam makes sense")

    with open(
        "../prompts/sanity_check_prompts/prompt_makes_sense_system.txt", "r"
    ) as file:
        system_prompt = file.read()
    with open(
        "../prompts/sanity_check_prompts/prompt_makes_sense_user.txt", "r"
    ) as file:
        user_message = file.read()

    user_message = user_message.format(
        instructions=state["instructions"],
        materials_candidate=state["materials_candidate"],
        submission=state["submission"],
        overview=state["overview"],
        materials_evaluator=state["materials_evaluator"],
        evaluation=state["evaluation"],
        grading=state["grading"],
        answer_key=state["answer_key"],
    )

    content, metadata = query_agent(
        system_prompt, user_message, state["exam_author_model"]
    )

    state["metadata"]["check_sense"] = metadata
    response = content.strip()

    try:
        result = json.loads(response)
        state["check_overall_makes_sense"] = bool(result.get("makes_sense", False))
        state["explanation_overall_makes_sense"] = str(result.get("explanation", ""))
    except:
        try:
            text = re.search(r"```json(.*?)```", response, re.DOTALL).group(1).strip()
            result = json.loads(text)
            state["check_overall_makes_sense"] = bool(result.get("makes_sense", False))
            state["explanation_overall_makes_sense"] = str(
                result.get("explanation", "")
            )
        except json.JSONDecodeError:
            # If the LLM's response isn't valid JSON, mark sense-check as False
            # and store the raw response for debugging.
            state["check_overall_makes_sense"] = False
            state["explanation_overall_makes_sense"] = (
                "Could not parse JSON. Raw LLM response:\n" + response
            )

    state["sequence"].append(
        "check_makes_sense " + str(state["check_overall_makes_sense"])
    )

    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_makes_sense.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(user_message))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_content_makes_sense.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(
            str(state["check_overall_makes_sense"])
            + str(state["explanation_overall_makes_sense"])
        )
    return state


def node_check_exam_feasibility(state: ExamState) -> ExamState:
    """
    Checks whether the exam can reasonably be completed by a human
    in a specific profession within 90 minutes and whether the level
    is appropriate for a basic exam for someone with a given degree.

    The function:
      - Loads a feasibility-check prompt from file.
      - Queries the model with exam instructions and materials.
      - Updates state with feasibility flag and reasoning metadata.


    Args:
        state (ExamState): A dictionary-like object containing:
                           - 'instructions' (str)
                           - 'materials_all' (str)
                           - 'exam_author_model' (str)
                           - 'metadata' (dict)
                           - 'target_profession' (str, optional)
                           - 'target_degree' (str, optional)

    Returns:
        ExamState: The updated state with:
            - 'metadata["exam_feasibility"]': dict with model response and usage metadata
            - 'exam_feasible': Boolean flag if exam is suitable
    """
    print("checking exam feasibility")
    # Extract profession/degree info from state (fallbacks provided)
    occupation = state["occupation"]
    education = state["education"]

    # Step 1: Load feasibility prompt
    with open("../prompts/sanity_check_prompts/prompt_feasibility.txt", "r") as file:
        prompt_feasibility = file.read()

    # Inject target role and degree into the prompt
    prompt_filled = prompt_feasibility.format(
        occupation=occupation, education=education, level=state["level"]
    )

    # Step 2: Query model
    content, metadata = query_agent(
        prompt_filled, state["exam"], state["exam_author_model"]
    )
    state["metadata"]["check_feasible"] = metadata
    response = content.strip()

    try:
        result = json.loads(response)
        state["check_feasible"] = bool(result.get("feasible", False))
        state["explanation_feasible"] = str(result.get("explanation", ""))
    except:
        try:
            text = re.search(r"```json(.*?)```", response, re.DOTALL).group(1).strip()
            result = json.loads(text)
            state["check_feasible"] = bool(result.get("makes_sense", False))
            state["explanation_feasible"] = str(result.get("explanation", ""))
        except json.JSONDecodeError:
            # If the LLM's response isn't valid JSON, mark sense-check as False
            # and store the raw response for debugging.
            state["check_feasible"] = False
            state["explanation_feasible"] = (
                "Could not parse JSON. Raw LLM response:\n" + response
            )
    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_').replace('.', '_')}/{state['counter']}_prompt_feasibility.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(prompt_filled))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_')}/{state['counter']}_content_feasiblity.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["check_feasible"]) + str(state["explanation_feasible"]))
    return state


def node_end(state: ExamState) -> ExamState:
    """
    Compiles the final exam based on various sanity checks and criteria.

    The function:
    1. Checks if the materials are real, if the exam contains no fake internet references,
       if the grade threshold is met, and if the overall exam makes sense.
    2. If all checks pass, it compiles the final exam by concatenating the instructions,
       candidate materials, and submission details into a single exam string.
    3. If any check fails, the exam is marked as invalid.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'check_real_materials': Boolean flag indicating the authenticity of materials.
            - 'check_no_internet': Boolean flag indicating the validity of internet references.
            - 'key_grade': The grade of the exam's answer key.
            - 'key_grade_threshold': The threshold grade required to validate the exam.
            - 'check_overall_makes_sense': Boolean flag indicating if the exam makes logical sense.
            - 'instructions', 'materials_candidate', 'submission': Exam components to be included if valid.

    Returns:
        ExamState: The updated state with:
            - 'exam': The final compiled exam or "Exam not valid" if any checks failed.
    """
    print("finalizing exam")
    # state["exam"] = (
    #     state["instructions"] + state["materials_candidate"] + state["submission"]
    # )
    # if not state["check_real_materials"]:
    #     state["fail_reason"] = "Materials contain fake materials"
    #     state["exam"] = "Exam not valid"
    # if not state["check_no_internet"]:
    #     state["fail_reason"] = (
    #         state["fail_reason"] + "; Materials contain fake internet references"
    #     )
    #     state["exam"] = "Exam not valid"
    # if state["key_grade"] < state["key_grade_threshold"]:
    #     state["fail_reason"] = state["fail_reason"] + "; Key grade below threshold"
    #     state["exam"] = "Exam not valid"
    # if not state["check_overall_makes_sense"]:
    #     state["fail_reason"] = state["fail_reason"] + "; Exam does not make sense"
    #     state["exam"] = "Exam not valid"
    # if state["answer_coverage"] > 0.3:
    #     state["fail_reason"] = (
    #         state["fail_reason"] + "; Exam contains more than 30% of answers"
    #     )
    #     state["exam"] = "Exam not valid"
    # if not state["feasible"]:
    #     state["fail_reason"] = state["fail_reason"] + "; Exam not appropriate for level"
    state["sequence"].append("end" + str(state["exam"][:30]))
    return state


def node_improve(state: ExamState) -> ExamState:
    """
    Improves the exam by addressing identified issues using a language model.

    The function:
    1. Loads an improvement prompt from file.
    2. Queries the model with the current exam and identified issues.
    3. Updates the state with the improved exam content and related metadata.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'exam': The current exam content.
            - 'fail_reason': A string describing issues to be addressed.
            - 'exam_author_model': The model to use for improvement.
            - 'metadata': A dictionary for storing usage metadata.
    """
    print("improving exam")
    # 1. Load improvement prompt
    prompt_path = Path("../prompts/exam_generation_prompts/prompt_improvement.txt")
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file not found at {prompt_path}")
    prompt_template = prompt_path.read_text(encoding="utf-8")

    if state["check_overall_makes_sense"]:
        sense_issues = ""
    else:
        sense_issues = state["explanation_overall_makes_sense"] + "; "
    if state["check_feasible"]:
        feasible_issues = ""
    else:
        feasible_issues = state["explanation_feasible"] + "; "

    if state["check_answer_coverage"] == False:
        issues = sense_issues
        +feasible_issues + "; Exam contains more than 30% of answers"

    else:
        issues = sense_issues + feasible_issues
    formatted_prompt = prompt_template.format(
        issues=issues,
        # evaluator_info=state["evaluation"],
        # exam=state["exam"],
    )
    print(formatted_prompt)
    print(state["exam"])
    # Step 2: Query model
    content, metadata = query_agent(
        formatted_prompt, state["exam"], state["exam_author_model"]
    )
    print("------------------")
    print(content)
    state["exam"] = content
    state["sequence"].append("improved exam")
    state["counter"] = state["counter"] + 1
    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_')}/{state['counter']}_prompt_improve.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(formatted_prompt))

    with open(
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_')}/{state['counter']}_exam_improved.txt",
        "w",
        encoding="utf-8",
    ) as f:
        f.write(content + "\n" + str(state["exam"]))
    return state


########################
# Routing functions
#######################
def route_after_materials_candidate(state: ExamState) -> str:
    """
    Determines the next node in the exam generation workflow after checking the candidate materials.

    This function evaluates whether the process should proceed with generating materials,
    retrying material extraction, or directly ending the process based on how many times
    the extraction has failed.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'failed_candidate_materials': The number of times material extraction has failed.
            - 'materials_candidate': The current status of the extracted candidate materials.

    Returns:
        str: The name of the next node to proceed with:
            - "node_end" if material extraction has failed three times.
            - "node_materials" if candidate materials were not extracted.
            - "node_check_images" if candidate materials were successfully extracted.
    """

    if state["check_candidate_materials"]:
        return "node_check_images"
    else:
        return "node_materials"


def route_after_image_check(state: ExamState) -> str:
    """
    Determines the next node in the exam generation workflow after checking the candidate images.

    This function evaluates whether the exam materials are valid (no fake materials)
    and decides whether to proceed to the website validation step or end the process.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'check_real_materials': Boolean flag indicating if the materials are real.

    Returns:
        str: The name of the next node to proceed with:
            - "node_end" if the materials are fake.
            - "node_check_websites" if the materials are valid.
    """

    if state["check_real_materials"]:
        return "node_check_websites"

    else:
        return "node_materials"


def route_after_internet_check(state: ExamState) -> str:
    """
    Determines the next node in the exam generation workflow after checking for internet references.

    This function evaluates whether the materials include valid internet references and decides
    whether to proceed to the submission phase or end the process.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'check_no_internet': Boolean flag indicating if the materials are free from internet references.

    Returns:
        str: The name of the next node to proceed with:
            - "node_end" if there are invalid internet references.
            - "node_submission" if there are no internet references.
    """
    if state["check_no_internet"]:
        return "node_check_consistency"

    else:
        return "node_materials"


def route_after_consistency_check(state: ExamState) -> str:
    if state["check_consistency"]:
        return "node_submission"
    else:
        if state["alter_target"] == "instructions":
            return "node_instructions"
        if state["alter_target"] == "materials":
            return "node_materials"
        return "node_materials"


# Update route
def route_after_key_check(state: ExamState) -> str:
    """
    Determines the next node in the exam generation workflow after checking the answer key.

    This function evaluates whether the answer key meets the required grade threshold,
    and decides whether to proceed with overall validation, evaluation, or end the process.

    Args:
        state (ExamState): A dictionary-like object containing:
            - 'key_grade': The grade of the exam's answer key.
            - 'key_grade_threshold': The grade threshold required to pass the exam.
            - 'answer_key_count': The number of times the answer key evaluation has been performed.

    Returns:
        str: The name of the next node to proceed with:
            - "node_overall_makes_sense" if the key grade meets the threshold.
            - "node_end" if the answer key has been evaluated more than three times or doesn't meet the threshold.
            - "node_evaluation" if the answer key requires further evaluation.
    """

    if state["check_answer_key"]:
        return "node_check_answer_coverage"
    else:
        return "node_evaluation"

        # Update route


def route_after_key_contamination_check(state: ExamState) -> str:
    if state["check_answer_coverage"]:
        return "node_end"
    else:
        return "node_improve"


def route_after_sense_check(state: ExamState) -> str:
    if state["check_overall_makes_sense"]:
        return "node_check_exam_feasibility"
    else:
        return "node_improve"


def route_after_feasibility_check(state: ExamState) -> str:
    if state["check_feasible"]:
        return "node_grading"
    else:
        return "node_improve"


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = "claude-sonnet-4-20250514"
    if len(sys.argv) > 2:
        occupation_group = sys.argv[2]
    else:
        occupation_groups = [
            # "Management Occupations",
            # "Business and Financial Operations Occupations",
            "Computer and Mathematical Occupations",
        ]
    if len(sys.argv) > 3:
        core_label = sys.argv[3]
    else:
        core_label = "CORE"
    if len(sys.argv) > 4:
        level = sys.argv[4]
    else:
        level = "basic"
    if level == "basic":
        level_string = ""
    if level == "advanced":
        level_string = "advanced_"
    if level == "moderate":
        level_string = "moderate_"
    for occupation_group in occupation_groups:
        print("Creating exams for ", occupation_group, "using ", model)
        tasks_file = f"../data/filtered_tasks/{occupation_group.replace(' ', '_').lower()}_{core_label}.csv"
        df_tasks = pd.read_csv(tasks_file)
        print("Overall number of tasks: ", df_tasks.shape)

        # folder for saving results
        folder_path = f"../data/exams/{level}/{model}/"
        if Path(
            os.path.join(
                folder_path, f"{occupation_group.replace(' ', '_').lower()}_exams.csv"
            )
        ).exists():
            df_existing = pd.read_csv(
                f"{folder_path}{occupation_group.replace(' ', '_').lower()}_exams.csv"
            )
            print(" existing file loaded!")
        else:
            print("no file exists")

        try:
            print("shape of existing file: ", df_existing.shape)
            df_existing = df_existing[df_existing["exam"] != ""]
            df_existing = df_existing[~df_existing["exam"].isnull()]
            print("shape of existing file: ", df_existing.shape)
            df_tasks = df_tasks[~(df_tasks["task_id"].isin(df_existing["task_id"]))]
            print(df_tasks["task_id"])

        except:
            print("No existing file found, starting from scratch")
        print("Number of tasks remaining: ", df_tasks.shape[0])
        df_tasks = df_tasks[
            [
                "occupation",
                "task_description",
                "task_id",
                "required_tools",
                "required_materials",
                "education",
            ]
        ]
        df_tasks = df_tasks[2:]
        # Initialize an empty list to store result states
        result_states = []
        graph_builder = StateGraph(ExamState)

        graph_builder.add_node("construct_system_prompt", node_system_prompt)
        graph_builder.add_node("node_overview", node_overview)
        graph_builder.add_node("node_instructions", node_instructions)
        graph_builder.add_node("node_materials", node_materials)
        graph_builder.add_node("node_check_images", node_check_materials_fake_image)
        graph_builder.add_node("node_check_websites", node_check_materials_fake_website)
        graph_builder.add_node(
            "node_check_consistency",
            node_check_consistency,
        )
        graph_builder.add_node("node_cleanup", node_cleanup)
        graph_builder.add_node("node_submission", node_submission)
        graph_builder.add_node("node_evaluation", node_evaluation)

        graph_builder.add_node("node_grading", node_grading)
        graph_builder.add_node("node_save_eval_and_answer", node_save_eval_and_answer)
        graph_builder.add_node("node_check_answer_key", node_check_answer_key)
        # graph_builder.add_node(
        #     "node_check_inconsistencies_duplicates",
        #     node_check_inconsistencies_duplicates,
        # )
        graph_builder.add_node(
            "node_check_exam_feasibility", node_check_exam_feasibility
        )
        graph_builder.add_node("node_check_answer_coverage", node_check_answer_coverage)
        graph_builder.add_node("node_overall_makes_sense", node_overall_makes_sense)
        graph_builder.add_node("node_improve", node_improve)
        graph_builder.add_node("node_end", node_end)
        # graph_builder.add_node("node_pause_before_evaluation", node_pause_before_evaluation)

        # Add edges the graph
        graph_builder.add_edge(START, "construct_system_prompt")
        graph_builder.add_edge("construct_system_prompt", "node_overview")
        graph_builder.add_edge("node_overview", "node_instructions")
        ### NOTE - probably add conditional edge depending on whether materials are required
        graph_builder.add_edge("node_instructions", "node_materials")
        ### Add conditional edges if materials_fake_website or materials_fake_image then end the process

        graph_builder.add_conditional_edges(
            "node_materials", route_after_materials_candidate
        )

        graph_builder.add_conditional_edges(
            "node_check_images", route_after_image_check
        )
        graph_builder.add_conditional_edges(
            "node_check_websites", route_after_internet_check
        )
        graph_builder.add_conditional_edges(
            "node_check_consistency", route_after_consistency_check
        )

        # If it passes will continue to generatl submissions and grading
        graph_builder.add_edge("node_submission", "node_cleanup")
        graph_builder.add_edge("node_cleanup", "node_evaluation")

        # graph_builder.add_edge("node_pause_before_evaluation", "node_evaluation")

        graph_builder.add_edge("node_evaluation", "node_overall_makes_sense")

        graph_builder.add_conditional_edges(
            "node_overall_makes_sense", route_after_sense_check
        )
        graph_builder.add_conditional_edges(
            "node_check_exam_feasibility", route_after_feasibility_check
        )
        graph_builder.add_edge("node_improve", "node_evaluation")

        # Now check the answer key and how much it scores
        graph_builder.add_edge("node_grading", "node_save_eval_and_answer")
        graph_builder.add_edge("node_save_eval_and_answer", "node_check_answer_key")
        # add conditional edges in case materials for candidate where not extracted

        graph_builder.add_conditional_edges(
            "node_check_answer_key", route_after_key_check
        )

        graph_builder.add_conditional_edges(
            "node_check_answer_coverage", route_after_key_contamination_check
        )

        # graph_builder.add_edge("node_overall_makes_sense", "node_end")

        # graph_builder.add_edge("node_check_answer_key", "node_overall_makes_sense")

        graph_builder.add_edge("node_end", END)
        print("compiling graph")

        graph = graph_builder.compile()

        for _, row in df_tasks.iterrows():  # Use iterrows() to iterate over rows
            # Initialize the state for the current row
            print("Creating exam for ", row["task_description"])
            init_state: ExamState = {
                "occupation": row["occupation"],
                "task_id": str(row["task_id"]),  # Convert task_id to a string
                "task_description": row["task_description"],
                "exam_author_model": model,
                # Map your row fields to the typed dict fields
                "tools": safe_eval(row["required_tools"]),
                "materials": safe_eval(row["required_materials"]),
                "level": level,
                # Provide defaults or placeholders for the rest
                "exam": {},
                "system_prompt": "",
                "overview": "",
                "instructions": "",
                "materials_evaluator": "",
                "materials_candidate": "",
                "submission": "",
                "evaluation": "",
                "grading": "",
                "answer_key": "",
                "errors": [],
                "check_candidate_materials": False,
                "check_real_materials": True,
                "check_no_internet": True,
                "check_consistency": True,
                "key_grade_threshold": 100,
                "key_grade": 0.0,
                "check_overall_makes_sense": True,
                "explanation_overall_makes_sense": "",
                "metadata": {},
                "education": row["education"],
                "check_answer_key": True,
                "check_answer_coverage": True,
                "fail_reasons": "",
                "check_feasible": True,
                "explanation_feasible": "",
                "sequence": [],
                "check_instructions_materials": True,
                "alter_target": None,
                "counter": 0,
            }

            cumulative_state = init_state.copy()  # start with your initial state
            try:
                for event in graph.stream(
                    cumulative_state,
                    stream_mode="values",
                    config={"recursion_limit": 50},
                ):
                    # Merge current node output into cumulative state
                    cumulative_state.update(event)
                    # result_states.append(cumulative_state.copy())  # keep a historical snapshot
                    last_state = cumulative_state.copy()

                # Final result after all nodes
                result_state = cumulative_state

                result_states.append(result_state)  # Append final state to results
            except Exception:
                print("Graph failed. Last state before crash:")
                # Append last state to results and save all historical states to CSV
                result_states.append(last_state)

                raise
            df_result_states = pd.DataFrame(result_states)
            df_result_states.to_csv(
                folder_path
                + f"{level_string}_{occupation_group.replace(' ', '_').lower()}_exams.csv",
                index=False,
            )
            # except Exception as e:
            #     # Handle any errors during graph invocation
            #     print(f"Error processing task_id {row['task_id']}: {e}")
            #     # Append an error state to the list for debugging
            # result_states.append(
            #     {
            #         "occupation": row["occupation"],
            #         "task_id": row["task_id"],
            #         "task_description": row["task_description"],
            #         "exam_author_model": model,
            #         "errors": [str(e)],
            #         "sequence": row["sequence"],
            #     }
            # )

            try:
                df_result_states = pd.concat(
                    [df_existing, df_result_states], ignore_index=True
                )
            except:
                print("No existing file found, saving the current results only.")
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)
            if level_string == "advanced":
                df_result_states.to_csv(
                    folder_path
                    + f"{level_string}_{occupation_group.replace(' ', '_').lower()}_exams.csv",
                    index=False,
                )
            if level_string == "moderate":
                df_result_states.to_csv(
                    folder_path
                    + f"{level_string}_{occupation_group.replace(' ', '_').lower()}_exams.csv",
                    index=False,
                )
            else:
                df_result_states.to_csv(
                    folder_path
                    + f"{occupation_group.replace(' ', '_').lower()}_exams.csv",
                    index=False,
                )
