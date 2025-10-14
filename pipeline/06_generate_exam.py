#### Note uses python 3.9 environment (newenv)
import ast
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Annotated, TypedDict
import uuid

import anthropic
from dotenv import find_dotenv, load_dotenv
from IPython.display import Image, display
from langchain.schema import HumanMessage, SystemMessage
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # or your equivalent import
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
import numpy as np
from openai import OpenAI
import pandas as pd
import regex as re
from typing_extensions import TypedDict

dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
#######################################
# Helper functions
from typing import Any, Dict, Tuple

from query_agents import *


def _save_step_output(
    base_dir: str, step_name: str, counter: int, prompt: str, content: str
) -> None:
    """Helper: save prompt and generated content for debugging/reproducibility."""
    os.makedirs(base_dir, exist_ok=True)
    with open(
        os.path.join(base_dir, f"{counter}_prompt_{step_name}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(prompt)
    with open(
        os.path.join(base_dir, f"{counter}_content_{step_name}.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(content)


def _get_exam_dir(state: Dict[str, Any]) -> str:
    """Helper: construct a consistent path for saving intermediate exam artifacts."""
    task_id_sanitized = str(state["task_id"]).replace(".", "_")
    return os.path.join(
        "..", "data", "exams", "basic", state["exam_author_model"], task_id_sanitized
    )


def _load_prompt_template(path: str) -> str:
    """Helper: safely load a text prompt template."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Prompt template not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


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
        tools_instructions = "- The candidate does not have access to any special tools."
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
    """
    Extracts Python code from either:
    1. Triple backticks ```python ... ```
    2. Output-style blocks like:
       ----------- task_evaluation.py -----------
       <code>
       ----------- end script -----------
    Saves the code to a file and returns it as a string.
    """
    # Try triple-backtick Python block first
    match = re.search(r"```python(.*?)```", script_text, re.DOTALL)
    if match:
        code = match.group(1).strip()
    else:
        # Try "output-style" block
        match = re.search(
            r"[-]{10,}\s*\w+\.py\s*[-]{10,}\n(.*?)\n[-]{10,}\s*end script\s*[-]{10,}",
            script_text,
            re.DOTALL,
        )
        if match:
            code = match.group(1).strip()
        else:
            raise ValueError("No Python code block found in the text.")

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return code


def extract_and_save_json(json_text: str, folder: str, filename: str = "answer_key.json"):
    """
    Extracts JSON from a string and saves it to a file.

    Handles two formats:
    1. JSON enclosed in triple backticks ```json ...```
    2. JSON starting immediately after a header line, e.g.,
       'Answer key (JSON for automated checking)\n{ ... }'
    """
    # First, try the triple backticks ```json ... ``` format
    match = re.search(r"```json(.*?)```", json_text, re.DOTALL)

    if match:
        json_str = match.group(1).strip()
    else:
        # Try to find JSON starting after a header line like "Answer key ..."
        match = re.search(r"Answer key.*?\n(\{.*\})", json_text, re.DOTALL)
        if not match:
            raise ValueError("No JSON block found in the text.")
        json_str = match.group(1).strip()

    # Parse the JSON
    data = json.loads(json_str)

    # Ensure the folder exists and save the JSON
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
    materials: str
    materials_evaluator: str
    materials_candidate: str
    check_candidate_materials: bool
    submission: str
    grading: str
    evaluation: str
    answer_key: str
    evaluator_instructions: str
    errors: list
    # Boolean flags for validation checks
    check_materials_structure_passed: bool
    check_consistency: bool
    alter_target: str
    # Key grade and cocheck_candidate_materialsunt how many times below threshold
    key_grade_threshold: float
    key_grade: float
    check_makes_sense: bool
    explanation_overall_makes_sense: str
    metadata: dict
    education: str
    check_answer_key: bool
    check_grading_script: bool
    check_feasible: bool
    explanation_feasible: str
    sequence: list
    counter: int
    next_node: str
    check_submission_vs_answer_key_passed: bool
    diagnosis: str
    diagnosis_rationale: str
    recommendation: str
    diagnose_reverts: int
    abort_reason: str
    diagnose_key_failure_calls: int
    exam_sanity_report: str


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


def node_overview(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates a structured exam overview using the LLM.
    Produces a summary of exam intent and design for the evaluator.
    """
    print("🧠 Generating exam overview...")

    # Load overview prompt
    prompt_path = "../prompts/exam_generation_prompts/prompt_overview.txt"
    prompt_overview = _load_prompt_template(prompt_path)

    # Query LLM
    content, metadata = query_agent(
        state["system_prompt"],
        prompt_overview,
        state["exam_author_model"],
    )

    # Update state
    state["overview"] = content.strip()
    state["metadata"]["overview"] = metadata
    state["sequence"].append("overview")
    state["counter"] += 1

    # Save prompt + content for traceability
    base_dir = _get_exam_dir(state)
    _save_step_output(base_dir, "overview", state["counter"], prompt_overview, state["overview"])

    return state


def node_instructions(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generates clear, structured exam instructions for the candidate,
    based on the overview produced in the previous step.
    """
    print("🧠 Generating exam instructions...")

    # Load instruction prompt template
    prompt_path = "../prompts/exam_generation_prompts/prompt_instructions.txt"
    prompt_template = _load_prompt_template(prompt_path)

    # Fill template with overview context
    prompt = prompt_template.format(overview=state.get("overview", "").strip())

    # Query LLM
    content, metadata = query_agent(
        state["system_prompt"],
        prompt,
        state["exam_author_model"],
    )

    # Update state
    state["instructions"] = content.strip()
    state["metadata"]["instructions"] = metadata
    state["sequence"].append("instructions")
    state["counter"] += 1

    # Save prompt + content
    base_dir = _get_exam_dir(state)
    _save_step_output(base_dir, "instructions", state["counter"], prompt, state["instructions"])

    return state


def node_check_instructions_structure(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node: Check that the generated exam instructions are structurally valid and complete.

    Validates that:
    1. The required markdown sections exist:
       - "### Exam Title"
       - "### Instructions for Candidate"
       - "### Expected Outcome"
       - "### Provided Materials"
       - "### Submission Format"
    2. The text includes reference to 'test_submission.json'.
    3. The text includes a code block with a JSON schema example.

    If any checks fail:
      - Add human-readable messages to `state['errors']`
      - Set `state['check_instructions_structure_passed'] = False`
      - Route back to `node_improve`
    If all checks pass:
      - Set `state['check_instructions_structure_passed'] = True`
      - Continue forward
    """
    print("🔍 Validating instructions structure...")
    text = state.get("instructions", "")
    errors = []

    # Define required sections
    required_sections = [
        "### Exam Title",
        "### Instructions for Candidate",
        "### Expected Outcome",
        "### Provided Materials",
        "### Submission Format",
    ]

    # 1️⃣ Check presence of all required markdown sections
    for section in required_sections:
        if section.lower() not in text.lower():
            errors.append(f"Missing required section: '{section}'")

    # 2️⃣ Check that test_submission.json is mentioned
    if "test_submission.json" not in text:
        errors.append("Missing reference to 'test_submission.json' in instructions.")

    # Update state
    if errors:
        state["errors"].extend(errors)
        state["check_instructions_structure_passed"] = False
        state["fail_reasons"] = "\n".join(errors)
        state["next_node"] = "node_instructions"
        print("❌ Instructions structure check failed:", errors)
    else:
        state["check_instructions_structure_passed"] = True
        state["next_node"] = "node_materials"  # or whatever your next node is
        print("✅ Instructions structure check passed.")

    return state


def node_materials(state: ExamState) -> ExamState:
    """
    Generates exam materials using a language model.
    Updates:
        - materials: raw model output (tagged)
        - metadata["materials"]: usage info from LLM
        - sequence and counter
        - next_node: points to structure check
    """
    print("🧠 Generating exam materials")
    with open("../prompts/exam_generation_prompts/prompt_materials.txt", "r") as file:
        prompt_template_materials = file.read()

    prompt = prompt_template_materials.format(
        answer_overview=state.get("overview", ""),
        answer_instructions=state.get("instructions", ""),
    )

    content, metadata = query_agent(
        state.get("system_prompt", ""), prompt, state.get("exam_author_model")
    )

    state["metadata"]["materials"] = metadata
    state["materials"] = content  # store raw output for checking later

    state["sequence"].append("materials_generated")
    state["counter"] = state.get("counter", 0) + 1

    # Save prompt and raw output
    task_folder = (
        f"../data/exams/basic/{state['exam_author_model']}/{state['task_id'].replace('.', '_')}/"
    )
    os.makedirs(task_folder, exist_ok=True)
    with open(f"{task_folder}{state['counter']}_prompt_materials.txt", "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(
        f"{task_folder}{state['counter']}_content_materials_raw.txt", "w", encoding="utf-8"
    ) as f:
        f.write(state["materials"])

    state["next_node"] = "node_check_materials_structure"
    return state


def node_check_materials_structure(state: ExamState) -> ExamState:
    """
    Validates and extracts generated exam materials.
    Updates:
        - materials_candidate
        - materials_evaluator
        - check_materials_structure_passed
        - errors / fail_reasons
        - next_node
    """
    print("🔍 Validating materials...")

    text = state.get("materials", "").strip()
    errors = []

    # Handle "No material required"
    if re.fullmatch(r"(?i)no material required", text):
        state["materials_candidate"] = "No material required"
        state["materials_evaluator"] = ""
        state["check_materials_structure_passed"] = True
        state["next_node"] = "node_check_consistency"
        print("✅ Materials check passed.")

        return state

    # Extract candidate/evaluator sections
    candidate_match = re.search(
        r"<MATERIALS_FOR_CANDIDATE>(.*?)</MATERIALS_FOR_CANDIDATE>", text, re.DOTALL
    )
    evaluator_match = re.search(
        r"<MATERIALS_EXPLANATION_FOR_EVALUATOR>(.*?)</MATERIALS_EXPLANATION_FOR_EVALUATOR>",
        text,
        re.DOTALL,
    )

    state["materials_candidate"] = candidate_match.group(1).strip() if candidate_match else ""
    state["materials_evaluator"] = evaluator_match.group(1).strip() if evaluator_match else ""

    if not state["materials_candidate"]:
        errors.append("Empty or missing <MATERIALS_FOR_CANDIDATE> section.")
    if not state["materials_evaluator"]:
        errors.append("Empty or missing <MATERIALS_EXPLANATION_FOR_EVALUATOR> section.")

    # Check for non-deterministic / placeholder terms
    ambiguous_terms = ["example only", "random", "placeholder", "sample data", "mockup"]
    if any(term in text.lower() for term in ambiguous_terms):
        errors.append("Detected non-deterministic or ambiguous terms in materials.")

    # LLM-based check for fake assets
    if not errors:
        try:
            with open("../prompts/sanity_check_prompts/prompt_check_fake_assets.txt", "r") as file:
                prompt_check_fake_assets = file.read()

            content, metadata = query_agent(
                prompt_check_fake_assets,
                (state.get("instructions", "") + "\n\n" + text),
                state.get("exam_author_model"),
            )

            try:
                check = json.loads(content)
                if check.get("fake_website_or_news") == "Y":
                    errors.append("Detected fake website/news reference.")
                if check.get("fake_image") == "Y":
                    errors.append("Detected missing image.")
            except json.JSONDecodeError:
                errors.append("LLM response could not be parsed for fake assets.")

        except Exception as e:
            errors.append(f"Error during fake asset check: {e}")

    # Final state update
    if errors:
        state["errors"].extend(errors)
        state["fail_reasons"] = "\n".join(errors)
        state["check_materials_structure_passed"] = False
        state["next_node"] = "node_materials"  # retry generation
        print("❌ Materials structure check failed:", errors)
    else:
        state["check_materials_structure_passed"] = True
        state["next_node"] = "node_submission"
        print("✅ Materials check passed.")

    return state


def node_submission(state: ExamState) -> ExamState:
    """
    Generates submission requirements for the exam using a language model.

    Formats a submission prompt with the previously generated overview, instructions,
    and materials, then queries the model. Stores the result and metadata in the state.

    Args:
        state (ExamState): A dictionary-like object containing keys such as:
            - 'overview'
            - 'instructions'
            - 'materials_candidate'
            - 'system_prompt'
            - 'exam_author_model'
            - 'metadata' (dict)
            - 'sequence' (list)
            - 'counter' (int)
            - 'task_id' (str)

    Returns:
        ExamState: Updated state including:
            - 'submission': the generated submission requirements
            - 'metadata["submission"]': usage metadata returned by the model
            - 'exam': concatenated instructions + materials + submission
    """
    print("🧠 Creating exam submission requirements...")

    # Load submission prompt template
    prompt_path = "../prompts/exam_generation_prompts/prompt_submission.txt"
    with open(prompt_path, "r", encoding="utf-8") as file:
        prompt_template = file.read()

    # Format prompt with overview, instructions, and materials
    prompt = prompt_template.format(
        answer_overview=state["overview"],
        answer_instructions=state["instructions"],
        answer_materials=state["materials_candidate"],
    )

    # Query the exam author model
    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state["submission"] = content
    state["metadata"]["submission"] = metadata

    # Update sequence and exam concatenation
    state["sequence"].append("submission")
    state["exam"] = state["instructions"] + state["materials_candidate"] + state["submission"]
    state["counter"] += 1

    # Build safe file paths
    safe_task_id = state["task_id"].replace(".", "_")
    base_path = os.path.join("../data/exams/basic", state["exam_author_model"], safe_task_id)
    os.makedirs(base_path, exist_ok=True)

    # Save prompt and output files
    files_to_save = {
        f"{state['counter']}_prompt_submission.txt": prompt,
        f"{state['counter']}_content_submission.txt": state["submission"],
        f"{state['counter']}_content_exam.txt": state["exam"],
    }

    for filename, content_to_write in files_to_save.items():
        with open(os.path.join(base_path, filename), "w", encoding="utf-8") as f:
            f.write(str(content_to_write))

    return state


def node_check_submission_structure(state: dict) -> dict:
    """
    Validates that the submission is a single valid JSON object.

    Updates state with:
        - submission
        - check_submission_structure_passed
        - errors / fail_reasons
        - next_node
    """
    import json
    import re

    print("🔍 Validating submission template...")

    text = state.get("submission", "").strip()
    if text.lower() == "no submission required":
        state["submission"] = "No submission required"
        state["check_submission_structure_passed"] = True
        state["next_node"] = "node_evaluation"
        print("✅ Submission template check passed (no submission required).")
        return state

    # Remove triple backticks (```json ... ```)
    cleaned_text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.IGNORECASE).strip()
    state["submission"] = cleaned_text

    # Attempt to parse JSON
    try:
        parsed_json = json.loads(cleaned_text)
        # Must be a JSON object at top level
        if not isinstance(parsed_json, dict):
            raise ValueError("Top-level JSON must be an object (dict).")
        state["check_submission_structure_passed"] = True
        state["next_node"] = "node_evaluation"
        print("✅ Submission template check passed.")
    except (json.JSONDecodeError, ValueError) as e:
        state.setdefault("errors", []).append(f"Invalid JSON: {e}")
        state["fail_reasons"] = f"Invalid JSON: {e}"
        state["check_submission_structure_passed"] = False
        state["next_node"] = "node_submission"  # retry
        print("❌ Submission template check failed:", e)

    return state


def node_evaluation(state: ExamState) -> ExamState:
    """
    Generates the exam answer key, explanations, and passing criteria using a language model.

    The prompt is formatted using the exam overview, exam content, and evaluator-only materials.
    The generated answer key and metadata are stored in the state. The function also increments
    a counter to track the number of answer key generations.

    Args:
        state (ExamState): A dictionary-like object containing keys such as:
            - 'overview': exam overview
            - 'exam': full exam content
            - 'materials_evaluator': evaluator-only materials
            - 'system_prompt': system prompt for the LLM
            - 'exam_author_model': model to use
            - 'answer_key_count': counter for generated answer keys
            - 'metadata': nested dictionary for storing LLM metadata

    Returns:
        ExamState: The updated state with:
            - 'answer_key': generated answer key JSON and explanations
            - 'metadata["answer_key"]': LLM usage metadata
            - 'answer_key_count': incremented by 1
    """
    print("🧠 Generating exam answer key...")

    # Load the prompt template
    with open(
        "../prompts/exam_generation_prompts/prompt_evaluation.txt", "r", encoding="utf-8"
    ) as file:
        prompt_template = file.read()

    # Fill in the template
    prompt = prompt_template.format(
        answer_overview=state.get("overview", ""),
        exam=state.get("exam", ""),
        materials_evaluator=state.get("materials_evaluator", ""),
        submission_template=state.get("submission", ""),
    )

    # Query the model
    content, metadata = query_agent(
        system_prompt=state.get("system_prompt"),
        user_prompt=prompt,
        model=state.get("exam_author_model"),
    )

    # Update state
    state["evaluation"] = content
    state["metadata"]["evaluation"] = metadata
    state["sequence"].append("evaluation")

    # Increment counter
    state["answer_key_count"] = state.get("answer_key_count", 0) + 1

    # Save prompt and generated content
    safe_task_id = state["task_id"].replace(".", "_")
    base_path = os.path.join("../data/exams/basic", state["exam_author_model"], safe_task_id)
    counter = state["answer_key_count"]

    with open(
        os.path.join(base_path, f"{state['counter']}_prompt_evaluation.txt"), "w", encoding="utf-8"
    ) as f:
        f.write(str(prompt))

    with open(
        os.path.join(base_path, f"{state['counter']}_content_evaluation.txt"),
        "w",
        encoding="utf-8",
    ) as f:
        f.write(str(state["evaluation"]))

    return state


def node_check_submission_vs_answer_key(state: dict) -> dict:
    """
    Validates that the candidate submission template does not contain
    any values from the answer key (pre-filled answers).

    Assumes `submission` and `answer_key` are valid JSON strings.

    Updates state with:
        - check_submission_vs_answer_key_passed
        - errors / fail_reasons
        - next_node
    """
    import json

    print("🔍 Checking submission template against answer key...")

    submission_json = state["submission"]

    answer_key_json = state["answer_key"]

    # Flatten both submission and answer key into leaf values
    def flatten_json(obj):
        values = []
        if isinstance(obj, dict):
            for v in obj.values():
                values.extend(flatten_json(v))
        elif isinstance(obj, list):
            for item in obj:
                values.extend(flatten_json(item))
        else:
            values.append(obj)
        return values

    submission_values = flatten_json(submission_json)
    answer_values = set(flatten_json(answer_key_json))

    # Count prefilled values
    prefilled = [
        v for v in submission_values if v in answer_values and v not in (None, "", [], {})
    ]
    total_fields = len([v for v in submission_values if v not in (None, "", [], {})])

    # Allow up to 30% prefilled fields; fail if more
    fail_threshold = 0.3
    fail_ratio = len(prefilled) / total_fields if total_fields > 0 else 0

    if fail_ratio > fail_threshold:
        err_msg = (
            f"❌ Submission contains {len(prefilled)} prefilled fields out of {total_fields} "
            f"({fail_ratio * 100:.1f}% > 30%)."
        )
        state.setdefault("errors", []).append(err_msg)
        state["fail_reasons"] = err_msg
        state["check_submission_vs_answer_key_passed"] = False
        state["next_node"] = "node_submission"  # retry
        print(err_msg)
    else:
        state["check_submission_vs_answer_key_passed"] = True
        state["next_node"] = "node_grading"
        print("✅ Submission does not contain significant prefilled answers.")

    return state


def node_check_answer_key_format(state: dict) -> dict:
    """
    Validates and extracts the JSON answer key and evaluator instructions from the model output.

    Steps:
    - Extract the JSON object (inside ```json ... ```)
    - Validate that it's valid JSON
    - Separate evaluator explanations/instructions
    - Save both parts into the state:
        - state["answer_key"]: parsed JSON object
        - state["evaluator_instructions"]: remaining explanatory text
    - Set control flags for pipeline flow:
        - check_answer_key_format_passed
        - errors / fail_reasons
        - next_node
    """

    print("🔍 Validating and extracting answer key JSON...")

    raw_text = state.get("evaluation", "").strip()
    errors = []
    answer_key_json = None
    evaluator_text = ""

    # 1️⃣ Try to extract text between triple backticks
    code_blocks = re.findall(r"```(?:json)?\s*([\s\S]*?)\s*```", raw_text, re.IGNORECASE)

    # Fallback: detect any standalone top-level JSON-like structure
    if not code_blocks:
        json_like = re.search(r"(\{[\s\S]+\})", raw_text)
        if json_like:
            code_blocks = [json_like.group(1)]

    # 2️⃣ Attempt to parse JSON
    for block in code_blocks:
        try:
            answer_key_json = json.loads(block)
            break
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON: {e}")

    # 3️⃣ Extract evaluator text (everything outside JSON block)
    if answer_key_json:
        json_text_pattern = re.escape(block)
        evaluator_text = re.sub(json_text_pattern, "", raw_text).strip()
        evaluator_text = re.sub(r"```(?:json)?|```", "", evaluator_text).strip()

    # 4️⃣ Update state based on result
    if not answer_key_json:
        errors.append("No valid JSON answer key found in the model output.")
        state["check_answer_key"] = False
        state["next_node"] = "node_evaluation"  # retry
        state.setdefault("errors", []).extend(errors)
        state["fail_reasons"] = "\n".join(errors)
        print("❌ Answer key validation failed:", errors)
    else:
        # Success — store extracted components
        state["answer_key"] = answer_key_json
        state["evaluator_instructions"] = evaluator_text
        state["check_answer_key"] = True
        state["next_node"] = "node_check_submission_vs_answer_key"
        print("✅ Answer key JSON extracted and validated.")

        # Optional: Save extracted files for traceability
        safe_task_id = state.get("task_id", "unknown").replace(".", "_")
        model_name = state.get("exam_author_model", "unknown_model")
        base_path = os.path.join("../data/exams/basic", model_name, safe_task_id)
        os.makedirs(base_path, exist_ok=True)

        counter = state.get("counter", 0)
        # with open(
        #     os.path.join(base_path, f"{counter}_answer_key.json"), "w", encoding="utf-8"
        # ) as f:
        #     json.dump(state["answer_key"], f, indent=2, ensure_ascii=False)

        with open(
            os.path.join(base_path, f"{counter}_evaluator_instructions.txt"), "w", encoding="utf-8"
        ) as f:
            f.write(state["evaluator_instructions"])

    return state


def node_grading(state: ExamState) -> ExamState:
    """
    Generates a Python grading script for evaluating candidate exam submissions.

    The prompt combines exam components (overview, exam content, evaluator materials, and evaluation
    criteria) and instructs the model to produce a runnable grading script. The generated script and
    its metadata are saved into the exam state and stored locally for traceability.

    Updates:
        - state["grading"]: full model-generated grading script (as text)
        - state["metadata"]["grading"]: LLM metadata for the grading step
        - state["sequence"]: appends "grading"
    """
    print("🧠 Generating grading script...")

    try:
        # Load the grading prompt template
        with open(
            "../prompts/exam_generation_prompts/prompt_grading.txt", "r", encoding="utf-8"
        ) as file:
            prompt_template = file.read()

        # Fill in placeholders
        prompt = prompt_template.format(
            overview=state.get("overview", ""),
            exam=state.get("exam", ""),
            materials_evaluator=state.get("materials_evaluator", ""),
            evaluation=state.get("evaluation", ""),
        )

        # Query the LLM
        content, metadata = query_agent(
            system_prompt=state.get("system_prompt"),
            user_prompt=prompt,
            model=state.get("exam_author_model"),
        )

        # Update state
        state["grading"] = content
        state["metadata"]["grading"] = metadata
        state["sequence"].append("grading")
        state["counter"] = state.get("counter", 0) + 1

        # Build file paths
        safe_task_id = state["task_id"].replace(".", "_")
        base_path = os.path.join("../data/exams/basic", state["exam_author_model"], safe_task_id)

        os.makedirs(base_path, exist_ok=True)

        # Save prompt and generated script
        with open(
            os.path.join(base_path, f"{state['counter']}_prompt_grading.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(prompt)
        with open(
            os.path.join(base_path, f"{state['counter']}_content_grading.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(content)

        print("✅ Grading script generated successfully.")

    except Exception as e:
        err_msg = f"❌ Error generating grading script: {e}"
        print(err_msg)
        state.setdefault("errors", []).append(err_msg)

    return state


def node_save_eval_and_answer(state: ExamState) -> ExamState:
    """
    Saves the generated evaluation outputs for an exam task.

    Actions:
      1. Saves the Python grading script from state["grading"] as `task_evaluation.py`.
      2. Saves the validated answer key JSON from state["answer_key"] as `answer_key.json`.

    Updates:
        - state["answer_key"]: parsed JSON (confirmed via extract_and_save_json)
        - Appends "save_eval_and_answer" to state["sequence"]
        - Logs any errors in state["errors"]
    """
    print("💾 Saving grading script and answer key...")

    task_id = state.get("task_id", "unknown_task")
    level = state.get("level", "basic")
    model = state.get("exam_author_model", "unknown_model")

    # Define folder paths
    base_path = os.path.join("../data/exams", level, model, task_id.replace(".", "_"))
    os.makedirs(base_path, exist_ok=True)

    try:
        # 1. Save the grading script
        extract_and_save_python_script(
            script_text=state.get("grading", ""),
            folder=base_path,
            filename="task_evaluation.py",
        )

        with open(os.path.join(base_path, "answer_key.json"), "w", encoding="utf-8") as f:
            json.dump(state.get("answer_key"), f, indent=2, ensure_ascii=False)

        # Update state
        print(f"✅ Grading script and answer key saved successfully for task '{task_id}'.")

    except Exception as exc:
        err_msg = f"❌ Error saving assets for task '{task_id}': {exc}"
        print(err_msg)
        state.setdefault("errors", []).append(err_msg)

    # Log accumulated errors (if any)
    # if state.get("errors"):
    #   print("⚠️ Errors encountered during save:", state["errors"])

    # Update sequence tracking
    state["sequence"].append("save_eval_and_answer")

    return state


def node_check_answer_key(state: dict) -> dict:
    """
    Validates that the grading script correctly assigns 100% to the official answer key.

    Process:
      1. Locates the saved grading script (task_evaluation.py) and answer key.
      2. Runs the script via subprocess: `python task_evaluation.py answer_key.json answer_key.json`
      3. Loads `test_results.json` and checks the overall score.
      4. Marks success if the score >= threshold (default 100).

    Updates:
      - state["key_grade"]: numeric score or NaN on failure
      - state["check_answer_key"]: True/False
      - state["next_node"]: next step in the pipeline
      - state["errors"]: list of error messages
      - state["sequence"]: logs this step and score

    Notes:
      - Assumes node_save_eval_and_answer already saved the script and key.
      - Uses robust error capture for subprocess and JSON handling.
    """
    import json
    import os

    print("🔎 Checking answer key grading consistency...")

    task_id = state.get("task_id", "unknown_task")
    level = state.get("level", "basic")
    model = state.get("exam_author_model", "unknown_model")
    threshold = state.get("key_grade_threshold", 100)
    base_path = os.path.join("../data/exams", level, model, task_id.replace(".", "_"))
    errors = []

    script_path = os.path.join(base_path, "task_evaluation.py")
    key_path = os.path.join(base_path, "answer_key.json")
    result_path = os.path.join(base_path, "test_results.json")

    # --- Basic existence checks ---
    for required in [script_path, key_path]:
        if not os.path.exists(required):
            msg = f"❌ Missing required file: {required}"
            print(msg)
            errors.append(msg)

    if errors:
        state.setdefault("errors", []).extend(errors)
        state["check_answer_key"] = False
        state["key_grade"] = np.nan
        state["next_node"] = "node_evaluation"  # regenerate grading script
        return state

    # --- Run the grading script ---
    try:
        result = subprocess.run(
            ["python", "task_evaluation.py", "answer_key.json", "answer_key.json"],
            cwd=base_path,
            check=True,
            capture_output=True,
            text=True,
        )
        print("✅ Grading script executed successfully.")
        if result.stdout.strip():
            print("📄 Script output:\n", result.stdout.strip())
        state["next_node"] = "node_end"  # default to end unless issues found

    except subprocess.CalledProcessError as e:
        msg = f"❌ Grading script failed (code {e.returncode}): {e.stderr.strip()}"
        print(msg)
        errors.append(msg)
        state.setdefault("errors", []).extend(errors)
        state["check_answer_key"] = False
        state["key_grade"] = np.nan
        state["next_node"] = "node_evaluation"
        return state

    except FileNotFoundError:
        msg = f"❌ Grading script not found in {base_path}."
        print(msg)
        errors.append(msg)
        state.setdefault("errors", []).extend(errors)
        state["check_answer_key"] = False
        state["key_grade"] = np.nan
        state["next_node"] = "node_evaluation"
        return state

    # --- Parse test_results.json ---
    try:
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        score = data.get("overall_score")
        if score is None:
            msg = "❌ No 'overall_score' field found in test_results.json."
            print(msg)
            errors.append(msg)
            raise ValueError("Missing overall_score")

        state["key_grade"] = float(np.round(score, 2))
        print(f"🏁 Answer key self-grade: {state['key_grade']}%")

        # --- Check pass/fail ---
        if state["key_grade"] >= threshold * 0.99:  # passes self-test
            state["check_answer_key"] = True
            state["next_node"] = "node_end"
            print(
                f"✅ Answer key validated (passes self-test): {state['key_grade']} >= {threshold}."
            )
        else:
            state["check_answer_key"] = False
            msg = f"⚠️ Answer key self-test scored below threshold ({state['key_grade']} < {threshold})."
            print(msg)
            errors.append(msg)
            # Route to diagnostic node to determine root cause
            state["next_node"] = "node_diagnose_key_failure"

    except FileNotFoundError:
        msg = f"❌ test_results.json not found at {result_path}."
        print(msg)
        errors.append(msg)
        state["key_grade"] = np.nan
        state["check_answer_key"] = False
        state["next_node"] = "node_evaluation"

    except json.JSONDecodeError as e:
        msg = f"❌ Invalid JSON in test_results.json: {e}"
        print(msg)
        errors.append(msg)
        state["key_grade"] = np.nan
        state["check_answer_key"] = False
        state["next_node"] = "node_evaluation"

    except Exception as e:
        msg = f"❌ Unexpected error checking answer key: {e}"
        print(msg)
        errors.append(msg)
        state["key_grade"] = np.nan
        state["check_answer_key"] = False
        state["next_node"] = "node_evaluation"

    # --- Update final state ---
    state.setdefault("errors", []).extend(errors)
    state["sequence"].append(f"check_answer_key_{state.get('key_grade', 'NaN')}")

    return state


def node_validate_grading_script(state: dict) -> dict:
    """
    Validates that the generated grading script (state['grading']) contains:
      - A valid Python code block (enclosed in triple backticks)
      - Syntactically correct Python code
      - Required functionality according to grading prompt:
          * Command-line arguments via sys.argv
          * JSON file handling (load/save)
          * Writes to 'test_results.json'
          * Robust error handling
          * Deterministic behavior (no randomness or external dependencies)
          * Only standard library imports

    If valid:
        - sets state["check_grading_script"] = True
        - sets state["next_node"] = "node_save_eval_and_answer"

    If invalid:
        - sets state["check_grading_script"] = False
        - sets state["next_node"] = "node_grading"

    Returns:
        Updated state dict.
    """
    import ast
    import re

    print("🔍 Validating grading script...")

    state.setdefault("errors", [])
    grading_text = state.get("grading", "")
    local_errors = []  # local to this validation step

    # --- 1. Extract Python code block ---
    code_block_match = re.search(r"```python(.*?)```", grading_text, re.DOTALL | re.IGNORECASE)
    if not code_block_match:
        err = "❌ No valid Python code block (```python ... ```) found in grading script."
        local_errors.append(err)
        print(err)
    else:
        code_block = code_block_match.group(1).strip()

        # --- 2. Validate syntax ---
        try:
            ast.parse(code_block)
        except SyntaxError as e:
            err = f"❌ Grading script contains invalid Python syntax: {e}"
            local_errors.append(err)
            print(err)

        # --- 3. Required structural elements ---
        required_keywords = [
            "import json",
            "import sys",
            "sys.argv",
            "test_results.json",
            "open(",
            "json.load(",
            "json.dump(",
        ]
        missing = [kw for kw in required_keywords if kw not in code_block]
        if missing:
            err = f"⚠️ Missing required components in grading script: {missing}"
            local_errors.append(err)
            print(err)

        # --- 4. Check forbidden / non-deterministic imports ---
        # forbidden = ["import random", "import numpy", "requests", "openai", "time.sleep"]
        # bad_found = [kw for kw in forbidden if kw in code_block]
        # if bad_found:
        #     err = f"⚠️ Forbidden or non-deterministic imports detected: {bad_found}"
        #     local_errors.append(err)
        #     print(err)

        # --- 5. Check for CLI entrypoint ---
        if "if __name__" not in code_block:
            err = "⚠️ Missing main guard (`if __name__ == '__main__':`)."
            local_errors.append(err)
            print(err)

    # --- 6. Evaluate results ---
    if not local_errors:
        print("✅ Grading script validated successfully.")
        state["check_grading_script"] = True
        state["next_node"] = "node_save_eval_and_answer"
    else:
        print("⚠️ Grading script validation completed with warnings/errors.")
        state["check_grading_script"] = False
        state["next_node"] = "node_grading"
        state["errors"].extend(local_errors)

    # --- 7. Log in sequence and return ---
    state["sequence"].append("validate_grading_script")
    return state


def node_diagnose_key_failure(state: dict) -> dict:
    """
    Uses an LLM call to diagnose why the answer key failed its self-grade and decide what to regenerate.
    Also implements retry and fallback control:
      - If called 3 times → revert to start
      - If the revert happens 3 times total → abort pipeline
    """
    import json
    import os

    print("🔍 Running LLM-based diagnosis for key failure...")

    # --- Retry tracking ---
    state.setdefault("diagnose_key_failure_calls", 0)
    state["diagnose_key_failure_calls"] += 1
    print(f"🔁 Diagnose node call count: {state['diagnose_key_failure_calls']}")

    # --- If this node has been called 3 times, revert ---
    if state["diagnose_key_failure_calls"] >= 3:
        state.setdefault("diagnose_reverts", 0)
        state["diagnose_reverts"] += 1
        print(
            f"⚠️ Diagnosis repeated 3 times. Reverting to start. Total reverts: {state['diagnose_reverts']}"
        )

        # Reset the counter for this phase
        state["diagnose_key_failure_calls"] = 0

        # If we've reverted 3 times in total, abort
        if state["diagnose_reverts"] >= 3:
            print("❌ Pipeline aborted after 3 reverts.")
            state["next_node"] = "node_end"
            state["abort_reason"] = "Repeated failure to self-correct after 3 diagnostic cycles."
            state["sequence"].append("diagnose_key_failure_abort")
            return state

        # Otherwise, restart from beginning
        state["next_node"] = "node_overview"
        state["sequence"].append("diagnose_key_failure_restart")
        return state

    # --- Build contextual prompt ---
    prompt = f"""
    You are an expert in automated exam validation systems.

    The system generated an exam with the following components:
    <exam_overview>
    {state.get("overview", "")}
    </exam_overview>

    <exam_content>
    {state.get("exam", "")}
    </exam_content>

    <evaluation_instructions>
    {state.get("evaluation", "")}
    </evaluation_instructions>

    The system also generated:
    <submission_template>
    {state.get("submission_candidate", "")}
    </submission_template>

    <answer_key>
    {json.dumps(state.get("answer_key", {}), indent=2, ensure_ascii=False)}
    </answer_key>

    <grading_script>
    {state.get("grading", "")}
    </grading_script>

    When grading the answer key against itself, the score was only {state.get("key_grade", "unknown")}%.

    The resulting grading output (if available):
    <test_results>
    {state.get("test_results", "")}
    </test_results>

    Based on this information, determine which component is *most likely at fault* for the failed self-grade,
    and what should be regenerated next.

    Respond ONLY in the following JSON format:
    {{
        "diagnosis": "grading_script_issue" | "submission_template_issue" | "answer_key_content_issue",
        "rationale": "short, clear explanation",
        "recommendation": "which component should be regenerated"
    }}
    """
    # --- LLM call ---
    try:
        content, metadata = query_agent(
            system_prompt="You are a meticulous diagnostic assistant for exam generation systems.",
            user_prompt=prompt,
            model=state.get("exam_author_model"),
        )

    except Exception as e:
        print(f"❌ LLM call failed: {e}")
        state.setdefault("errors", []).append(f"LLM call failed: {e}")
        state["diagnosis"] = "grading_script_issue"
        state["diagnosis_rationale"] = "Fallback due to LLM failure."
        state["recommendation"] = "rework grading script"
        state["next_node"] = "node_grading"
        state["sequence"].append("diagnose_key_failure_fallback")
        return state

    # --- Parse LLM response ---
    try:
        diagnosis_result = json.loads(content)
    except json.JSONDecodeError:
        print("⚠️ LLM did not return valid JSON; defaulting to grading_script_issue.")
        diagnosis_result = {
            "diagnosis": "grading_script_issue",
            "rationale": "Model output could not be parsed; assuming grading issue.",
            "recommendation": "rework grading script",
        }

    # --- Update state ---
    state["diagnosis"] = diagnosis_result.get("diagnosis", "grading_script_issue")
    state["diagnosis_rationale"] = diagnosis_result.get("rationale", "")
    state["recommendation"] = diagnosis_result.get("recommendation", "")

    # --- Determine next node ---
    next_node_map = {
        "grading_script_issue": "node_grading",
        "answer_key_content_issue": "node_evaluation",
        "submission_template_issue": "node_submission",
    }
    state["next_node"] = next_node_map.get(state["diagnosis"], "node_grading")

    print(f"Diagnosis: {state['diagnosis']} → next: {state['next_node']}")
    print(f"Rationale: {state['diagnosis_rationale']}")

    # --- Save LLM output for traceability ---
    safe_task_id = state.get("task_id", "unknown").replace(".", "_")
    model_name = state.get("exam_author_model", "unknown_model")
    base_path = os.path.join(
        "../data/exams", state.get("level", "basic"), model_name, safe_task_id
    )
    os.makedirs(base_path, exist_ok=True)
    with open(os.path.join(base_path, "diagnosis_llm_output.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "prompt_summary": "LLM diagnosis of answer key self-test failure",
                "response_raw": content,
                "parsed_result": diagnosis_result,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # --- Append to sequence ---
    state["sequence"].append("diagnose_key_failure")
    return state


def node_assess_exam_sanity(state: ExamState) -> ExamState:
    """
    Performs a final sanity check on the generated exam.
    Updates:
        - exam_sanity_passed (bool)
        - exam_sanity_report (dict)
        - next_node
    """
    print("🔎 Performing final exam sanity assessment...")

    errors = []
    exam_report = {
        "makes_sense": None,  # True/False
        "explanation": "",  # short textual explanation
    }

    # LLM-based sanity check (optional)
    try:
        with open("../prompts/sanity_check_prompts/prompt_check_sense.txt", "r") as file:
            sanity_prompt = file.read()
        # Query agent with exam content (overview + instructions + materials)
        content, metadata = query_agent(
            sanity_prompt,
            state.get("overview", "")
            + "\n\n"
            + state.get("instructions", "")
            + "\n\n"
            + str(state.get("submission_template", {}))
            + "\n\n"
            + state.get("materials_candidate", ""),
            state.get("exam_author_model"),
        )
        try:
            llm_check = json.loads(content)
            if "makes_sense" in llm_check and "explanation" in llm_check:
                exam_report.update(
                    {
                        "makes_sense": llm_check["makes_sense"],
                        "explanation": llm_check["explanation"],
                    }
                )
            else:
                errors.append("LLM sanity check output missing required fields.")
        except json.JSONDecodeError:
            errors.append("LLM response could not be parsed as JSON for exam sanity check.")
    except Exception as e:
        errors.append(f"Error during LLM sanity check: {e}")

    # Fallback if LLM check failed or not used
    if not exam_report["makes_sense"]:
        exam_report["makes_sense"] = False
        exam_report["explanation"] = "; ".join(errors) if errors else "Unknown issue detected."

    # Update final state
    state["check_makes_sense"] = exam_report["makes_sense"]
    state["exam_sanity_report"] = exam_report
    # Update next node based on sanity check
    if exam_report["makes_sense"]:
        state["next_node"] = "node_assess_exam_sanity"
        print("✅ Exam sanity check passed.")

    else:
        state["next_node"] = "node_overview"

        # Increment diagnostic revert counter
        state["diagnose_reverts"] = state.get("diagnose_reverts", 0) + 1

        # Check if we exceeded the allowed number of reverts
        if state["diagnose_reverts"] >= 3:
            print("❌ Pipeline aborted after 3 reverts.")
            state["next_node"] = "node_end"
            state["abort_reason"] = "Repeated failure to self-correct after 3 diagnostic cycles."
            state["sequence"].append("diagnose_key_failure_abort")
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

    state["sequence"].append("end" + str(state["exam"][:30]))
    return state


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        # model = "claude-sonnet-4-20250514"
        model = "gpt-5-mini-2025-08-07"
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
        tasks_file = (
            f"../data/filtered_tasks/{occupation_group.replace(' ', '_').lower()}_{core_label}.csv"
        )
        df_tasks = pd.read_csv(tasks_file)[0:10]
        print("Overall number of tasks: ", df_tasks.shape)

        # folder for saving results
        folder_path = f"../data/exams/{level}/{model}/"
        if Path(
            os.path.join(folder_path, f"{occupation_group.replace(' ', '_').lower()}_exams.csv")
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
        # Initialize an empty list to store result states
        result_states = []
        graph_builder = StateGraph(ExamState)

        graph_builder.add_node("construct_system_prompt", node_system_prompt)
        graph_builder.add_node("node_overview", node_overview)
        graph_builder.add_node("node_instructions", node_instructions)
        graph_builder.add_node(
            "node_check_instructions_structure", node_check_instructions_structure
        )

        # after generating instructions

        graph_builder.add_node("node_materials", node_materials)
        graph_builder.add_node("node_check_materials_structure", node_check_materials_structure)
        graph_builder.add_node("node_submission", node_submission)
        graph_builder.add_node("node_check_submission_structure", node_check_submission_structure)

        # graph_builder.add_node("node_cleanup", node_cleanup)
        graph_builder.add_node("node_evaluation", node_evaluation)
        graph_builder.add_node("node_check_answer_key_format", node_check_answer_key_format)
        graph_builder.add_node("node_grading", node_grading)
        graph_builder.add_node("node_validate_grading_script", node_validate_grading_script)

        graph_builder.add_node("node_save_eval_and_answer", node_save_eval_and_answer)
        graph_builder.add_node("node_check_answer_key", node_check_answer_key)
        graph_builder.add_node(
            "node_check_submission_vs_answer_key", node_check_submission_vs_answer_key
        )
        graph_builder.add_node("node_diagnose_key_failure", node_diagnose_key_failure)
        graph_builder.add_node("node_assess_exam_sanity", node_assess_exam_sanity)
        # # graph_builder.add_node(
        # #     "node_check_inconsistencies_duplicates",
        # #     node_check_inconsistencies_duplicates,
        # # )
        # graph_builder.add_node("node_check_exam_feasibility", node_check_exam_feasibility)
        # graph_builder.add_node("node_check_answer_coverage", node_check_answer_coverage)
        # graph_builder.add_node("node_overall_makes_sense", node_overall_makes_sense)
        # graph_builder.add_node("node_improve", node_improve)
        graph_builder.add_node("node_end", node_end)
        # graph_builder.add_node("node_pause_before_evaluation", node_pause_before_evaluation)

        # Add edges the graph
        graph_builder.add_edge(START, "construct_system_prompt")
        graph_builder.add_edge("construct_system_prompt", "node_overview")
        graph_builder.add_edge("node_overview", "node_instructions")
        graph_builder.add_edge("node_instructions", "node_check_instructions_structure")
        graph_builder.add_conditional_edges(
            "node_check_instructions_structure", lambda state: state["next_node"]
        )
        graph_builder.add_edge("node_materials", "node_check_materials_structure")
        graph_builder.add_conditional_edges(
            "node_check_materials_structure", lambda state: state["next_node"]
        )
        graph_builder.add_edge("node_submission", "node_check_submission_structure")
        graph_builder.add_conditional_edges(
            "node_check_submission_structure", lambda state: state["next_node"]
        )
        graph_builder.add_edge("node_evaluation", "node_check_answer_key_format")
        graph_builder.add_conditional_edges(
            "node_check_answer_key_format", lambda state: state["next_node"]
        )
        graph_builder.add_conditional_edges(
            "node_check_submission_vs_answer_key", lambda state: state["next_node"]
        )
        graph_builder.add_edge("node_grading", "node_validate_grading_script")
        graph_builder.add_conditional_edges(
            "node_validate_grading_script", lambda state: state["next_node"]
        )
        graph_builder.add_edge("node_save_eval_and_answer", "node_check_answer_key")
        graph_builder.add_conditional_edges(
            "node_check_answer_key", lambda state: state["next_node"]
        )
        graph_builder.add_conditional_edges(
            "node_diagnose_key_failure", lambda state: state["next_node"]
        )
        graph_builder.add_conditional_edges(
            "node_assess_exam_sanity", lambda state: state["next_node"]
        )
        # Link it right after materials creation

        # Conditional edge to route based on validation result

        ### Add conditional edges if materials_fake_website or materials_fake_image then end the process
        # If it passes will continue to generatl submissions and grading
        # graph_builder.add_edge("node_submission", "node_cleanup")
        # graph_builder.add_edge("node_cleanup", "node_evaluation")

        # # graph_builder.add_edge("node_pause_before_evaluation", "node_evaluation")

        # graph_builder.add_edge("node_evaluation", "node_overall_makes_sense")

        # graph_builder.add_conditional_edges("node_overall_makes_sense", route_after_sense_check)
        # graph_builder.add_conditional_edges(
        #     "node_check_exam_feasibility", route_after_feasibility_check
        # )
        # graph_builder.add_edge("node_improve", "node_evaluation")

        # # Now check the answer key and how much it scores
        # graph_builder.add_edge("node_grading", "node_save_eval_and_answer")
        # graph_builder.add_edge("node_save_eval_and_answer", "node_check_answer_key")
        # # add conditional edges in case materials for candidate where not extracted

        # graph_builder.add_conditional_edges(
        #     "node_check_answer_coverage", route_after_key_contamination_check
        # )

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
                "materials": "",
                "materials_evaluator": "",
                "materials_candidate": "",
                "submission": "",
                "grading": "",
                "evaluation": "",
                "answer_key": "",
                "evaluator_instructions": "",
                "errors": [],
                "key_grade_threshold": 100,
                "key_grade": 0.0,
                "check_makes_sense": True,
                "explanation_overall_makes_sense": "",
                "check_submission_vs_answer_key_passed": False,
                "metadata": {},
                "education": row["education"],
                "check_answer_key": False,
                "check_grading_script": False,
                "fail_reasons": "",
                "check_feasible": True,
                "explanation_feasible": "",
                "sequence": [],
                "check_instructions_materials": True,
                "alter_target": None,
                "counter": 0,
                "diagnosis": "",
                "diagnosis_rationale": "",
                "recommendation": "",
                "diagnose_reverts": 0,
                "abort_reason": "",
                "diagnose_key_failure_calls": 0,
                "exam_sanity_report": "",
            }

            cumulative_state = init_state.copy()  # start with your initial state
            try:
                for event in graph.stream(
                    cumulative_state,
                    stream_mode="values",
                    config={"recursion_limit": 150},
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
                df_result_states = pd.concat([df_existing, df_result_states], ignore_index=True)
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
                    folder_path + f"{occupation_group.replace(' ', '_').lower()}_exams.csv",
                    index=False,
                )
