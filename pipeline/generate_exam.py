#### Note uses python 3.9 environment (newenv)
from typing import TypedDict
import pandas as pd
import random
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import anthropic
import regex as re
import sys
import ast
import json
import subprocess
import numpy as np

from typing import Annotated
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI  # or your equivalent import
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain.schema import SystemMessage, HumanMessage
from langgraph.graph import START
from langgraph.graph import END
from IPython.display import Image, display


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

def join_items(items, conj='and'):
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
        return ', '.join(items[:-1]) + f' {conj} ' + items[-1]
    return ''

def build_system_prompt(occupation, task_description, task_id, required_tools, required_materials, level, template):
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
            f"{join_items(required_tools, conj='and')}" )
    else:
        tools_instructions = "- The candidate does not have access to any special tools."
    # Materials
    if required_materials:
        materials_instructions = (
            f"- The candidate can also be given digital materials such as "
            f"{join_items(required_materials, conj='or')} that must be used for the test.")
    else:
        materials_instructions = "- The candidate does not have access to any additional digital materials."

    return template.format(
        occupation=occupation,
        task_description=task_description,
        task_id=task_id,
        tools_instructions=tools_instructions,
        materials_instructions=materials_instructions,
        level=level
    )


def extract_and_save_python_script(script_text: str, folder: str, filename: str = "task_evaluation.py"):
    """Finds Python code enclosed in triple backticks ```python ...``` and saves it to file.
    useful for extractign grading script
    """
    match = re.search(r'```python(.*?)```', script_text, re.DOTALL)
    if not match:
        raise ValueError("No ```python ... ``` code block found in the grading text.")
    code = match.group(1).strip()

    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, filename)
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(code)
    return code

def extract_and_save_json(json_text: str, folder: str, filename: str = "answer_key.json"):
    """
    Finds JSON enclosed in triple backticks ```json ...``` and saves it to a file.
    """
    match = re.search(r'```json(.*?)```', json_text, re.DOTALL)
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
    materials_all: str
    materials_candidate: str
    submission: str
    evaluation: str
    grading: str
    answer_key: str
    errors: list
    # Boolean flags for validation checks
    check_real_materials: bool
    check_no_internet: bool
    failed_candidate_materials:int
    # Key grade and count how many times below threshold
    key_grade_threshold:float
    key_grade:float
    answer_key_count: int
    check_overall_makes_sense: bool
    explanation_overall_makes_sense:str
    metadata: dict

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
        system_prompt_template =  file.read()
    print('compiling system prompt')
    state["system_prompt"] = build_system_prompt(state['occupation'], state['task_description'], state['task_id'], state['tools'], state['materials'],state['level'],template=system_prompt_template)

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
        prompt_overview =  file.read()
    print('creating exam overview')
    content, metadata = query_agent(state["system_prompt"], prompt_overview, state["exam_author_model"])
    state["overview"] = content
    state['metadata']['overview'] = metadata
    print('metadata',state['metadata']['overview'])
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
    print('creating exam instructions')
    with open("../prompts/exam_generation_prompts/prompt_instructions.txt", "r") as file:
        prompt_template_instructions =  file.read()
    prompt = prompt_template_instructions.format(answer_overview=state["overview"])
    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state["instructions"] = content
    state['metadata']['instructions'] = metadata
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
    print('creating exam materials')
    with open("../prompts/exam_generation_prompts/prompt_materials.txt", "r") as file:
        prompt_template_materials =  file.read()

    prompt = prompt_template_materials.format(answer_overview=state["overview"], answer_instructions=state["instructions"])
    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state['metadata']['materials'] = metadata
    state["materials_all"] = content
    try:
        state["materials_candidate"] = re.search(r'<MATERIALS_FOR_CANDIDATE>(.*?)</MATERIALS_FOR_CANDIDATE>', state["materials_all"], re.DOTALL).group(1)
    except:
        if state["materials_all"] == "No material required":
            state["materials_candidate"] = "No material required"
        else:
            state["materials_candidate"] = "Not extracted"
            # keep track of how many times this fails, if more than 3 then break
            state["failed_candidate_materials"] += 1
            print("materials candidate was not able to be extracted")

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
    print('creating exam submission requirements    ')
    with open("../prompts/exam_generation_prompts/prompt_submission.txt", "r") as file:
        prompt_template_submission =  file.read()
    prompt = prompt_template_submission.format(
        answer_overview=state["overview"],
        answer_instructions=state["instructions"],
        answer_materials=state["materials_all"]
    )
    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state["submission"] = content
    state['metadata']['submission'] = metadata
 
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

    print('creating exam evaluation material')
    with open("../prompts/exam_generation_prompts/prompt_evaluation.txt", "r") as file:
        prompt_template_evaluation =  file.read()
    prompt = prompt_template_evaluation.format(
        answer_overview=state["overview"],
        answer_instructions=state["instructions"],
        answer_materials=state["materials_all"],
        answer_submission=state["submission"]
    )

    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state['evaluation']= content
    state["answer_key_count"] += 1
    state['metadata']['evaluation'] = metadata

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
    print('generating grading script')
    # Note, I modified the prompt so that files are passed as argument
    with open("../prompts/exam_generation_prompts/prompt_grading.txt", "r") as file:
        prompt_template_grading =  file.read()
    prompt = prompt_template_grading.format(
        answer_overview=state["overview"],
        answer_instructions=state["instructions"],
        answer_materials=state["materials_all"],
        answer_submission=state["submission"],
        answer_evaluation=state["evaluation"]
    )

    content, metadata = query_agent(state["system_prompt"], prompt, state["exam_author_model"])
    state["grading"] = content
    state['metadata']['grading'] = metadata

    return state

def node_save_eval_and_answer(state: ExamState) -> ExamState:
    """
    1) Saves the Python grading script from state["grading"] into `task_evaluation.py`
    2) Saves the answer key JSON from state["evaluation"] into `answer_key.json`
    """
    task_id = state["task_id"]
    path =  "../data/exams/" + state["exam_author_model"] + "/"
    folder = task_id.replace(".", "_") 
    if state['level'] =='advanced':
        path = path + "/advanced"
    
    try:
        # 1. Save the Python grading script
        script = extract_and_save_python_script(
            script_text=state["grading"], 
            folder= path + folder, 
            filename="task_evaluation.py"
        )
        # 2. Save the answer key
        key = extract_and_save_json(
            json_text=state["evaluation"], 
            folder=path + folder, 
            filename="answer_key.json"
        )
        state['answer_key'] = key
        print(f"Grading script and answer key saved successfully for task {task_id}.")
    except Exception as exc:
        err_msg = f"Error saving assets for {task_id}: {exc}"
        print(err_msg)
        state["errors"].append(err_msg)
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
    with open("../prompts/sanity_check_prompts/prompt_fake_images.txt", "r") as file:
        prompt_check_fake_image =  file.read()
    content, metadata = query_agent(prompt_check_fake_image, state['instructions'] + state["materials_all"], state["exam_author_model"])
    state['metadata']['check_materials'] = metadata
    if content == "Y":
        state["check_real_materials"] = False
    else:
        state["check_real_materials"] = True
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
    with open("../prompts/sanity_check_prompts/prompt_fake_websites.txt", "r") as file:
        prompt_check_fake_website =  file.read()
    
    content, metadata = query_agent(prompt_check_fake_website, state["instructions"] + state["materials_all"], state["exam_author_model"])
    state['metadata']['check_website'] = metadata

    if content == "Y":
        state["check_no_internet"] = False
    else:
        state["check_no_internet"] = True
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
    errors =[]
    task_id = state["task_id"]
    subfolder = task_id.replace(".", "_") 
    path =  "../data/exams/" + state["exam_author_model"] + "/" 
    folder = task_id.replace(".", "_") 
    if state['level'] =='advanced':
        path = path + "/advanced" + subfolder + "/"
    else:
        path = path + subfolder + "/"
    try:
        result = subprocess.run(
            ["python", "task_evaluation.py", "answer_key.json", "answer_key.json"],
            cwd=path,
            check=True,  # Raise an exception if the command fails
            stderr=subprocess.PIPE,  # Capture stderr
            stdout=subprocess.PIPE   # Capture stdout (if needed)
            )
        print("Script executed successfully.")

        # If the script runs successfully, append None to errors list (no errors)
        errors.append(None)
            # Now get answer key grade
        try:
            # Load the JSON file
            with open(path + 'test_results.json', 'r') as f:
                data = json.load(f)
            
            # Extract the overall_score
            overall_score = data.get("overall_score", None)
            state["key_grade"] = np.round(overall_score)
            return state

        except FileNotFoundError:
            print(path)
            print(f"Error: The file '{path}' was not found.")
            errors.append('no overall score found')
            state["errors"].append(errors)
            state["key_grade"] = np.nan
            return state
        except json.JSONDecodeError:
            print(f"Error: The file '{path}' is not a valid JSON file.")
            errors.append('not json file')
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
        errors.append(e.stderr.decode('utf-8'))  # Append the error message to the errors list
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
    with open("../prompts/sanity_check_prompts/prompt_makes_sense_system.txt", "r") as file:
        system_prompt =  file.read()
    with open("../prompts/sanity_check_prompts/prompt_makes_sense_user.txt", "r") as file:
        user_message =  file.read()
    content, metadata = query_agent(system_prompt, user_message, state["exam_author_model"])
    state['metadata']['check_sense'] = metadata
    response= content.strip()

    try:
        result = json.loads(response)
    except:
        try:
            text = re.search(r'```json(.*?)```', response, re.DOTALL).group(1).strip()
            result = json.loads(text)
            state["check_overall_makes_sense"] = bool(result.get("makes_sense", False))
            state["explanation_overall_makes_sense"] = str(result.get("explanation", ""))
        except json.JSONDecodeError:
            # If the LLM's response isn't valid JSON, mark sense-check as False
            # and store the raw response for debugging.
            state["check_overall_makes_sense"] = False
            state["explanation_overall_makes_sense"] = (
                "Could not parse JSON. Raw LLM response:\n" + response
        )

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
    if state["check_real_materials"] and state["check_no_internet"] and state["key_grade"] >= state["key_grade_threshold"] and state['check_overall_makes_sense']:
        state['exam'] = state['instructions'] + state['materials_candidate'] + state['submission'] 
    else:
        state['exam'] = "Exam not valid"
    return state

#### ###################
# Routing functions
#######################
def route_after_materials_candiate(state: ExamState) -> str:
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
    if state["failed_candidate_materials"] >= 3:
        # if it has consistently failed in generating materials for candiate just end it
        return "node_end"
    elif state["materials_candidate"] == "Not extracted":
        return "node_materials"
    else:
        return "node_check_images"


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

    if state["check_real_materials"] == False:
        return "node_end"
    else:
        return "node_check_websites"

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
    if state["check_no_internet"] == False:
        return "node_end"
    else:
        return "node_submission"


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
    if np.round(state["key_grade"]) == state["key_grade_threshold"]:
        return "node_overall_makes_sense"
    elif state["answer_key_count"] > 3:
        return "node_end"
    else:
        return "node_evaluation"



if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = 'claude-3-7-sonnet-20250219'
    if len(sys.argv) > 2:
        occupation_group = sys.argv[2]
    else:
        occupation_group = "Management Occupations"
    if len(sys.argv) > 3:
        core_label = sys.argv[3]
    else:
        core_label = 'CORE'
    if len(sys.argv) > 4:
        level = sys.argv[4]
    else:
        level = 'basic'
    if level == 'basic':
        level_string = ''
    if level == 'advanced':
        level_string = 'advanced_'
    print('Creating exams for ', occupation_group, 'usiing ',model)
    tasks_file = f'../data/filtered_tasks/{occupation_group.replace(" ", "_").lower()}_{core_label}.csv'
    df_tasks = pd.read_csv(tasks_file)
    df_tasks=df_tasks[0:1]
    print('Overall number of tasks: ', df_tasks.shape)
    df_tasks = df_tasks[['occupation', 'task_description', 'task_id', 'required_tools_standard', 'required_materials_standard']]


    # Initialize an empty list to store result states
    result_states = []
    graph_builder = StateGraph(ExamState)

    graph_builder.add_node("construct_system_prompt", node_system_prompt)
    graph_builder.add_node("node_overview", node_overview)
    graph_builder.add_node("node_instructions", node_instructions)
    graph_builder.add_node("node_materials", node_materials)
    graph_builder.add_node("node_check_images", node_check_materials_fake_image)
    graph_builder.add_node("node_check_websites", node_check_materials_fake_website)
    graph_builder.add_node("node_submission", node_submission)
    graph_builder.add_node("node_evaluation", node_evaluation)
    graph_builder.add_node("node_grading", node_grading)
    graph_builder.add_node("node_save_eval_and_answer", node_save_eval_and_answer)
    graph_builder.add_node("node_check_answer_key", node_check_answer_key)
    graph_builder.add_node("node_overall_makes_sense", node_overall_makes_sense)
    graph_builder.add_node("node_end", node_end)
    #graph_builder.add_node("node_pause_before_evaluation", node_pause_before_evaluation)

    #Add edges the graph
    graph_builder.add_edge(START, "construct_system_prompt")
    graph_builder.add_edge("construct_system_prompt", "node_overview")
    graph_builder.add_edge("node_overview", "node_instructions")
    ### NOTE - probably add conditional edge depending on whether materials are required
    graph_builder.add_edge("node_instructions", "node_materials")
    # add conditional edges in case materials for candidate where not extracted
    graph_builder.add_conditional_edges("node_materials", route_after_materials_candiate)
    ### Add conditional edges if materials_fake_website or materials_fake_image then end the process
    graph_builder.add_conditional_edges("node_check_images", route_after_image_check)
    graph_builder.add_conditional_edges("node_check_websites", route_after_internet_check)
    # If it passes will continue to generatl submissions and grading
    graph_builder.add_edge("node_submission", 'node_evaluation')
    # graph_builder.add_edge("node_pause_before_evaluation", "node_evaluation")

    graph_builder.add_edge("node_evaluation", 'node_grading')
    # Now check the answer key and how much it scores
    graph_builder.add_edge("node_grading", 'node_save_eval_and_answer')
    graph_builder.add_edge("node_save_eval_and_answer", "node_check_answer_key")
    # graph_builder.add_edge("node_check_answer_key", "node_overall_makes_sense")
    graph_builder.add_conditional_edges("node_check_answer_key", route_after_key_check)
    graph_builder.add_edge("node_overall_makes_sense", "node_end")
    graph_builder.add_edge("node_end", END)
    print('compiling graph')

    graph = graph_builder.compile()

    for _, row in df_tasks.iterrows():# Use iterrows() to iterate over rows
        # Initialize the state for the current row
        print('Creating exam for ',row['task_description'])

        init_state: ExamState = {
            "occupation": row["occupation"],
            "task_id": str(row["task_id"]),  # Convert task_id to a string
            "task_description": row["task_description"],
            "exam_author_model": model,

            # Map your row fields to the typed dict fields
            "tools": safe_eval(row["required_tools_standard"]),
            "materials": safe_eval(row["required_materials_standard"]),

            "level": level,

            # Provide defaults or placeholders for the rest
            "exam": {},
            "system_prompt": "",
            "overview": "",
            "instructions": "",
            "materials_all": "",
            "materials_candidate": "",
            "submission": "",
            "evaluation": "",
            "grading": "",
            "answer_key": "",

            "errors": [],
            "check_real_materials": True,
            "check_no_internet": True,
            "failed_candidate_materials": 0,
            "key_grade_threshold": 100,
            "key_grade": 0.0,
            "answer_key_count": 0,
            "check_overall_makes_sense": True,
            "explanation_overall_makes_sense": "",
            "metadata": {}
            
        }
        try:
            # Invoke the compiled graph with the initial state
            result_state = graph.invoke(init_state)
            # Append the result_state to the list
            result_states.append(result_state)
        except Exception as e:
            # Handle any errors during graph invocation
            print(f"Error processing task_id {row['task_id']}: {e}")
            # Append an error state to the list for debugging
            result_states.append({
                "occupation": row["occupation"],
                "task_id": row["task_id"],
                "task_description": row["task_description"],
                "exam_author_model": model,
                "errors": [str(e)]
            })

        df_result_states = pd.DataFrame(result_states)
        folder_path = f"../data/exams/{model}/"
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        if level_string=='advanced':
            df_result_states.to_csv(folder_path+f"{level_string}_{occupation_group.replace(' ', '_').lower()}_exams.csv", index=False)
        else:
            df_result_states.to_csv(folder_path+f"{occupation_group.replace(' ', '_').lower()}_exams.csv", index=False)


