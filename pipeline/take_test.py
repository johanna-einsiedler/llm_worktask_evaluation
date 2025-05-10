import pandas as pd
import pandas as pd
import random
import os
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
import anthropic
import regex as re
import json
import subprocess
import shutil
import ast
import sys
# import google.generativeai as genai
from query_agents import query_agent, take_test
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

system_prompt_template = """You are an expert worker within the domain of {occupation}. Complete the following exam."""


#############
# support functions
###################

def save_answer_json(row, path, model):
    """
    Saves a test answer from a DataFrame row as a JSON file in a structured directory.

    Args:
        row (pd.Series): A row from a DataFrame containing the test answers.
        path (str): The base directory where the JSON file should be saved.
        model (str): The model name used to fetch the test answer from the row.

    Returns:
        bool: True if the JSON file was successfully saved, False otherwise.

    Process:
        1. Creates a folder structure based on the `task_id` from the row.
        2. Extracts the test answer JSON from the column `test_answers_<model>`.
           - If the answer is formatted within triple backticks (```json ... ```), it extracts the JSON content using regex.
           - If the direct value is a valid JSON string, it loads it directly.
           - If both attempts fail, it defaults to an empty JSON object (`{}`) and returns False.
        3. Saves the parsed JSON file in a subdirectory named after the model.
    """
    folder = os.path.join(path, str(row['task_id']).replace(".", "_"))
    try:
        text = re.search(r'```json(.*?)```', row['test_answers_'+model], re.DOTALL).group(1).strip()
        answer_file = json.loads(text)
    
    except:
        try:
            text = re.search(r'```json(.*?)```', row['test_answers_'+model], re.DOTALL).group(1).strip()

            # Step 1: Remove comment lines manually or using regex
            cleaned = "\n".join(line for line in text.splitlines() if not line.strip().startswith("//"))

            # Step 2: Remove outer quotes if present
            if cleaned.startswith("'") and cleaned.endswith("'"):
                cleaned = cleaned[1:-1]
            cleaned = re.sub(r'(?<=: )(-?\d{1,3}(?:,\d{3})+)', lambda m: m.group().replace(',', ''), cleaned)
            cleaned = ast.literal_eval(f'''"""{cleaned}"""''')  # Converts escaped \n etc. to real chars

            # Step 4: Parse to JSON
            answer_file = json.loads(cleaned)
        except:
            try:
                cleaned = "\n".join(line for line in row['test_answers_'+model].splitlines() if not line.strip().startswith("//"))
                if cleaned.startswith("'") and cleaned.endswith("'"):
                    cleaned = cleaned[1:-1]
                cleaned = re.sub(r'(?<=: )(-?\d{1,3}(?:,\d{3})+)', lambda m: m.group().replace(',', ''), cleaned)
                answer_file =  json.loads(cleaned)
            except:
                answer_file = '''{}'''
                json_path = os.path.join(path, str(row['task_id']),'/')
                os.makedirs(folder+'/'+model, exist_ok=True)

                with open(folder+'/'+model+"/test_submission.json", "w") as json_file:
                    json.dump(answer_file, json_file, ensure_ascii=False, indent=4)
                return False
    json_path = os.path.join(path, str(row['task_id']),'/')
    os.makedirs(folder+'/'+model, exist_ok=True)

    with open(folder+'/'+model+"/test_submission.json", "w") as json_file:
        json.dump(answer_file, json_file, ensure_ascii=False, indent=4)

    return True


def save_evaluation(row, path):
    """
    Extracts and saves Python evaluation code from a DataFrame row to a file.

    Args:
        row (pd.Series): A row from a DataFrame containing `task_id` and `answer_grading`.
        path (str): The base directory where the evaluation file should be saved.

    Returns:
        bool: True if the file was successfully saved, False otherwise.

    Process:
        1. Creates a directory named after the `task_id`, replacing problematic characters (e.g., "." and "/").
        2. Extracts Python code enclosed in triple backticks (` ```python ... ``` `) from the `answer_grading` column.
        3. If no Python code is found, prints a warning and returns False.
        4. Writes the extracted code to a file named `task_evaluation.py` inside the task directory.
    """
    folder = os.path.join(path, str(row['task_id']).replace(".", "_").replace("/", "_"))
    
    # Extract Python code from answer_grading
    try:
        match = re.search(r'```python(.*?)```', row['grading'], re.DOTALL)
        if not match:
            print(f"Warning: No Python code found in grading for task_id {row['task_id']}")
            return False
    except Exception as e:
        print(f"Error extracting Python code for task_id {row['task_id']}: {e}")
        return False
    
    eval_file = match.group(1).strip()

    # Create the directory
    os.makedirs(folder, exist_ok=True)
    file_path = os.path.join(folder, 'task_evaluation.py')

    print(f"Writing file: {file_path}")

    # Write to the Python file
    with open(file_path, "w", encoding="utf-8") as py_file:
        py_file.write(eval_file)

    print(f"File saved successfully: {file_path}")
    return True




def save_answer_key(row, path):
    """
    Extracts and saves the answer key from a DataFrame row to a JSON file.

    Args:
        row (pd.Series): A row from a DataFrame containing `task_id` and `answer_evaluation`.
        path (str): The base directory where the answer key should be saved.

    Returns:
        bool: True if the answer key was successfully saved, False otherwise.

    Process:
        1. Extracts the JSON-formatted answer key from the `answer_evaluation` column.
           - Attempts to extract JSON data enclosed in triple backticks (```json ... ```).
           - If extraction fails, attempts to load `answer_evaluation` as raw JSON.
           - If both methods fail, defaults to an empty dictionary (`{}`) and returns False.
        2. Creates a directory named after the `task_id`, replacing problematic characters.
        3. Saves the extracted JSON data as `answer_key.json` in the task directory.
    """
    print('saving answer key')
    folder = os.path.join(path, str(row['task_id']).replace(".", "_"))

    try:
        answer_file =  ast.literal_eval(row['answer_key'])
        print('got answer key')
        #answer_file = json.loads(re.search(r'```json(.*?)```', row['answer_key'], re.DOTALL).group(1).strip())
        #print(answer_file)
    except:
        try:
            answer_file =  json.loads(row['answer_key'])

            print('got answer key')

        except:
            answer_file = '''{}'''
            print('empty answer key')
            return False
    
    # Create the folder if it doesn't exist
    os.makedirs(folder, exist_ok=True)

    # Write to the Python file
    with open(folder+'/answer_key.json', "w", encoding="utf-8") as json_file:
        json.dump(answer_file, json_file, ensure_ascii=False, indent=4)
        print(f"File saved successfully: {folder}/answer_key.json")
    return True


def run_evaluation(row,path, model):
    """
    Executes the evaluation script for a given task and captures any errors.

    Args:
        row (pd.Series): A row from a DataFrame containing `task_id`, `answer_key_json`, and `evaluation_python`.
        path (str): The base directory where the evaluation script is located.
        model (str): The model name used to specify the subdirectory.

    Returns:
        list: A list of error messages encountered during execution, or [None] if the script runs successfully.
              If `answer_key_json` or `evaluation_python` is missing, returns a string message.

    Process:
        1. Constructs the folder path using `task_id`, replacing problematic characters.
        2. Checks if both `answer_key_json` and `evaluation_python` exist.
        3. Attempts to run `task_evaluation.py` in the appropriate directory.
        4. Captures standard error output if the script fails.
        5. Handles different types of errors:
           - `CalledProcessError`: If the script execution fails.
           - `FileNotFoundError`: If the script or directory is missing.
           - General exceptions for unexpected errors.
        6. Returns a list of error messages or confirmation of successful execution.

    Notes:
        - Uses `subprocess.run()` to execute the Python script.
        - `cwd` ensures execution in the correct directory.
        - Captures both `stderr` and `stdout` for debugging purposes.
    """

    errors =[]
    folder = os.path.join(path, str(row['task_id']).replace('.','_'))
    #path =  "../../data/exam_approach/test_results/" + folder + "/"
    print(f'{folder}/{model}/')
    # Passes answer_key isntead of test_submission to later check answer key gets full marks
    # subprocess.run(["ls", "-l", path])
    try:
        result = subprocess.run(
            ["python", "task_evaluation.py", 'test_submission.json', "answer_key.json"],
            cwd=f'{folder}/{model}/',
            check=True,  # Raise an exception if the command fails
            stderr=subprocess.PIPE,  # Capture stderr
            stdout=subprocess.PIPE   # Capture stdout (if needed)
            )
        print("Script executed successfully.")
        errors.append(None)
        return errors

    
    except subprocess.CalledProcessError as e:
        # Capture and store the error output in the errors list
        print(f"Error: Script failed with return code {e.returncode}")
        print(f"Error Output:\n{e.stderr.decode('utf-8')}")
        errors.append(e.stderr.decode('utf-8'))  # Append the error message to the errors list
        return errors
    except FileNotFoundError:
        error_message = "Error: The script or directory was not found. Check the path."
        print(error_message)
        print(path)
        errors.append(error_message)  # Append the error message to the errors list
        return errors
    except Exception as e:
        # Capture and store any unexpected error
        error_message = f"An unexpected error occurred: {str(e)}"
        print(error_message)
        errors.append(error_message)  # Append the error message to the errors list   
        return errors


  


def copy_answer_key(row,folder):
    """
    Copies the answer key JSON file to all subdirectories within the task-specific folder.

    Args:
        row (pd.Series): A row from a DataFrame containing `task_id`, which is used to determine the folder structure.
        folder (str): The parent directory where task-specific folders are stored.

    Process:
        1. Constructs the path to the task-specific folder using `task_id`, replacing problematic characters.
        2. Identifies the source file (`answer_key.json`) within this folder.
        3. Iterates over all subdirectories within the task folder.
        4. Copies `answer_key.json` into each subdirectory.

    Returns:
        None

    Notes:
        - Uses `os.walk()` to retrieve subdirectory names at the first level.
        - Assumes `answer_key.json` exists in the task-specific folder.
        - Prints confirmation for each copied file.

    Example:
        If `folder = "/data/tasks"` and `task_id = 123.4`, the function will:
        - Look for `/data/tasks/123_4/answer_key.json`
        - Copy it into all subdirectories inside `/data/tasks/123_4/`
    """

    parent_directory = folder+'/'+str(row['task_id']).replace(".", "_")
    source_file1 = parent_directory+'/answer_key.json'
    source_file2 = parent_directory+'/task_evaluation.py'

    if not os.path.isdir(parent_directory):
        print(f"Directory does not exist: {parent_directory}")
        return

    try:
        subdirs = next(os.walk(parent_directory))[1]
    except StopIteration:
        print(f"No subdirectories found in: {parent_directory}")
        return

    # Iterate over all subdirectories
    for subdir in next(os.walk(parent_directory))[1]:  # Get only subfolder names
        subdir_path = os.path.join(parent_directory, subdir)  # Full path to subfolder
        destination1 = os.path.join(subdir_path, os.path.basename(source_file1))  # Destination path
        destination2 = os.path.join(subdir_path, os.path.basename(source_file2))  # Destination path

        try:
            shutil.copy(source_file1, destination1)
            shutil.copy(source_file2, destination2)

            print(f"Copied {source_file1} to {destination1}")
            print(f"Copied {source_file2} to {destination2}")

        except FileNotFoundError:
            print(f"Source file not found: {source_file1}")
            print(f"Source file not found: {source_file2}")

            break


def collect_overall_scores(row,parent_directory):
    """
    Collects the 'overall_score' from test result JSON files across subdirectories within a task-specific folder.

    Args:
        row (pd.Series): A row from a DataFrame containing `task_id`, which is used to determine the folder structure.
        parent_directory (str): The base directory where task-specific folders are stored.

    Process:
        1. Constructs the path to the task-specific folder using `task_id`, replacing problematic characters.
        2. Iterates over all subdirectories within the task folder.
        3. Checks for the existence of `test_results.json` in each subdirectory.
        4. Reads and parses the JSON file, extracting the 'overall_score' if present.
        5. Stores scores in a dictionary where keys are subfolder names and values are corresponding scores.

    Returns:
        dict: A dictionary mapping each subfolder to its corresponding 'overall_score' value.

    Notes:
        - Handles JSON files containing either a dictionary or a list of dictionaries.
        - Prints warnings if files are missing, have unexpected formats, or lack the 'overall_score' field.
        - Catches JSON parsing errors and other unexpected exceptions.

    Example:
        If `parent_directory = "/data/tasks"` and `task_id = 123.4`, the function will:
        - Look in `/data/tasks/123_4/` for subdirectories.
        - Check for `test_results.json` in each subdirectory.
        - Extract and store the 'overall_score' values in a dictionary.

    """
    scores_dict = {}  # Dictionary to store scores with subfolder names as keys
    parent_directory = parent_directory+str(row['task_id']).replace(".", "_")
    # Iterate over each subfolder in the parent directory
    for subfolder in os.listdir(parent_directory):
        subfolder_path = os.path.join(parent_directory, subfolder)

        # Check if it's a directory
        if os.path.isdir(subfolder_path):
            submission_file = os.path.join(subfolder_path, "test_results.json")

            # Check if the submission.json file exists
            if os.path.isfile(submission_file):
                try:
                    # Read the JSON file
                    with open(submission_file, "r", encoding="utf-8") as file:
                        data = json.load(file)
                    
                    # Convert JSON to DataFrame (if it's a list of dicts)
                    if isinstance(data, list):
                        df = pd.DataFrame(data)
                    elif isinstance(data, dict):
                        df = pd.DataFrame([data])  # Convert single dict to DataFrame
                    else:
                        print(f"Unexpected format in {submission_file}")
                        continue
                    # Ensure 'overall_score' exists
                    if 'overall_score' in df.columns:
                        scores_dict[subfolder] = df['overall_score'].iloc[0]
                    else:
                        print(f"'overall_score' column missing in {submission_file}")

                except (json.JSONDecodeError, ValueError) as e:
                    print(f"Error reading {submission_file}: {e}")
                except Exception as e:
                    print(f"Unexpected error in {submission_file}: {e}")
    return scores_dict



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
    model = 'gemini-2.5-pro-preview-03-25'

    print('Taking the exams of', model)

    if level == 'basic':
        df = pd.read_csv(f"../data/exams/{model}/{occupation_group.replace(' ', '_').lower()}_exams.csv")

    if level == 'advanced':
        df = pd.read_csv(f"../data/exams/{model}/advanced_{occupation_group.replace(' ', '_').lower()}_exams.csv")


    test_takers = ['gemini-1.5-flash', 'gemini-2.0-flash']#, 'claude-3-7-sonnet-20250219', 'gpt-4o', 'gpt-3.5-turbo-0125', 'deepseek-chat', 'gemini-2.5-pro-preview-03-25', 'o3-2025-04-16', 'claude-3-5-sonnet-20240620', 'claude-3-sonnet-20240229']
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    model_folder_path = f'../data/test_results/{model}/'

    for idx, row in df.iterrows():
        if row['exam'] !='Exam not valid':
            print(row['task_id'])
            print('testing LLMs')
            for test_taker in test_takers:
                df.at[idx,'test_answers_'+test_taker] = take_test(row, system_prompt_template, row['exam'], test_taker)
                if level == 'basic':
                    df.to_csv(f"../data/test_results/{model}/{occupation_group.replace(' ', '_').lower()}_test_answers.csv",index=False)
                if level == 'advanced':
                    df.to_csv(f"../data/test_results/{model}/advanced_{occupation_group.replace(' ', '_').lower()}_test_answers.csv",index=False)

            df['answer_empty'] = df.apply(save_answer_json, axis=1, args=(model_folder_path, 'empty_submission'))


            print('running evaluations')
            df['evaluation_python'] = df.apply(save_evaluation, axis=1, args=(model_folder_path,))
            df['answer_key_json']=  df.apply(save_answer_key, axis=1, args=(model_folder_path,))
            df.apply(copy_answer_key, axis=1, args=(model_folder_path,))

            for test_taker in test_takers:
                df['answer_valid_'+test_taker] = df.apply(save_answer_json, axis=1, args=(model_folder_path, test_taker))
                df['errors_'+test_taker] = df.apply(run_evaluation, axis=1, args=(model_folder_path,test_taker,))


        df['errors_empty'] =df.apply(run_evaluation, axis=1, args=(model_folder_path,'empty_submission',))
        print('collecting scores')
        df['scores'] =  df.apply(collect_overall_scores,axis=1, args= (model_folder_path,))
        scores_df = pd.json_normalize(df['scores'])
        scores_df.columns = 'score_'+scores_df.columns
        # Combine the original DataFrame with the new columns
        df_expanded = pd.concat([df.drop('scores', axis=1), scores_df], axis=1)
        if level == 'basic':
            df_expanded.to_csv(f"../data/test_results/{model}/{occupation_group.replace(' ', '_').lower()}_test_results.csv", index=False)
        if level == 'advanced':
            df_expanded.to_csv(f"../data/test_results/{model}/advanced_{occupation_group.replace(' ', '_').lower()}_test_results.csv", index=False)





