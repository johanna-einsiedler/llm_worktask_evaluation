import json
import os
import re
import sys
from typing import List, Union

from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from query_agents import query_agent

# Load environment variables from a .env file if available
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

with open("../prompts/material_tools/system_prompt.txt", "r") as file:
    system_prompt_template = file.read()

with open("../prompts/material_tools/user_prompt.txt", "r") as file:
    user_prompt_template = file.read()


def get_requirements(
    df: pd.DataFrame, system_prompt_template: str, user_prompt_template: str, model: str
) -> pd.DataFrame:
    """
    Processes a DataFrame to generate OpenAI API responses based on formatted prompts.

    Parameters:
        df (pd.DataFrame): Input DataFrame containing columns 'title', 'task', and 'task_id'.
        system_prompt_template (str): Template for the system prompt.
        user_prompt_template (str): Template for the user prompt.
        model (str): The OpenAI model to use for generating responses.

    Returns:
        pd.DataFrame: A DataFrame containing the row index, formatted prompt, and parsed JSON output.
    """
    results = []

    for i, row in df.iterrows():
        try:
            # Extract relevant values from the row for prompt formatting
            occupation = row["title"]
            task_description = row["task"]
            task_id = row["task_id"]

            # Format system and user prompts using the provided templates
            system_prompt = system_prompt_template.format(
                occupation=occupation,
                task_description=task_description,
            )
            user_prompt = user_prompt_template.format(
                occupation=occupation, task_description=task_description, task_id=task_id
            )

            # Generate the response via the query agent
            output = query_agent(system_prompt, user_prompt, model)

            try:
                # Attempt to parse output as JSON directly
                parsed_output = json.loads(output) if output else {}
            except Exception:
                # Fallback: Extract JSON using regex if direct parsing fails
                parsed_output = json.loads(
                    re.search(r"```json\n(.*?)\n```", output, re.DOTALL).group(1)
                )
        except Exception as e:
            print(f"Error processing row {i}: {e}")
            parsed_output = {}  # Fallback to an empty dict on error

        # Construct the result dictionary and update with parsed output
        result = {"row": i, "prompt": user_prompt}
        result.update(parsed_output)
        results.append(result)

        print(f"Processed row {i}")

    # Convert the list of result dictionaries into a DataFrame
    df_results = pd.json_normalize(results)
    return df_results


def get_required(
    row: pd.Series, keyword: str, cols: List[str], other=True
) -> Union[List[str], str]:
    """
    Extracts required items from a row based on a given keyword.

    Parameters:
        row (pd.Series): A DataFrame row.
        keyword (str): The keyword to remove from column names.
        cols (List[str]): List of column names to check.

    Returns:
        Union[List[str], str]: List of required items if found; otherwise, an empty string.
    """
    required = [col.replace(keyword, "") for col in cols if row[col] == "Required"]
    if any("Other.classification" in item for item in required):
        required = [item for item in required if item != "Other.classification"]
        if other:
            required.append(row[keyword + "Other.name"])
    return required if required else ""


def get_requirement_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates lists of required tools and materials for each row.

    Parameters:
        df (pd.DataFrame): A DataFrame containing columns for tools, materials, and submission requirements.

    Returns:
        pd.DataFrame: The updated DataFrame with new columns 'required_tools' and 'required_materials'.
    """
    # Identify columns related to tools (starting with 'tool')
    tool_columns = [col for col in df.columns if col.startswith("tools")]
    df["required_tools"] = df.apply(get_required, axis=1, args=("tools.", tool_columns, True))
    df["required_tools_standard"] = df.apply(
        get_required, axis=1, args=("tools.", tool_columns, False)
    )

    # Identify columns related to materials (starting with 'material')
    material_columns = [col for col in df.columns if col.startswith("materials")]
    df["required_materials"] = df.apply(
        get_required, axis=1, args=("materials.", material_columns, True)
    )
    df["required_materials_standard"] = df.apply(
        get_required, axis=1, args=("materials.", material_columns, False)
    )

    return df


if __name__ == "__main__":
    if len(sys.argv) > 1:
        path_to_data = sys.argv[1]
    else:
        path_to_data = "../data/task_lists/management_occupations_CORE.csv"

    if len(sys.argv) > 2:
        models = sys.argv[2]
    else:
        model = "claude-3-7-sonnet-20250219"
    if len(sys.argv) > 3:
        overwrite = sys.argv[3]
    else:
        overwrite = True

    # Define the path to the CSV file containing tasks
    file_name = os.path.basename(path_to_data)
    print("Reading in", file_name)

    # Read the CSV file into a DataFrame and rename columns for consistency
    df = pd.read_csv(path_to_data)
    df = df.rename(columns={"Task ID": "task_id", "Task": "task", "Title": "title"})

    # Flag to determine whether to overwrite existing output
    print("Overwrite is set to", overwrite)

    # Process requirements for each specified model
    print("Generating requirements using", model)

    # Create the output directory for the current model if it does not exist
    output_dir = f"../data/required_materials_tools/{model}/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If not overwriting, attempt to load existing output and filter out already processed tasks
    if not overwrite:
        try:
            existing_df = pd.read_csv(
                os.path.join(output_dir, f"materials_tools_{file_name}"), index_col=0
            )
            if existing_df.shape[0] > 0:
                df_processing = df[~df["task_id"].isin(existing_df["task_id"])]
                print("Existing data shape:", existing_df.shape)
            else:
                df_processing = df
        except Exception as e:
            print("No existing file found or error reading file:", e)
            df_processing = df
    else:
        df_processing = df
    print("Number of tasks to be processed:", df_processing.shape[0])

    # Generate requirements for the filtered DataFrame
    out = get_requirements(df_processing, system_prompt_template, user_prompt_template, model)
    out = get_requirement_lists(out)
    if not overwrite and "existing_df" in locals():
        out = pd.concat([existing_df, out])
    # Save the resulting DataFrame to CSV
    out.to_csv(os.path.join(output_dir, f"materials_tools_{file_name}"))
