from datetime import datetime
import json
import os
import re
import sys
from typing import List, Union

from dotenv import find_dotenv, load_dotenv
import numpy as np
import pandas as pd
from query_agents import query_agent
from required_materials_tools import get_requirement_lists, get_requirements

# Load environment variables from a .env file if available
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
with open("../prompts/materials_tools/system_prompt.txt", "r") as file:
    system_prompt_template = file.read()

with open("../prompts/materials_tools/user_prompt_05_23.txt", "r") as file:
    user_prompt_template = file.read()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        occupation = sys.argv[1]
    else:
        occupation = "management_occupations"
    # occupation = "business_and_financial_operations_occupations"
    # occupation = "computer_an_mathematical_occupations"

    if len(sys.argv) > 2:
        models = sys.argv[2]
    else:
        model = "claude-3-7-sonnet-20250219"
    if len(sys.argv) > 3:
        overwrite = sys.argv[3]
    else:
        overwrite = True

    if len(sys.argv) > 4:
        experiment = sys.argv[4]
    else:
        experiment = False

    path_to_data = f"../data/task_lists/{occupation}_CORE.csv"

    # Define the path to the CSV file containing tasks
    file_name = os.path.basename(path_to_data)
    print("Reading in", file_name)

    # Read the CSV file into a DataFrame and rename columns for consistency
    df = pd.read_csv(path_to_data)
    df = df.rename(columns={"Task ID": "task_id", "Task": "task", "Title": "title"})
    output_dir = f"../data/required_materials_tools/{model}/"

    # Check which are still missing
    materials_tools = pd.read_csv(
        f"../data/required_materials_tools/claude-3-7-sonnet-20250219/materials_tools_{occupation}_CORE.csv"
    )
    materials_tools["missing"] = materials_tools["task_id"].isna()

    missing = pd.concat([df, materials_tools[["missing"]]], axis=1)

    missing = missing[missing["missing"] == True]
    print("missing tasks: ", missing.shape[0])

    print(missing["task_id"])
    # Generate requirements for the filtered DataFrame
    out_missing = get_requirements(missing, system_prompt_template, user_prompt_template, model)
    out_missing = get_requirement_lists(out_missing)
    out = pd.concat(
        [materials_tools[~materials_tools["task_id"].isna()], out_missing], ignore_index=True
    )
    print("still missing tasks: ", out[out["task_id"].isna()].shape[0])
    # Save the resulting DataFrame to CSV
    out.to_csv(os.path.join(output_dir, f"materials_tools_{file_name}"))
