import os
import pandas as pd
import numpy as np
import json
import re
from typing import Union, List
from dotenv import load_dotenv, find_dotenv
from query_agents import query_agent 
import sys

# Load environment variables from a .env file if available
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

# System prompt template for the query agent
system_prompt_template = '''
You are an excellent examiner of {occupation} capabilities. 
The overall objective is to evaluate, via a **practical** online exam without a time limit, whether {occupation} can {task_description}'''

# User prompt template with detailed instructions and expected JSON output structure
user_prompt_template = '''
Task ID: {task_id}

Your assignment is to determine if it is possible to design a meaningful, **practical** exam for this task that can be performed remotely and to identify the required tools and materials for the evaluation. 

**Definitions:**
- **Tools:** Software or applications (e.g., coding languages, code editor, spreadsheets, text editor, presentation software, image generator), that the candidate needs to use to complete the test.
- **Materials:** Digital content (e.g., data files, PDFs, images, audio files, virtual lab environments) that form part of the test content.
- **Practical exam:** A practical exam is a an exam actually testing whether the described task can be performed successfully. An exam testing the knowledge about the task is NOT a practical exam.

**Instructions:**
1. **Remote Feasibility:**  
   Evaluate whether the task can be performed online/remotely or if it requires in-person presence.
   - **If in-person required:** Output `"can_be_performed_remotely": false` and set all other fields (tools and materials) to `"NA"`.
   - **If remote:** Output `"can_be_performed_remotely": true` and continue with the evaluation.

2. **Tools Required:**  
   For each tool, assess whether it is needed for the task ({task_description}) for the role of {occupation}. Options are: "Required" or "Not Required". The tools include:
   - "Coding"
   - "Spreadsheets"
   - "Text editor"
   - "PDF viewer"
   - "Presentation software"
   - "Image Generator"
   - "Online search engine"
   - "Other" (specify tool name and classification if needed; otherwise "NA")

3. **Materials Required:**  
   For each material, determine if it is necessary as part of the test to evaluate {occupation}'s ability to perform the task ({task_description}). Options are: "Required"or "Not required" The materials include:
   - "Text"
   - "Data"
   - "Images"
   - "Audio files"
   - "Video files"
   - "Virtual labs or sandbox environments"
   - "Other" (specify material name and classification if needed; otherwise "NA")

4. **Feasability of a practical exam:**
    Evaluate whether the task can meaningfuly be tested in a practical, remote exam.
    - If you think this is possible, answer True,
    - Otherwise answer False

5. **Chain-of-Thought Reasoning:**  
   Optionally, include a brief chain-of-thought explanation (no more than 150 words) for your evaluations in a field called `"chain_of_thought"`.

**Output Requirement:**  
Return a JSON object strictly adhering to the provided structure, without any extra commentary outside of the JSON fields.

**Expected JSON Structure:**
{{
  "task_id": "{task_id}",
  "occupation": "{occupation}",
  "task_description": "{task_description}",
  "can_be_performed_remotely": true/false,
  "tools": {{
    "Coding": "Not Required/Required/NA",
    "Spreadsheets": "Not Required/Required/NA",
    "Text editor": "Not Required/Required/NA",
    "PDF viewer": "Not Required/Required/NA", 
    "Presentation software": "Not Required/Required/NA",
    "Online search engine": "Not Required/Required/NA",
    "Image Generator": "Not Required/Required/NA",
    "Other": {{
      "name": "Tool Name/NA",
      "classification": "Not Required/Required/NA"
    }}
  }},
  "materials": {{
    "Text": "Not Required/Required/NA",
    "Data": "Not Required/Required/NA",
    "Images": "Not Required/Required/NA",
    "Audio files": "Not Required/Required/NA",
    "Video files": "Not Required/Required/NA",
    "Virtual labs or sandbox environments": "Not Required/Required/NA",
    "Other": {{
      "name": "Material Name/NA",
      "classification": "Not Required/Required/NA"
    }}
  }},
  "feasiblity_practical": true/false
  "chain_of_thought": "Brief explanation (no more than 150 words)."
}}
'''

def get_requirements(df: pd.DataFrame, system_prompt_template: str, user_prompt_template: str, model: str) -> pd.DataFrame:
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
            occupation = row['title']
            task_description = row['task']
            task_id = row['task_id']

            # Format system and user prompts using the provided templates
            system_prompt = system_prompt_template.format(
                occupation=occupation,
                task_description=task_description,
            )
            user_prompt = user_prompt_template.format(
                occupation=occupation,
                task_description=task_description,
                task_id=task_id
            )

            # Generate the response via the query agent
            output = query_agent(system_prompt, user_prompt, model)

            try:
                # Attempt to parse output as JSON directly
                parsed_output = json.loads(output) if output else {}
            except Exception:
                # Fallback: Extract JSON using regex if direct parsing fails
                parsed_output = json.loads(re.search(r'```json\n(.*?)\n```', output, re.DOTALL).group(1))
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

def get_required(row: pd.Series, keyword: str, cols: List[str], other=True) -> Union[List[str], str]:
    """
    Extracts required items from a row based on a given keyword.

    Parameters:
        row (pd.Series): A DataFrame row.
        keyword (str): The keyword to remove from column names.
        cols (List[str]): List of column names to check.

    Returns:
        Union[List[str], str]: List of required items if found; otherwise, an empty string.
    """
    required = [col.replace(keyword, '') for col in cols if row[col] == "Required"]
    if any('Other.classification' in item for item in required ):
        required = [item for item in required if item != 'Other.classification']
        if other:
            required.append(row[keyword+'Other.name'])
    return required if required else ''

def get_requirement_lists(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates lists of required tools and materials for each row.

    Parameters:
        df (pd.DataFrame): A DataFrame containing columns for tools, materials, and submission requirements.

    Returns:
        pd.DataFrame: The updated DataFrame with new columns 'required_tools' and 'required_materials'.
    """
    # Identify columns related to tools (starting with 'tool')
    tool_columns = [col for col in df.columns if col.startswith('tools')]
    df["required_tools"]  = df.apply(get_required, axis=1, args=('tools.', tool_columns, True))
    df["required_tools_standard"]  = df.apply(get_required, axis=1, args=('tools.', tool_columns, False))

    # Identify columns related to materials (starting with 'material')
    material_columns = [col for col in df.columns if col.startswith('materials')]
    df["required_materials"] = df.apply(get_required, axis=1, args=('materials.', material_columns,True))
    df["required_materials_standard"] = df.apply(get_required, axis=1, args=('materials.', material_columns,False))

    return df

if __name__ == "__main__":

    if len(sys.argv) >1:
        path_to_data = sys.argv[1]
    else:
        path_to_data = '../data/task_lists/management_occupations_CORE.csv'

    if len(sys.argv)>2:
        models = sys.argv[2]
    else:
        model = "claude-3-7-sonnet-20250219"
    if len(sys.argv)>3:
        overwrite = sys.argv[3]
    else:
        overwrite = True

    # Define the path to the CSV file containing tasks
    file_name = os.path.basename(path_to_data)
    print('Reading in', file_name)

    # Read the CSV file into a DataFrame and rename columns for consistency
    df = pd.read_csv(path_to_data)
    df = df.rename(columns={'Task ID': 'task_id', 'Task': 'task', 'Title': 'title'})

    # Flag to determine whether to overwrite existing output
    print('Overwrite is set to', overwrite)

    # Process requirements for each specified model
    print('Generating requirements using', model)

    # Create the output directory for the current model if it does not exist
    output_dir = f'../data/required_materials_tools/{model}/'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # If not overwriting, attempt to load existing output and filter out already processed tasks
    if not overwrite:
        try:
            existing_df = pd.read_csv(os.path.join(output_dir, f'materials_tools_{file_name}'), index_col=0)
            if existing_df.shape[0] > 0:
                df_processing = df[~df['task_id'].isin(existing_df['task_id'])]
                print('Existing data shape:', existing_df.shape)
            else:
                df_processing = df
        except Exception as e:
            print("No existing file found or error reading file:", e)
            df_processing = df
    else:
        df_processing = df
    print('Number of tasks to be processed:', df_processing.shape[0])

    # Generate requirements for the filtered DataFrame
    out = get_requirements(df_processing, system_prompt_template, user_prompt_template, model)
    out = get_requirement_lists(out)
    if not overwrite and 'existing_df' in locals():
        out = pd.concat([existing_df, out])
    # Save the resulting DataFrame to CSV
    out.to_csv(os.path.join(output_dir, f'materials_tools_{file_name}'))
