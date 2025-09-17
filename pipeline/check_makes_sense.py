import json
import re

import pandas as pd
from query_agents import query_agent


def node_overall_makes_sense(row, model):
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
        system_prompt = file.read()
    with open("../prompts/sanity_check_prompts/prompt_makes_sense_user.txt", "r") as file:
        user_message = file.read()
    state = pd.DataFrame()
    user_message = user_message.format(
        overview=row["overview"],
        instructions=row["instructions"],
        materials_all=row["materials_all"],
        materials_candidate=row["materials_candidate"],
        submission=row["submission"],
        evaluation=row["evaluation"],
        grading=row["grading"],
        answer_key=row["answer_key"],
    )
    content, metadata = query_agent(system_prompt, user_message, model)
    print(content)

    try:
        response = content.strip()
        result = json.loads(response)
        check_overall_makes_sense = bool(result.get("makes_sense", False))
        explanation_overall_makes_sense = str(result.get("explanation", ""))
    except:
        try:
            response = content.strip()
            text = re.search(r"```json(.*?)```", response, re.DOTALL).group(1).strip()
            result = json.loads(text)
            check_overall_makes_sense = bool(result.get("makes_sense", False))
            explanation_overall_makes_sense = str(result.get("explanation", ""))
        except:  # json.JSONDecodeError:
            # If the LLM's response isn't valid JSON, mark sense-check as False
            # and store the raw response for debugging.
            check_overall_makes_sense = False
            explanation_overall_makes_sense = "Could not parse JSON. Raw LLM response:\n" + str(
                content
            )
    row["check_overall_makes_sense"] = check_overall_makes_sense
    row["explanation_overall_makes_sense"] = explanation_overall_makes_sense

    return row


if __name__ == "__main__":
    generating_model = "o3-2025-04-16"
    folder_path = f"/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/exams/basic/{generating_model}/"
    occ = "management_occupations"
    # occ = "computer_and_mathematical_occupations"
    # occ = "business_and_financial_operations_occupations"

    df = pd.read_csv(folder_path + occ + "_exams.csv")
    df = df.apply(lambda x: node_overall_makes_sense(x, model=generating_model), axis=1)
    df.loc[df["check_overall_makes_sense"] == False, "exam"] = "Exam not valid"
    df.to_csv(folder_path + occ + "_exams_sanity_checked.csv", index=False)
