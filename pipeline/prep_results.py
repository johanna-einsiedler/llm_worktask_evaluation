import json
import os
from pathlib import Path

from inspect_ai.analysis.beta import evals_df, samples_df
from inspect_ai.log import read_eval_log, write_eval_log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# Safely parse JSON
def safe_parse_json(s):
    try:
        return json.loads(s)
    except:
        return {}


# Flatten with safety check
def flatten_usage(usage_dict):
    if not usage_dict:
        return {"model": None}
    model = next(iter(usage_dict))
    data = usage_dict[model]
    data["model"] = model
    return data


def aggregate_save_results(generating_model):
    occupations = [
        "Business and Financial Operations Occupations",
        "Computer and Mathematical Occupations",
        "Management Occupations",
    ]
    occupations_file_names = [occ.lower().replace(" ", "_") for occ in occupations]
    all_results = pd.DataFrame()
    for occ in occupations_file_names:
        results = samples_df(f"../logs/{generating_model}/{occ}")
        results["id"] = results["id"].astype(float)
        # make sure only valid exams are included
        exams = pd.read_csv(f"../data/exams/basic/{generating_model}/{occ}_exams.csv")
        results = results.merge(
            exams[["task_id", "exam", "key_grade", "task_description"]],
            how="left",
            left_on="id",
            right_on="task_id",
        )
        results = results[results["exam"] != "Exam not valid"]
        results = results[results["key_grade"] != ""]
        results = results[results["key_grade"].notna()]

        results["occupation_group"] = occ
        all_results = pd.concat([all_results, results])

    # all_results = all_results[all_results["overall_makes_sense"] == True]
    print("Number of exams: " + str(len(all_results["id"].unique())))
    all_results["model_usage_parsed"] = all_results["model_usage"].apply(safe_parse_json)

    # Apply and expand
    usage_expanded = all_results["model_usage_parsed"].apply(flatten_usage).apply(pd.Series)
    # Merge
    all_results = pd.concat([all_results, usage_expanded], axis=1)

    # Optional cleanup
    all_results = all_results.drop(columns=["model_usage", "model_usage_parsed"])
    all_results.drop_duplicates(subset=["model", "id"], inplace=True)

    models = all_results["model"].unique()
    tasks = all_results["id"].unique()
    print(all_results.columns)
    print(all_results)
    all_results = all_results[
        [
            "id",
            "input",
            "metadata_occupation",
            "score_runScoringScripts",
            "total_time",
            "input_tokens",
            "output_tokens",
            "total_tokens",
            "reasoning_tokens",
            "model",
            "occupation_group",
            "task_description",
        ]
    ].rename(
        columns={
            "id": "task_id",
            "input": "exam",
            "metadata_occupation": "occupation",
            "score_runScoringScripts": "score",
        }
    )

    all_results["generating_model"] = generating_model

    exam_path = f"../data/exams/basic/{generating_model}/"
    submissions = []
    for id in tasks:
        for model in models:
            if model is not None:
                if os.path.exists(
                    exam_path + str(id).replace(".", "_") + "/" + model + "/test_submission.json"
                ):
                    with open(
                        exam_path
                        + str(id).replace(".", "_")
                        + "/"
                        + model
                        + "/test_submission.json",
                        "r",
                    ) as f:
                        data = json.load(f)

                    if not data:
                        submissions.append([id, model, False])

                    else:
                        if data == "{}":
                            submissions.append([id, model, False])
                        else:
                            submissions.append([id, model, True])
                else:
                    submissions.append([id, model, False])

    submissions = pd.DataFrame(submissions)
    submissions.columns = ["task_id", "model", "submission_valid"]
    all_results = all_results.merge(submissions, how="left", on=["task_id", "model"])

    # drop any exams where any of the models scored more than 100
    task_ids_too_high = list(all_results.loc[all_results["score"] > 100, "task_id"])
    all_results = all_results[~all_results["task_id"].isin(task_ids_too_high)]

    print(str(len(np.unique(task_ids_too_high))), " exams excluded with too high scores")

    all_results.to_csv(f"../data/test_results/basic_{generating_model}.csv")

    print("saved results successfully")


if __name__ == "__main__":
    for generating_model in ["gemini-2.5-pro", "claude-3-7-sonnet-20250219", "o3-2025-04-16"]:
        print(generating_model)
        aggregate_save_results(generating_model)
