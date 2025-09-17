from collections import Counter
import os
import sys

import pandas as pd

# Main script entry point
if __name__ == "__main__":
    # Determine the path to the data directory from command-line arguments (if provided), or set a default path

    if len(sys.argv) > 1:
        occupation_group = sys.argv[1]
    else:
        # occupation_group = "Computer and Mathematical Occupations"
        # occupation_group = "Management Occupations"
        occupation_group = "Business and Financial Operations Occupations"

    if len(sys.argv) > 2:
        model = sys.argv[2]
    else:
        model = "claude-3-7-sonnet-20250219"
    if len(sys.argv) > 3:
        core_label = sys.argv[3]
    else:
        core_label = "CORE"
    # Check whether we should overwrite the existing files (default is True)
    if len(sys.argv) > 4:
        overwrite = sys.argv[4]
    else:
        overwrite = True

    if len(sys.argv) > 5:
        exclusion_tools = sys.argv[5]
    else:
        # List of tools and materials to exclude
        exclusion_tools = [
            # "Spreadsheet software",
            "Image viewer",
            "Image generator",
            "Presentation software",
            "Online search engine",
            "Video conferencing",
            "Chat interfaces",
            "Internet access",
            "Specialized software",
            "Email client",
        ]

    if len(sys.argv) > 6:
        exclusion_materials = sys.argv[6]
    else:
        exclusion_materials = [
            "Images",
            "Audio files",
            "Video files",
            "Virtual labs",
            "Specialized files",
        ]

    exclusion_tools = [
        "tools." + item for item in exclusion_tools
    ]  # Modify tool names to match the column names
    exclusion_materials = [
        "materials." + item for item in exclusion_materials
    ]  # Modify material names to match the column names

    path_to_data = f"../data/required_materials_tools/{model}/materials_tools_{occupation_group.replace(' ', '_').lower()}_{core_label}.csv"

    excluded_tasks = []  # List to store tasks that need to be excluded

    df = pd.read_csv(path_to_data)
    print(df.columns)
    print("Overall ", df.shape[0], " tasks in the data")  # Print the number of tasks in the file
    df = df.dropna(subset=["task_id"])
    print(
        "after removing empty task_ids", df.shape[0], " tasks in the data"
    )  # Print the number of tasks in the
    print("Exclusion tools: ", exclusion_tools)
    # Filter out tasks based on the exclusion tools
    excluded_tools = list(df.loc[(df[exclusion_tools] == "Required").sum(axis=1) >= 1, "task_id"])
    print(len(excluded_tools), " tasks excluded based on tools")

    print("Exclusion materials: ", exclusion_materials)
    # Further filter out tasks based on the exclusion materials
    excluded_mat = list(
        df.loc[(df[exclusion_materials] == "Required").sum(axis=1) >= 1, "task_id"]
    )
    print(len(excluded_mat), " tasks excluded based on materials")

    exlcuded_remote = list(df.loc[df["can_be_performed_remotely"] == False, "task_id"])
    print(len(exlcuded_remote), " tasks excluded based on remote infeasibility")

    exlcuded_practical = list(df.loc[df["feasibility_practical"] == False, "task_id"])
    print(len(exlcuded_practical), " tasks excluded based on practical infeasibility")

    excluded_ids = excluded_tools + excluded_mat + exlcuded_remote + exlcuded_practical
    excluded_ids = list(set(excluded_ids))  # Remove duplicates
    print(len(excluded_ids), " tasks excluded in total")

    df = df[~df["task_id"].isin(excluded_ids)]  # Filter out excluded tasks
    print("Remaining tasks: ", df.shape[0])
    # Save the filtered DataFrame to a CSV file
    save_path = (
        f"../data/filtered_tasks/{occupation_group.replace(' ', '_').lower()}_{core_label}.csv"
    )
    pd.DataFrame(df).to_csv(save_path)
