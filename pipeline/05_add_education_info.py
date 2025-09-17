import ast
import os

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from upsetplot import UpSet

if __name__ == "__main__":
    for occupation_group in [
        "business_and_financial_operations_occupations",
        "computer_and_mathematical_occupations",
        "management_occupations",
    ]:
        df = pd.read_csv(
            f"/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/filtered_tasks/{occupation_group}_CORE.csv",
            index_col=0,
        )
        print(df.shape)
        # remove unnecessary columns
        df = df.loc[:, ~df.columns.str.startswith("Unnamed")]
        # read in education information
        educ = pd.read_excel(
            "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/external/onet_data/education_onet.xlsx"
        )

        # read in mapping of education level numbers to descriptinos
        educ_mapping = pd.read_excel(
            "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/external/onet_data/educ_levels_explanation_mapping.xlsx"
        )
        # filter to include only education levels
        educ = educ[educ["Element Name"] == "Required Level of Education"]

        # read in onet for mapping
        onet = pd.read_excel(
            "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/external/onet_data/tasks_to_dwa.xlsx"
        )

        # merge data to onet
        df = df.merge(
            onet[["O*NET-SOC Code", "Task ID"]], how="left", left_on="task_id", right_on="Task ID"
        )

        # merge data to education information
        df_educ = (
            df[["O*NET-SOC Code", "occupation"]]
            .drop_duplicates()
            .merge(
                educ[["O*NET-SOC Code", "Category", "Data Value"]],
                how="left",
                left_on="O*NET-SOC Code",
                right_on="O*NET-SOC Code",
            )
        )

        # group by occupation and calcualte cumulative percentage of education levels
        df_educ["cum_perc"] = df_educ.groupby("occupation")["Data Value"].cumsum()
        min_idx = df_educ[df_educ["cum_perc"] > 50].groupby("occupation")["cum_perc"].idxmin()

        # merge to mapping
        educ_mapping = educ_mapping[educ_mapping["Element Name"] == "Required Level of Education"]
        df_educ = df_educ.loc[min_idx].merge(
            educ_mapping[["Category", "Category Description"]], how="left", on="Category"
        )
        df_educ = df_educ[["O*NET-SOC Code", "Category Description"]].rename(
            columns={"Category Description": "education"}
        )

        # merge back to actual dataframe
        df = df.merge(df_educ, how="left", on="O*NET-SOC Code")
        # save updated version
        df.to_csv(
            f"/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/filtered_tasks/{occupation_group}_CORE.csv",
            index=False,
        )
