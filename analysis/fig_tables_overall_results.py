import ast
import json
import os
import sys

from fig_tables_materials_tools import mark_invalid, read_in_data
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

model_dict = {
    "openai/gpt-3.5-turbo-0125": "GPT 3.5 Turbo",
    "openai/gpt-4o": "GPT 4o",
    "openai/o3-2025-04-16": "GPT o3",
    "anthropic/claude-3-haiku-20240307": "Claude 3 Haiku",
    "anthropic/claude-3-5-sonnet-20240620": "Claude 3.5 Sonnet",
    "anthropic/claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
    "anthropic/claude-sonnet-4-20250514": "Claude Sonnet 4",
    "google/gemini-1.5-flash": "Gemini 1.5 Flash",
    "google/gemini-2.0-flash": "Gemini 2.0 Flash",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
    "google/gemini-2.5-pro": "Gemini 2.5 Pro",
    "openai-api/deepseek/deepseek-reasoner": "Deep Seek R1",
    "openai-api/deepseek/deepseek-chat": "Deep Seek V3",
}


# Rename the occupation groups
occupation_group_mapping = {
    "business_and_financial_operations_occupations": "Business & Financial Operations",
    "computer_and_mathematical_occupations": "Computer & Mathematical",
    "management_occupations": "Management",
}


def plot_submission_failures(df, fig_path):
    exams = df[df["exam"] != "Exam not valid"]
    error_columns = [col for col in df.columns if col.startswith("errors_")]
    error_columns = [col for col in error_columns if col != "errors_empty"]
    models = [col.removeprefix("errors_") for col in error_columns]

    failure_counts = {}
    for model in models:
        not_valid_json = 0
        ex_error = 0
        zero_counts = 0
        for _, row in exams.iterrows():
            if not row["answer_valid_" + model]:
                not_valid_json = not_valid_json + 1
            elif not "[None]" == row["errors_" + model]:
                ex_error += 1
            elif row["score_" + model] == 0:
                zero_counts += 1
        failure_counts["score_" + model] = {
            "invalid_json": not_valid_json,
            "execution_error": ex_error,
            "zero_score": zero_counts,
        }

    answer_fails = pd.DataFrame(failure_counts).T.reset_index()
    answer_fails = answer_fails.melt(id_vars="index", var_name="error_type", value_name="count")

    # Example: preprocess data (assuming 'answer_fails' has 'index', 'error_type', 'count')
    answer_fails["index"] = answer_fails["index"].replace(model_dict)
    answer_fails["error_type"] = answer_fails["error_type"].replace(
        {
            "invalid_json": "Invalid JSON",
            "execution_error": "Execution Error",
            "zero_score": "Zero Score",
        }
    )

    # Pivot the data to wide format for stacking
    pivot_df = answer_fails.pivot_table(
        index="index", columns="error_type", values="count", aggfunc="sum"
    ).fillna(0)

    # Sort columns to keep consistent stacking order
    pivot_df = pivot_df[["Invalid JSON", "Execution Error", "Zero Score"]]  # adjust as needed

    # Set up colors
    barplot_colors = {
        "Invalid JSON": "#739E82",
        "Execution Error": "#E0777D",
        "Zero Score": "#8E3B46",
    }

    # Apply style
    theme_dict = {
        **sns.axes_style("white"),
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.spines.top": False,
        "grid.linestyle": ":",
        "legend.frameon": False,
        "legend.facecolor": "white",
        "legend.edgecolor": "white",
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "legend.framealpha": 0,
    }
    sns.set_theme(rc=theme_dict)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series([0] * len(pivot_df), index=pivot_df.index)

    for error_type in pivot_df.columns:
        ax.bar(
            pivot_df.index,
            pivot_df[error_type],
            label=error_type,
            bottom=bottom,
            alpha=0.8,
            color=barplot_colors[error_type],
        )
        bottom += pivot_df[error_type]

    # Aesthetics
    ax.set_title("LLM submission errors", fontsize=14)
    ax.set_xlabel("LLM")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90)

    # Move legend inside
    ax.legend(title="Error type", loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)

    # White background
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Successfully saved figure to: ", fig_path)


def plot_overall_scores(df, fig_path, fill_na=True):
    # exams = df[df["exam"] != "Exam not valid"]
    # score_cols = [col for col in df.columns if col.startswith("score")]
    # score_cols = [col for col in score_cols if col != "score_empty_submission"]

    # if fill_na:
    #     exam_scores = pd.melt(
    #         exams,
    #         id_vars=["task_id", "occupation_group"],
    #         value_vars=score_cols,
    #         var_name="Model",
    #         value_name="Value",
    #     )
    #     exam_scores["Value"] = exam_scores["Value"].fillna(0)
    # else:
    #     exams = exams.dropna(subset=score_cols)
    #     exam_scores = pd.melt(
    #         exams,
    #         id_vars=["task_id", "occupation_group"],
    #         value_vars=score_cols,
    #         var_name="Model",
    #         value_name="Value",
    #     )
    exam_scores = df
    # if no model managed to score a point remove exam
    ids = exam_scores.groupby("task_id").first().index

    ids = ids[exam_scores.groupby("task_id")["score"].max() == 0]
    exam_scores = exam_scores[~exam_scores["task_id"].isin(ids)]
    exam_scores["Model"] = exam_scores["model"].map(model_dict)
    exam_scores = exam_scores.sort_values("Model")
    exam_scores["occupation_group"] = exam_scores["occupation_group"].replace(
        occupation_group_mapping
    )

    plt.figure(figsize=(8, 4))
    sns.barplot(
        x="Model",
        y="score",
        errorbar=("sd"),
        hue="occupation_group",
        alpha=0.7,
        data=exam_scores,
        palette=["#FFBD59", "#8E3B46", "#38B6FF"],
    )
    plt.xticks(rotation=90)

    # Set plot labels and title
    plt.title(
        "Exam performance by LLM and occupation group, incl. standard deviation", fontsize=16
    )
    plt.xlabel("LLM")
    plt.ylabel("Exam score")
    plt.legend(
        title="Occupation group",
        frameon=True,
        loc="lower right",
        framealpha=0.6,
    )
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    print("Successfully saved figure to: ", fig_path)


def correlation_exam_performance(df, fig_path):
    # only include valid submission
    # df = df[df["submission_valid"] == True]

    scores = (
        df[["task_id", "generating_model", "score", "model", "occupation_group"]]
        .pivot(index=["task_id", "generating_model", "occupation_group"], columns="model")
        .reset_index()
    )
    scores.columns = [
        col[1] if col[0] == "score" and pd.notna(col[1]) else col[0] for col in scores.columns
    ]
    scores.rename(columns=model_dict, inplace=True)
    scores["occupation_group"] = scores["occupation_group"].replace(occupation_group_mapping)
    # print(exams_renamed)
    g = sns.PairGrid(
        scores,
        vars=[
            "GPT o3",
            "Claude 3.7 Sonnet",
            "Deep Seek R1",
            "Gemini 2.5 Pro",
        ],
        hue="occupation_group",
        palette=["#FFBD59", "#8E3B46", "#38B6FF"],
        corner=True,
        height=1.8,
    )

    g.map_diag(
        sns.histplot, alpha=0.7, multiple="stack", edgecolor="w"
    )  # ,multiple="stack"Set transparency with alpha=0.4 (0 is fully transparent, 1 is fully opaque)
    g.map_lower(sns.scatterplot, edgecolor="w")
    g.add_legend(
        title="Occupation Group",
        loc="upper right",
        bbox_to_anchor=(0.8, 0.95),
        ncol=1,
        frameon=False,
        fontsize=12,
        title_fontsize=14,
    )
    legend = g.legend
    legend.set_title(
        "Occupation Group", prop={"size": 14}
    )  # Use 'prop' to set font size for the title
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)  # x-axis label
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)  # y-axis label
            ax.tick_params(axis="both", labelsize=8)
    # Set explicit axis labels for both x and y axes

    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def overview_table(df, table_path):
    # print("min exam scores")
    # print(np.min(df.groupby("task_id")["score"].max()))

    df["model"] = df["model"].replace(model_dict)
    num_exams = df[["generating_model", "task_id"]].drop_duplicates().shape[0]
    # count submission errors
    submission_failure = pd.DataFrame(
        df.groupby("model")["submission_valid"].value_counts()
    ).reset_index()
    submission_failure = submission_failure[submission_failure["submission_valid"] == False]

    submission_0 = pd.DataFrame(
        df.groupby("model")["score"].apply(lambda x: (x == 0).sum())
    ).reset_index()

    failed_submissions = submission_failure[["model", "count"]].merge(
        submission_0[["model", "score"]],
        on="model",
        how="outer",
    )
    failed_submissions["score"] = failed_submissions["score"] - failed_submissions["count"]
    failed_submissions[["count", "score"]] = (
        failed_submissions[["count", "score"]] / num_exams * 100
    )
    failed_submissions = failed_submissions.rename(
        columns={
            "count": "Percentage of Failed Submissions",
            "score": "Percentage of Valid Submissions Scoring 0 points",
        }
    )
    # get minimum scores
    mins = pd.DataFrame(df.groupby("model")["score"].min()).rename(columns={"score": "Min. Score"})

    # get maximum scores
    maxs = pd.DataFrame(df.groupby("model")["score"].max()).rename(columns={"score": "Max. Score"})

    # get quantiles
    quantiles = df.groupby("model")["score"].quantile([0.25, 0.5, 0.75]).unstack()
    quantiles = quantiles.rename(
        columns={0.25: "1st Quartile", 0.5: "Median", 0.75: "3rd Quartile"}
    ).round(2)

    overview = (
        failed_submissions.merge(mins, on="model")
        .merge(quantiles, on="model")
        .merge(maxs, on="model")
    ).round(2)
    latex_code = overview.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=False,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully saved table to ", table_path)


def overall_submission_errors(
    all_exams, table_path="../results/tables/overall_submission_errors.tex"
):
    num_exams = all_exams[["generating_model", "task_id"]].drop_duplicates().shape[0]
    # print(num_exams)
    # print(all_exams["model"].unique())
    # print(all_exams[all_exams["model"] == "openai/gpt-3.5-turbo-0125"].shape)
    # print("all exams shape", all_exams.shape)

    submission_failure = pd.DataFrame(
        all_exams.groupby("model")["submission_valid"].value_counts()
    ).reset_index()
    submission_failure = submission_failure[submission_failure["submission_valid"] == False]
    print(submission_failure)
    # 0 submissions
    submission_0 = pd.DataFrame(
        all_exams.groupby("model")["score"].apply(lambda x: (x == 0).sum())
    ).reset_index()

    failed_submissions = submission_failure[["model", "count"]].merge(
        submission_0[["model", "score"]],
        on="model",
        how="outer",
    )

    failed_submissions["score"] = failed_submissions["score"] - failed_submissions["count"]
    failed_submissions[["count", "score"]] = (
        failed_submissions[["count", "score"]] / num_exams * 100
    )
    failed_submissions = failed_submissions.rename(
        columns={
            "count": "Percentage of Failed Submissions",
            "score": "Percentage of Valid Submissions Scoring 0 points",
        }
    )
    failed_submissions["model"] = failed_submissions["model"].replace(model_dict)
    failed_submissions = np.round(failed_submissions, 1)
    latex_code = failed_submissions.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=True,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully saved table to ", table_path)


def self_bias_comparison(df, table_path="../results/tables/self_bias.tex"):
    models = ["o3-2025-04-16", "claude-3-7-sonnet-20250219", "gemini-2.5-pro"]
    all_exams_list = []
    for model in models:
        exam_list = pd.read_csv(f"../data/test_results/basic_{model}.csv")
        exam_list["generating_model"] = model
        all_exams_list.append(exam_list)
    df = pd.concat(all_exams_list, ignore_index=True)
    ids_all = (
        df.groupby("task_id")["generating_model"]
        .apply(set)
        .loc[lambda s: s.apply(lambda model: set(models).issubset(model))]
        .index
    )
    print(len(ids_all), " exams with all models")
    df_3 = df[
        (df["task_id"].isin(ids_all))
        * (
            df["model"].isin(
                [
                    "anthropic/claude-3-7-sonnet-20250219",
                    "google/gemini-2.5-pro",
                    "openai/o3-2025-04-16",
                ]
            )
        )
    ]
    mean_scores = df_3.groupby(['task_id','generating_model'])['score'].mean().reset_index()
    
    df_3["model"] = df_3["model"].map(model_dict)
    df_3["generating_model"] = df_3["generating_model"].map(
        {
            "o3-2025-04-16": "GPT o3",
            "claude-3-7-sonnet-20250219": "Claude 3.7 Sonnet",
            "gemini-2.5-pro": "Gemini 2.5 Pro",
        }
    )
    self_bias = df_3.groupby(["generating_model", "model"])["score"].mean().unstack()
    self_bias = np.round(self_bias, 2)
    latex_code = self_bias.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=True,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully save self bias table")


def all_exams(models):
    all_exams_list = []
    for model in models:
        exam_list = pd.read_csv(f"../data/test_results/basic_{model}.csv")
        exam_list["generating_model"] = model
        all_exams_list.append(exam_list)
    all_exams = pd.concat(all_exams_list, ignore_index=True)

    # remove those where all models scored 0
    max_scores = all_exams.groupby(["task_id", "generating_model"])["score"].max().reset_index()
    max_scores = max_scores[max_scores["score"] == 0][
        ["task_id", "generating_model"]
    ].drop_duplicates()
    print("max scores shape", max_scores.shape)

    max_scores = max_scores.itertuples(index=False, name=None)

    all_exams = all_exams[
        ~all_exams[["score", "generating_model"]].apply(tuple, axis=1).isin(max_scores)
    ]
    return all_exams


def single_exam_gen(model):
    all_exams = pd.read_csv(f"../data/test_results/basic_{model}.csv")
    max_scores = all_exams.groupby(["task_id", "generating_model"])["score"].max().reset_index()
    max_scores = max_scores[max_scores["score"] == 0][
        ["task_id", "generating_model"]
    ].drop_duplicates()
    print("max scores shape", max_scores.shape)

    max_scores = max_scores.itertuples(index=False, name=None)

    all_exams = all_exams[
        ~all_exams[["score", "generating_model"]].apply(tuple, axis=1).isin(max_scores)
    ]
    return all_exams


def mean_performance_comparison(fig_path="../results/figures/mean_performance_comparison.pdf"):
    gemini = pd.read_csv(
        "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/test_results/basic_gemini-2.5-pro.csv"
    )
    claude = pd.read_csv(
        "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/test_results/basic_claude-3-7-sonnet-20250219.csv"
    )
    gpt = pd.read_csv(
        "/Users/einsie0004/Documents/research/21_automatisation/llm_worktask_evaluation/data/test_results/basic_o3-2025-04-16.csv"
    )

    claude_37_values = (
        (claude.groupby("model")["score"].mean() - claude["score"].mean()) / claude["score"].mean()
    ) * 100
    gemini_25_values = (
        (gemini.groupby("model")["score"].mean() - gemini["score"].mean()) / gemini["score"].mean()
    ) * 100
    gpt_o3_values = (
        (gpt.groupby("model")["score"].mean() - gpt["score"].mean()) / gpt["score"].mean()
    ) * 100

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 6))

    # Colors for bars (red for below average, blue for above average)
    def get_colors(values):
        return ["#E8A4A4" if v < 0 else "#87CEEB" for v in values]

    colors_claude = get_colors(claude_37_values)
    colors_gemini = get_colors(gemini_25_values)
    colors_gpto3 = get_colors(gpt_o3_values)

    models = claude_37_values.index.map(model_dict).values
    # Left subplot - Claude 3.7 Sonnet
    bars1 = ax1.barh(models, claude_37_values, color=colors_claude)
    ax1.set_xlabel("Relative deviation from mean", fontsize=14)
    ax1.set_ylabel("Exam taker model", fontsize=14)
    ax1.set_title("Claude 3.7 Sonnet", fontsize=14)
    ax1.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax1.set_xlim(-100, 100)
    ax1.grid(True, axis="y", alpha=0.3)

    # Right subplot - Gemini 2.5 Pro
    bars2 = ax2.barh(models, gemini_25_values, color=colors_gemini)
    ax2.set_xlabel("Relative deviation from mean", fontsize=14)
    ax2.set_title("Gemini 2.5 Pro", fontsize=14)
    ax2.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax2.set_xlim(-100, 100)
    ax2.grid(True, axis="y", alpha=0.3)

    # Third subplot - GPT-4o
    bars3 = ax3.barh(models, gpt_o3_values, color=colors_gpto3)
    ax3.set_xlabel("Relative deviation from mean", fontsize=14)
    ax3.set_title("GPT-o3", fontsize=14)
    ax3.axvline(x=0, color="black", linestyle="-", linewidth=0.5)
    ax3.set_xlim(-100, 100)
    ax3.grid(True, axis="y", alpha=0.3)

    # Remove y-axis labels from middle and right subplots to avoid duplication
    ax2.set_yticklabels([])
    ax3.set_yticklabels([])

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor="#E8A4A4", label="Below average"),
        plt.Rectangle((0, 0), 1, 1, facecolor="#87CEEB", label="Above average"),
    ]

    # Main title (below legend)
    fig.suptitle(
        "Comparison of AI agents' performance by exam generation model",
        fontsize=16,
        y=1,
        fontweight="bold",
    )

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=2,
        fontsize=12,
    )
    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)

    # Display the plot
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")


def most_improved(df, table_path="../results/tables/most_improved.tex"):
    model_dict_time = {
        "openai/gpt-3.5-turbo-0125": np.nan,
        "openai/gpt-4o": "early",
        "openai/o3-2025-04-16": "late",
        "anthropic/claude-3-haiku-20240307": "early",
        "anthropic/claude-3-5-sonnet-20240620": "early",
        "anthropic/claude-3-7-sonnet-20250219": "late",
        "anthropic/claude-sonnet-4-20250514": "late",
        "google/gemini-1.5-flash": "early",
        "google/gemini-2.0-flash": "late",
        "google/gemini-2.5-flash": "late",
        "google/gemini-2.5-pro": "late",
        "openai-api/deepseek/deepseek-reasoner": "late",
        "openai-api/deepseek/deepseek-chat": "late",
    }

    df["timing"] = df["model"].map(model_dict_time)
    task_timing = df.groupby(["timing", "task_id"])["score"].mean().reset_index()
    task_timing = task_timing[task_timing["timing"] == "early"].merge(
        task_timing[task_timing["timing"] == "late"], on="task_id", suffixes=["_early", "_late"]
    )
    task_timing["score_diff"] = task_timing["score_late"] - task_timing["score_early"]
    most_improved = (
        task_timing.merge(
            df[
                ["task_id", "task_description", "exam", "occupation_group", "occupation"]
            ].drop_duplicates(),
            on="task_id",
        )
        .sort_values("score_diff", ascending=False)
        .head(11)
    )
    most_improved = most_improved[
        [
            "task_description",
            "occupation",
            "occupation_group",
            "score_early",
            "score_late",
            "score_diff",
        ]
    ].round(2)
    most_improved["occupation_group"] = most_improved["occupation_group"].replace(
        occupation_group_mapping
    )

    latex_code = most_improved.to_latex(
        escape=False,  # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,  # Include the header row (column labels)
        index=True,
        float_format="%.2f",  # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print("Successfully save self bias table")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        # model = "claude-3-7-sonnet-20250219"
        # model = "gemini-2.5-pro"  # IGNORE
        model = "o3-2025-04-16"  # IGNORE
    occupations = [
        "Business and Financial Operations Occupations",
        "Computer and Mathematical Occupations",
        "Management Occupations",
    ]
    models = ["o3-2025-04-16", "claude-3-7-sonnet-20250219", "gemini-2.5-pro"]
    all_models_exams = all_exams(models)
    # occupations_file_names = [occ.lower().replace(" ", "_") for occ in occupations]
    # exam_list = read_in_data(occupations_file_names, f"../data/test_results/{model}/test_results_")
    # exam_list = mark_invalid(exam_list)
    # plot_submission_failures(
    #     exam_list, f"../results/figures/{model}/{model}_submission_errors.pdf"
    # )
    # correlation_exam_performance(
    #     exam_list, f"../results/figures/{model}/{model}_correlation_exam_performance.pdf"
    # )

    # overall_submission_errors(all_models_exams)
    # self_bias_comparison(all_models_exams, table_path="../results/tables/self_bias.tex")

    # overview_table(all_models_exams, "../results/tables/all_models_overall_submission_scores.tex")
    correlation_exam_performance(
        all_models_exams, "../results/figures/all_models_correlation_exam_performance.pdf"
    )
    # mean_performance_comparison()
    for model in ["o3-2025-04-16", "claude-3-7-sonnet-20250219", "gemini-2.5-pro"]:
        exam_list = single_exam_gen(model)
        most_improved(exam_list, f"../results/tables/{model}/{model}_most_improved.tex")
        plot_overall_scores(exam_list, f"../results/figures/{model}/{model}_overall_scores.pdf")
        overview_table(
            exam_list, f"../results/tables/{model}/{model}_overall_submission_scores.tex"
        )
