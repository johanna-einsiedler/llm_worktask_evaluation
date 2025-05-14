import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np
import json
import ast
import sys
from fig_tables_materials_tools import read_in_data, mark_invalid


model_dict = {
    'score_gpt-3.5-turbo-0125': 'GPT 3.5 Turbo',
    'score_gpt-4o': 'GPT 4o',
    'score_claude-3-7-sonnet-20250219': 'Claude 3.7 Sonnet',
    'score_deepseek-chat': 'DeepSeek Chat',
    'score_gemini-1.5-flash': 'Gemini 1.5 Flash',
    'score_gemini-2.0-flash': 'Gemini 2.0 Flash',
    'score_gemini-2.5-pro-preview-03-25': 'Gemini 2.5 Pro Preview',
    'score_o3-2025-04-16': 'GPT o3',
    'score_claude-3-5-sonnet-20240620': 'Claude 3.5 Sonnet',
    'score_claude-3-sonnet-20240229': 'Claude 3 Sonnet',

    }

# Rename the occupation groups
occupation_group_mapping = {
    'business_and_financial_operations_occupations': 'Business & Financial Operations',
    'computer_and_mathematical_occupations': 'Computer & Mathematical',
    'management_occupations': 'Management'
}

def plot_sumbission_failures(df, fig_path):
    exams = df[df['exam'] !='Exam not valid']
    error_columns =  [col for col in df.columns if col.startswith('errors_')]
    error_columns = [col for col in error_columns if col != 'errors_empty']
    models = [col.removeprefix('errors_') for col in error_columns]

    failure_counts ={}
    for model in models:
        not_valid_json = 0
        ex_error = 0 
        zero_counts = 0
        for _, row in exams.iterrows():
            if not row['answer_valid_'+model]:
                not_valid_json = not_valid_json + 1
            elif not '[None]' ==row['errors_'+model]:
                ex_error +=1
            elif row['score_'+model]==0:
                zero_counts +=1
        failure_counts['score_'+model]={'invalid_json':not_valid_json, 'execution_error': ex_error, 'zero_score': zero_counts}
    
    answer_fails = pd.DataFrame(failure_counts).T.reset_index()
    answer_fails = answer_fails.melt(id_vars='index', var_name='error_type', value_name='count')

    # Example: preprocess data (assuming 'answer_fails' has 'index', 'error_type', 'count')
    answer_fails['index'] = answer_fails['index'].replace(model_dict)
    answer_fails['error_type'] = answer_fails['error_type'].replace({
        'invalid_json': 'Invalid JSON',
        'execution_error': 'Execution Error',
        'zero_score': 'Zero Score'
    })

    # Pivot the data to wide format for stacking
    pivot_df = answer_fails.pivot_table(index='index', columns='error_type', values='count', aggfunc='sum').fillna(0)

    # Sort columns to keep consistent stacking order
    pivot_df = pivot_df[['Invalid JSON', 'Execution Error', 'Zero Score']]  # adjust as needed

    # Set up colors
    barplot_colors = {
        'Invalid JSON': '#739E82',
        'Execution Error': '#E0777D',
        'Zero Score': '#8E3B46'
    }

    # Apply style
    theme_dict = {
        **sns.axes_style("white"),
        'axes.spines.right': False, 'axes.spines.left': False, 'axes.spines.top': False,
        "grid.linestyle": ":", 'legend.frameon': False, 'legend.facecolor': 'white',
        'legend.edgecolor': 'white', 'axes.titlesize': 18, 'axes.labelsize': 16,
        'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14,
        'legend.framealpha': 0
    }
    sns.set_theme(rc=theme_dict)

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = pd.Series([0]*len(pivot_df), index=pivot_df.index)

    for error_type in pivot_df.columns:
        ax.bar(pivot_df.index, pivot_df[error_type], label=error_type,
            bottom=bottom, alpha=.8, color=barplot_colors[error_type])
        bottom += pivot_df[error_type]

    # Aesthetics
    ax.set_title("LLM submission errors", fontsize=14)
    ax.set_xlabel("LLM")
    ax.set_ylabel("Count")
    plt.xticks(rotation=90)

    # Move legend inside
    ax.legend(title="Error type", loc="upper right", bbox_to_anchor=(0.98, 0.98), frameon=False)

    # White background
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print('Successfully saved figure to: ', fig_path)

def plot_overall_scores(df, fig_path, fill_na=True):
    exams = df[df['exam'] !='Exam not valid']
    score_cols = [col for col in df.columns if col.startswith('score')]
    score_cols = [col for col in score_cols if col != 'score_empty_submission']

    if fill_na:
        exam_scores = pd.melt(exams, id_vars=['task_id','occupation_group'], value_vars=score_cols,
                    var_name='Model', value_name='Value')
        exam_scores['Value'] = exam_scores['Value'].fillna(0)
    else:
        exams = exams.dropna(subset=score_cols)
        exam_scores = pd.melt(exams, id_vars=['task_id','occupation_group'], value_vars=score_cols,
                    var_name='Model', value_name='Value')

    exam_scores['Model'] = exam_scores['Model'].replace(model_dict)
    exam_scores['occupation_group'] = exam_scores['occupation_group'].replace(occupation_group_mapping)



    plt.figure(figsize=(8, 4))
    sns.barplot(x='Model', y='Value', errorbar=('sd'), hue='occupation_group', alpha=.7, data=exam_scores,palette = ['#FFBD59','#38B6FF','#8E3B46'])
    plt.xticks(rotation=90)

    # Set plot labels and title
    plt.title('Exam performance by LLM and occupation group, incl. standard deviation', fontsize=16)
    plt.xlabel('LLM')
    plt.ylabel('Exam score')
    plt.legend(
    title='Occupation group',
    frameon=True,
    loc='lower right',
    framealpha=0.6,
    )
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print('Successfully saved figure to: ', fig_path)


def correlation_exam_performance(df, fig_path):
    exams = df[df['exam'] !='Exam not valid']
    score_cols = [col for col in df.columns if col.startswith('score')]
    score_cols = [col for col in score_cols if col != 'score_empty_submission']
    # Rename the variables in the DataFrame
    exams_renamed = exams
    exams_renamed[score_cols] = exams[score_cols].fillna(0)
    exams_renamed = exams.rename(columns=model_dict)
    exams_renamed = exams_renamed.rename(columns={'Gemini 2.5 Pro Preview': 'Gemini 2.5 Pro'})




    exams_renamed['occupation_group'] = exams_renamed['occupation_group'].replace(occupation_group_mapping)

    g = sns.PairGrid(exams_renamed,vars=['Gemini 2.5 Pro', 'Claude 3.7 Sonnet', 'GPT o3', 'DeepSeek Chat'], \
        hue="occupation_group", palette = ['#FFBD59','#38B6FF','#8E3B46'], corner=True, height=1.8)
    g.map_diag(sns.histplot, alpha=0.7,multiple="stack", edgecolor='w')  # ,multiple="stack"Set transparency with alpha=0.4 (0 is fully transparent, 1 is fully opaque)
    g.map_lower(sns.scatterplot, edgecolor='w')
    g.add_legend(title="Occupation Group", loc='upper right', bbox_to_anchor=(0.8, 0.95), ncol=1, frameon=False, fontsize=12, title_fontsize=14)
    legend = g.legend
    legend.set_title("Occupation Group", prop={'size': 14})  # Use 'prop' to set font size for the title
    for ax in g.axes.flatten():
        if ax is not None:
            ax.set_xlabel(ax.get_xlabel(), fontsize=10)  # x-axis label
            ax.set_ylabel(ax.get_ylabel(), fontsize=10)  # y-axis label
            ax.tick_params(axis='both', labelsize=8)   
    # Set explicit axis labels for both x and y axes

    plt.savefig(fig_path, dpi=300, bbox_inches='tight')



if __name__ == "__main__":
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = 'claude-3-7-sonnet-20250219'


    occupations =['Business and Financial Operations Occupations',
    'Computer and Mathematical Occupations',
    'Management Occupations']

    occupations_file_names = [occ.lower().replace(' ', '_') for occ in occupations]
    exam_list = read_in_data(occupations_file_names, f'../data/test_results/{model}/test_results_')
    exam_list = mark_invalid(exam_list)
    plot_sumbission_failures(exam_list, f'../results/figures/{model}/{model}_submission_errors.pdf')
    plot_overall_scores(exam_list, f'../results/figures/{model}/{model}_overall_scores.pdf')
    correlation_exam_performance(exam_list, f'../results/figures/{model}/{model}_correlation_exam_performance.pdf')

