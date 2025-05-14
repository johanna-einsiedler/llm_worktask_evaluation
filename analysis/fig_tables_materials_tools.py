import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import numpy as np
from itertools import groupby
import itertools
import re
import matplotlib.ticker as mtick
import sys


def preprocess_materials_tools(df):
    tools_data = pd.DataFrame()
    materials_data = pd.DataFrame()
    for occ_group in occupations_file_names:
        materials, tools = get_shares(df_all[df_all['occupation_group'] == occ_group])
        materials.index = materials['Category'].str.replace('materials.', '', regex=True)
        materials.drop('Category', axis=1, inplace=True)

        tools.index = tools['Category'].str.replace('tools.', '', regex=True)
        tools.drop('Category', axis=1, inplace=True)
        materials_data = pd.concat([materials_data, materials.rename(columns={'Value':occ_group})],axis=1)
        tools_data = pd.concat([tools_data, tools.rename(columns={'Value':occ_group})], axis=1)

    return materials_data, tools_data

def read_in_data(occupations_file_names,path):
    df_all = pd.DataFrame()
    for occ in occupations_file_names:
        df = pd.read_csv(path+f'{occ}_CORE.csv')
        df['occupation_group'] = occ
        df_all = pd.concat([df_all, df], ignore_index=True)
    return df_all

    print(df_all.shape[0], ' tasks in the data')
def plot_grouped_horizontal_bars(ax, data, title, colors, bar_height, occupations_file_names):
    categories = data['Category'].str.replace('materials.|tools.', '', regex=True).unique()
    groups = occupations_file_names

    title_fontsize = 20
    label_fontsize = 18
    tick_fontsize = 18
    legend_fontsize = 18
    annot_fontsize = 14

    y = np.arange(len(categories))
    group_width = bar_height * len(groups)
    offsets = np.linspace(-group_width/2 + bar_height/2, group_width/2 - bar_height/2, len(groups))

    ax.set_title(title, fontsize=title_fontsize)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel('Percentage', fontsize=label_fontsize)
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=tick_fontsize)
    ax.tick_params(axis='x', which='major', labelsize=tick_fontsize)

    for spine in ax.spines.values():
        spine.set_visible(False)

    bars_list = []  # for legend handles

    for i, group in enumerate(groups):
        values = data[group].values

        bars = ax.barh(
            y + offsets[i],
            values,
            height=bar_height,
            color=colors[i],
            label=group.title().replace('_', ' '),
            alpha=0.7,
            edgecolor=colors[i],
        )

        bars_list.append(bars[0])  # Save one bar per group for the legend

        for bar in bars:
            width = bar.get_width()
            label = f"{width*100:.0f}%"
            if width > 0.2:
                ax.text(width - 0.02, bar.get_y() + bar.get_height()/2,
                        label, ha='right', va='center', color='black', fontsize=annot_fontsize)
            else:
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                        label, ha='left', va='center', fontsize=annot_fontsize)

    return bars_list  # Return bar objects for legend

def get_shares(df):

    material_columns = ['materials.Text', 'materials.Data',
    'materials.Images', 'materials.Audio files', 'materials.Video files',
    'materials.Virtual labs or sandbox environments']
    tool_columns =  ['tools.Coding', 'tools.Spreadsheets',\
    'tools.Text editor', 'tools.PDF viewer', 'tools.Presentation software',\
    'tools.Online search engine', 'tools.Image Generator']
    relevant_columns = material_columns + tool_columns
    df_binary = (df[relevant_columns] =='Required').astype(int)


    df_shares = df_binary.sum()/df.shape[0]
    df_shares = pd.DataFrame(df_shares).reset_index().rename(columns={0:'Value', 'index':'Category'})
    df_shares.loc[df_shares['Category']  == 'materials.Virtual labs or sandbox environments', 'Category'] ='materials.Virtual labs'
    # Separate materials and tools
    materials = df_shares[df_shares['Category'].str.startswith('materials')]
    tools = df_shares[df_shares['Category'].str.startswith('tools')]

    return [materials, tools]

def plot_tools_materials_requirements(materials_data, tools_data, fig_path):

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    fig.subplots_adjust(wspace=0.4)  # Increase space between the two plots




    # Plot materials
    bars_materials = plot_grouped_horizontal_bars(
        axes[0],
        materials_data.reset_index(),
        'Materials',
        ['#FFBD59','#38B6FF','#8E3B46'],
        0.3,
        occupations_file_names
    )

    # Plot tools
    bars_tools = plot_grouped_horizontal_bars(
        axes[1],
        tools_data.reset_index(),
        'Tools',
        ['#FFBD59','#38B6FF','#8E3B46'],
        0.3,
        occupations_file_names
    )

    # Joint legend at top center
    fig.legend(
        bars_materials,  # or bars_tools (they're the same bar styles)
        labels=[group.title().replace('_', ' ') for group in occupations_file_names],
        loc='upper center',
        ncol=len(occupations_file_names),
        fontsize=16,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08)  # Center it above the plots
    )

    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print('Successfully saved figure to ', fig_path)

def mark_invalid(df):
    # mark exams with empty entry, nan entry or key grade scores over 100 as invalid
    df.loc[df['exam']=='','exam'] = 'Exam not valid'
    df['exam'] = df['exam'].fillna('Exam not valid')
    df.loc[df['key_grade']>100,'exam'] = 'Exam not valid'
    df.loc[df['score_empty_submission']>0, 'exam'] = 'Exam not valid'
    df.loc[df['check_overall_makes_sense']==False, 'exam'] = 'Exam not valid'
    df.loc[df['check_no_internet']==False, 'exam'] = 'Exam not valid'
    df.loc[df['check_real_materials']==False, 'exam'] = 'Exam not valid'
    score_columns = df.filter(like='score', axis=1).columns
    df.loc[np.any(df[score_columns]>100,axis=1), 'exam'] = 'Exam not valid'
    return df

def create_exams_table(task_list, exams, table_path):
    # count number of tasks per occupation group
    overall_count = task_list.groupby('occupation_group')['task_id'].nunique().rename('\# of tasks')
    # count number of examined tasks per occupation group
    examined_tasks_count = exams[exams['exam']!='Exam not valid'].groupby('occupation_group')['task_id'].nunique().rename('\# of examined tasks')
    valid_exams = exams[exams['exam']!='Exam not valid'].groupby('occupation_group')['task_id'].nunique().rename('\# of valid exams')
    # create table with overview statistics
    stats_table= pd.concat([overall_count, examined_tasks_count], axis=1)
    stats_table['\% of examined tasks'] = np.round(stats_table['\# of examined tasks'] / stats_table['\# of tasks']*100,2)
    stats_table = pd.concat([stats_table, valid_exams], axis=1)
    stats_table['\% of valid exams'] = np.round(stats_table['\# of valid exams'] / stats_table['\# of examined tasks']*100,2)
    stats_table.index = [group.title().replace('_',' ') for group in occupations_file_names]
    stats_table = stats_table.round(2)
    latex_code = stats_table.to_latex(
    escape=False,   # Allows LaTeX special characters like '&' to appear correctly
    multicolumn=True,  # Adds multi-column support in the output for the column names
    header=True,    # Include the header row (column labels)
    index=True,
    float_format="%.2f"      # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print('Successfully saved table to ', table_path)


def plot_exams_share(task_list, exams, fig_path):
    task_count =task_list.groupby('occupation').size()
    attempt_count = exams.groupby('occupation').size()
    success_count = exams[exams['exam'] != 'Exam not valid'].groupby('occupation').size()

    counts = pd.concat([task_count, attempt_count, success_count], axis=1).rename(columns={0:'\# of tasks', 1:'\# of attempts', 2:'\# of successes'})
    counts = counts.merge(df_all[['occupation', 'occupation_group']].drop_duplicates(), on='occupation', how='left')
    counts = counts.sort_values(['occupation_group', 'occupation']).reset_index(drop=True)

    # Create a new DataFrame to store the final result
    new_rows = []
    i=0
    blank_row_names =[' ','  ','   ','    ','     ','      ']
    # Iterate over each group
    for group_name, group_df in counts.groupby('occupation_group'):
        # Create the label row
        label_row = {
            'occupation': group_name.replace('_', ' ').title(),  # optional: prettify
            '\# of tasks': np.nan,
            '\# of attempts': np.nan,
            '\# of successes': np.nan,
            'occupation_group': group_name
        }
        blank_row = {
            'occupation': blank_row_names[i],  # optional: prettify
            '\# of tasks': np.nan,
            '\# of attempts': np.nan,
            '\# of successes': np.nan,
            'occupation_group': np.nan
        }
        new_rows.append(pd.DataFrame([blank_row]))
        i+=1
        # Append the label row and the group rows
        new_rows.append(pd.DataFrame([label_row]))
        blank_row = {
        'occupation': blank_row_names[i],  # optional: prettify
        '\# of tasks': np.nan,
        '\# of attempts': np.nan,
        '\# of successes': np.nan,
        'occupation_group': np.nan
        }
        new_rows.append(pd.DataFrame([blank_row]))
        new_rows.append(group_df)
        i+=1

    # Concatenate all parts into one final DataFrame
    df_with_labels = pd.concat(new_rows, ignore_index=True)

    # Create 2x2 grid of subplots
    fig, axes = plt.subplots(
        1, 3,
        figsize=(20, 8),
        #gridspec_kw={"height_ratios": [5, 3]}  # Top row is 2x the height of the bottom row
    )
    axes = axes.flatten()
    #bar_colors=[#FFBD59, '#38B6FF',#8E3B46]
    bar_colors = [
        ("#739E82", "\# of tasks", '# of core tasks'),
        ("#E0777D", "\# of attempts", "# of exams attempted"),
        ("#8E3B46", "\# of successes","# of created exams")
    ]

    # Plot for each occupation group
    for idx, group in enumerate(occupations_file_names):
        ax = axes[idx]
        df_occ = counts[counts['occupation_group'] == group]

        for color, name,label in bar_colors:
            sns.barplot(
                x=name,  # gets 'tasks', 'exams', 'successes'
                y="occupation",
                data=df_occ,
                label=label,
                color=color,
                alpha=.8,
                edgecolor=color,
                ax=ax,
                legend=False
            )

        ax.set(xlim=(0, 24), ylabel="", xlabel="# of core tasks per occupation")
        sns.despine(ax=ax, left=True, bottom=True)
        ax.set_title(group.replace("_", " ").title(), fontsize=14)

        # Clean up ticks
        ax.tick_params(left=False)
        ax.set_yticklabels([label.get_text() for label in ax.get_yticklabels()])

    # Fourth plot (bottom right) is used for legend only
    #legend_ax = axes[3]
    #legend_ax.axis('off')  # Hide the axis

    # Create dummy bars for the legend
    handles = [
        plt.Rectangle((0, 0), 1, 1, color=color, alpha=0.8)
        for color, name, label in bar_colors
    ]
    labels = [label for _,_, label in bar_colors]
    #legend_ax.legend(handles, labels, loc='upper left', fontsize=14, title="Legend")
    fig.legend(handles, labels, loc='upper center', fontsize=12, title="Legend", bbox_to_anchor=(0.5, 1.05), ncol=3,title_fontsize=16)

    # Adjust layout
    plt.tight_layout(pad=5.0)  # Increase padding between subplots
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')

def create_failure_modes_table(df, table_path):
    invalid_exams = df[df['exam']=='Exam not valid'].groupby('occupation_group')['task_id'].nunique().rename('\# of invalid exams')
    key_grade_low = df[df['key_grade']<99].groupby('occupation_group')['task_id'].nunique().rename('\% of answer keys with score \< 99')
    key_grade_high = df[df['key_grade']>100].groupby('occupation_group')['task_id'].nunique().rename('\% of answer keys with score \> 100')
    check_no_internet = df[df['check_no_internet']==False].groupby('occupation_group')['task_id'].nunique().rename('\% of exams relying on fabricated websites')
    failed_candidate_materials = df[df['failed_candidate_materials']>2].groupby('occupation_group')['task_id'].nunique().rename('\% of exam without separate candidate materials')        
    check_overall_makes_sense = df[df['check_overall_makes_sense']==False].groupby('occupation_group')['task_id'].nunique().rename('\% of exams not making sense overall')
    check_real_materials = df[df['check_real_materials']==False].groupby('occupation_group')['task_id'].nunique().rename('\% of exams relying on non-existing image materials')
    failure_modes = pd.concat([ invalid_exams,key_grade_low, key_grade_high, check_no_internet, failed_candidate_materials, check_overall_makes_sense, check_real_materials], axis=1).fillna(0)
    failure_modes =  failure_modes.div(failure_modes['\# of invalid exams'], axis=0).drop('\# of invalid exams', axis=1)
    failure_modes = np.round(failure_modes*100,2)
    failure_modes.index = [group.title().replace('_',' ') for group in occupations_file_names]
    latex_code = failure_modes.to_latex(
        escape=False,   # Allows LaTeX special characters like '&' to appear correctly
        multicolumn=True,  # Adds multi-column support in the output for the column names
        header=True,    # Include the header row (column labels)
        index=True,
        float_format="%.2f"      # Include the index (row labels)
    )
    with open(table_path, "w") as f:
        f.write(latex_code)
    print('Successfully saved table to ', table_path)

if __name__ == "__main__":
    if len(sys.argv) >1:
        model = sys.argv[1]
    else:
        model = 'claude-3-7-sonnet-20250219'


    occupations =['Business and Financial Operations Occupations',
    'Computer and Mathematical Occupations',
    'Management Occupations']
    occupations_file_names = [occ.lower().replace(' ', '_') for occ in occupations]

    df_all = read_in_data(occupations_file_names,f'../data/required_materials_tools/{model}/materials_tools_')
   # Print the number of tasks in the file

    # preprocess data to extract materials and tools lists
    materials_data, tools_data = preprocess_materials_tools(df_all)

    # create figure showing percetages of tools and materials required for each occupation group
    plot_tools_materials_requirements(materials_data, tools_data, f'../results/figures/{model}/{model}_tools_materials_requirements.pdf')

    # get exams
    exam_list = read_in_data(occupations_file_names, f'../data/test_results/{model}/test_results_')

    exam_list = mark_invalid(exam_list)
    create_exams_table(df_all, exam_list, f'../results/tables/{model}/{model}_exams_overview.tex')
    plot_exams_share(df_all, exam_list, f'../results/figures/{model}/{model}_exams_shares_overview.pdf')
    create_failure_modes_table(exam_list, f'../results/tables/{model}/{model}_xams_failure_modes.tex')
