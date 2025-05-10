import numpy as np
import pandas as pd
import sys


def read_tsv(path):
    """
    Reads a TSV (Tab-Separated Values) file and returns a pandas DataFrame.
    
    Parameters:
    path (str): Path to the TSV file.
    
    Returns:
    pd.DataFrame: The loaded DataFrame.
    """
    return pd.read_csv(path, delimiter="\t")


def filter_occupation_group(df, occupation_group):
    """
    Filters tasks based on occupation group using SOC codes.
    
    Parameters:
    df (pd.DataFrame): The DataFrame containing task data.
    occupation_group (str): The occupation group to filter for.
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing tasks for the specified occupation group.
    """
    print(f"Filtering for occupation group: {occupation_group}")
    
    # Extract the first two digits of SOC codes for higher-level categorization
    df['SOC-2digit'] = df['O*NET-SOC Code'].astype(str).str.split('-').str[0]
    
    # Load SOC code definitions
    soc = pd.read_excel("../data/external/soc_data/soc_2018_definitions.xlsx")
    
    # Filter to only major SOC categories
    soc = soc[soc['SOC Group'] == 'Major']
    soc['SOC-2digit'] = soc['SOC Code'].astype(str).str.split('-').str[0]
    # Merge occupation data
    df = df.merge(soc, how='left', on='SOC-2digit')
    df.drop(['SOC Definition','SOC Group','SOC Code'], axis=1, inplace=True)

    # Filter for the specific occupation group
    df = df[df['SOC Title'] == occupation_group]
    print(f"Resulting DataFrame contains {df.shape[0]} tasks")

    df.rename(columns={'O*NET-SOC Code': 'soc_code', 'Task ID': 'task_id', 'Task': 'task','Task Type':'task_type',\
                       'Title':'title'}, inplace=True)
    
    print(" There are ", len(df['title'].unique()), " occupations in this group.")
    print("There are ", df.shape[0], " task pertaining to ", occupation_group)
    print("On average an occupation has ", df['title'].value_counts().mean(), ' tasks listed.')
    print("The minimum number of tasks is ", df['title'].value_counts().min())
    print("The maximum number of tasks is ", df['title'].value_counts().max())
    print("Share of Core vs. Supplemental tasks ", df['task_type'].value_counts()/df.shape[0])
    return df

if __name__ == "__main__":

    if len(sys.argv) >1:
        occupation_group = sys.argv[1]
    else:
        occupation_group = "Management Occupations"
    if len(sys.argv)>2:
        core_only = sys.argv[2]
    else:
        core_only = True

df = read_tsv('../data/external/gpts-are-gpts/full_labelset.tsv')
df = filter_occupation_group(df, occupation_group=occupation_group)
if df.shape[0] ==0:
    print("No tasks found for the specified occupation group.")
    sys.exit(1)
else:
    print("There are ", df.shape[0], " tasks within ", occupation_group)
    df.to_csv(f'../data/task_lists/{occupation_group.replace(" ", "_").lower()}.csv')

    core_label = 'CORE' if core_only else ''
    # filter core vs. supplemental
    if core_only:
        df=df[df['task_type'] =='Core']
        print("There are ", df.shape[0]," CORE tasks within ", occupation_group)

    save_path = f'../data/task_lists/{occupation_group.replace(" ", "_").lower()}_{core_label}.csv'
    df.to_csv(save_path, index=False)