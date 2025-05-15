import os
import json
import ast
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import matplotlib.dates as mdates
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
from matplotlib.ticker import FuncFormatter, LogFormatter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
import matplotlib.lines as mlines
import matplotlib.ticker as ticker
from adjustText import adjust_text
from scipy import stats

# Define paths
path_test_data = '../../data/exam_approach/test_results/claude-3-7-sonnet-20250219/'
path_epoch = '../../data/external/epoch_ai/'

# Color palette and markers for all visualizations
COLOR_PALETTE = ['#FFBD59', '#38B6FF', '#8E3B46', '#E0777D', '#739E82']
ORGANIZATION_MARKERS = {
    "Anthropic": "o",  # Circle
    "OpenAI": "s",     # Square
    "Google DeepMind": "D",  # Diamond
    "DeepSeek": "^"    # Triangle
}

# Model name mapping dictionaries - define only once
MODEL_MAPPING = {
    'claude-3-7-sonnet-20250219': 'score_claude_sonnet',
    'gpt-4o': 'score_chatgpt4o',
    'deepseek-chat': 'score_deepseek',
    'gemini-1.5-flash': 'score_gemini_flash_15',
    'gemini-2.0-flash': 'score_gemini_flash',
    "claude-3-5-sonnet-202410": 'score_claude_sonnet_35',
    'claude-3-5-haiku-202410': 'score_claude_haiku',
    'gpt-3.5-turbo-0125': 'score_chatgpt35',
    'gemini-2.5-pro-preview-03-25': 'score_gemini_25',
    'o3-2025-04-16': 'score_chatgpt_o3',
    'claude-3-sonnet': 'score_sonnet30'
}

# Reverse mapping for display names
REVERSE_MAPPING = {
    'score_claude_sonnet': 'Claude 3.7 Sonnet',
    'score_chatgpt4o': 'GPT-4o',
    'score_deepseek': 'DeepSeek Chat',
    'score_gemini_flash_15': 'Gemini 1.5 Flash',
    'score_gemini_flash': 'Gemini 2.0 Flash',
    'score_claude_haiku': 'Claude 3.5 Haiku',
    'score_chatgpt35': 'GPT-3.5 Turbo',
    'score_claude_sonnet_35': 'Claude 3.5 Sonnet',
    'score_gemini_25': 'Gemini 2.5',
    'score_chatgpt_o3': 'GPT o3',
    'score_sonnet30': 'Claude 3 Sonnet'
}

# Model dictionary for epoch_ai data mapping
MODEL_DICT = {
    "claude-3-7-sonnet-20250219": ["claude-3-7-sonnet-20250219", "Claude 3.7 Sonnet"],
    "gpt-4o": ["gpt-4o-2024-08-06", "GPT-4o"],
    "deepseek-chat": ["DeepSeek-V3", "DeepSeek-V3"],
    "gemini-1.5-flash": ["gemini-1.5-flash-002", "Gemini 1.5 Pro"],
    "gemini-2.0-flash": ["gemini-2.0-flash-001", "NA"],
    "claude-3-5-sonnet-202410": ["claude-3-5-sonnet-20241022", "Claude 3.5 Sonnet"],
    "gpt-3.5-turbo-0125": ['gpt-3.5-turbo-0125', "GPT-3.5 Turbo"],
    "gemini-2.5-pro-preview-03-25": ["gemini-2.5-flash-preview-04-17", "NA"],
    "o3-2025-04-16": ["o3-2025-04-16_high", "NA"],
    "claude-3-sonnet": ['claude-3-sonnet-20240229', 'Claude 3 Sonnet']
}

# Define model families
MODEL_FAMILIES = {
    "claude": {k: v for k, v in REVERSE_MAPPING.items() if 'claude' in k.lower()},
    "gemini": {k: v for k, v in REVERSE_MAPPING.items() if 'gemini' in k.lower()},
    "gpt": {k: v for k, v in REVERSE_MAPPING.items() if 'gpt' in k.lower() or 'chatgpt' in k.lower()}
}
# Add Claude 3 Sonnet to Claude family
MODEL_FAMILIES["claude"]["score_sonnet30"] = 'Claude 3 Sonnet'

# Remove 'score_claude_haiku' from the 'claude' family
if 'score_claude_haiku' in MODEL_FAMILIES['claude']:
    del MODEL_FAMILIES['claude']['score_claude_haiku']

# Print the updated MODEL_FAMILIES to verify
print(MODEL_FAMILIES)

# Model family version orders
MODEL_FAMILY_ORDERS = {
    'claude': {'score_sonnet30': 0, 'score_claude_sonnet_35': 1, 'score_claude_sonnet': 2},
    'gemini': {'score_gemini_flash_15': 0, 'score_gemini_flash': 1, 'score_gemini_25': 2},
    'gpt': {'score_chatgpt35': 0, 'score_chatgpt4o': 1, 'score_chatgpt_o3': 2}
}

# Model family middle and last columns
MODEL_FAMILY_VERSIONS = {
    'claude': {'middle': 'score_claude_sonnet_35', 'last': 'score_claude_sonnet'},
    'gemini': {'middle': 'score_gemini_flash', 'last': 'score_gemini_25'},
    'gpt': {'middle': 'score_chatgpt4o', 'last': 'score_chatgpt_o3'},
}

def load_exam_data():
    """
    Load and process all exam data
    """
    # Define a dictionary with occupation categories as keys and file names as values
    files_score = {
        "business_and_financial_operations": "scores_only_business_and_financial_operations_occupations.csv",
        "computer_and_mathematical": "scores_only_computer_and_mathematical_occupations.csv",
        "management": "scores_only_management_occupations.csv"
    }
    
    # Initialize an empty list to store DataFrames
    dataframes = []
    
    # Load and process score files
    for category, file_name in files_score.items():
        df = pd.read_csv(path_test_data + file_name)
        # Remove the 'Unnamed: 0' column
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
        # Add the 'occupation_category' column
        df['occupation_category'] = category
        # Append the processed DataFrame to the list
        dataframes.append(df)
    
    # Concatenate all DataFrames into one
    all_exams = pd.concat(dataframes, ignore_index=True)
    
    return all_exams

def load_exam_results():
    """
    Load and process all exam results
    """
    occupations = [
        'Business and Financial Operations Occupations',
        'Computer and Mathematical Occupations',
        'Management Occupations'
    ]
    
    occupations_file_names = [occ.lower().replace(' ', '_') for occ in occupations]
    
    exam_list = pd.DataFrame()
    for occ in occupations_file_names:
        results = pd.read_csv(
            f'../../data/exam_approach/test_results/claude-3-7-sonnet-20250219/test_results_{occ}.csv',
            index_col=0
        )
        results = results.loc[:, ~results.columns.str.startswith('Unnamed')]
        # NOTE renamed here to align with old code
        results['occupation_category'] = occ
        exam_list = pd.concat([exam_list, results], axis=0, ignore_index=True)
    
    # Mark exams with empty entry, nan entry or key grade scores over 100 as invalid
    exam_list.loc[exam_list['exam'] == '', 'exam'] = 'Exam not valid'
    exam_list['exam'] = exam_list['exam'].fillna('Exam not valid')
    exam_list.loc[exam_list['key_grade'] > 100, 'exam'] = 'Exam not valid'
    exam_list.loc[exam_list['check_overall_makes_sense']==False, 'exam'] = 'Exam not valid'

    exams = exam_list[exam_list['exam'] != 'Exam not valid']
    
    return exams

def load_model_info():
    """
    Load and process model info data from epoch_ai
    """
    # Select family of models to focus on
    model_family = ["GPT", "Claude", "Gemini", "DeepSeek"]
    
    # Load model info
    df_model_info = pd.read_csv(path_epoch + 'notable_ai_models.csv')
    df_model_info = df_model_info[
        df_model_info['Model'].str.contains('|'.join(model_family), case=False, na=False)
    ]
    
    # Load benchmark data
    df_model_benchmark = pd.read_csv(path_epoch + 'benchmark_data/benchmarks_runs.csv')
    
    model_family_lower = [model.lower() for model in model_family]
    
    # Filter the DataFrame based on whether any keyword is in the 'model' column
    df_model_benchmark = df_model_benchmark[
        df_model_benchmark['model'].str.lower().str.contains('|'.join(model_family_lower), na=False) |
        (df_model_benchmark['model'] == 'o3-2025-04-16_high')
    ]
    
    df_model_benchmark = df_model_benchmark.sort_values(by='model', ascending=True)
    
    return df_model_info, df_model_benchmark

def create_model_benchmark_dataframe(df_model_info, df_model_benchmark):
    """
    Create model benchmark dataframe with combined data
    """
    # Pivot df_model_benchmark to make 'task' the columns and 'Best score (across scorers)' the values
    df_model_benchmark_pivot = df_model_benchmark.pivot_table(
        index='model', 
        columns='task', 
        values='Best score (across scorers)', 
        aggfunc='first'
    ).reset_index()
    
    # Remove the name of the columns
    df_model_benchmark_pivot.columns.name = None
    df_model_benchmark_pivot = df_model_benchmark_pivot.rename_axis(None, axis=1)
    
    # Define the columns for the new DataFrame
    columns = [
        "model", "Publication date", "Organization", "Organization categorization",
        "Parameters", "Training compute (FLOP)", "Training time (hours)",
        "Training compute cost (2023 USD)", "Model accessibility",
        "MATH level 5", "GPQA diamond", "OTIS Mock AIME 2024-2025",
        "FrontierMath-2025-02-28-Public", "FrontierMath-2025-02-28-Private",
        "SWE-Bench verified"
    ]
    
    # Iterate over the MODEL_DICT
    data = []
    for model_key, values in MODEL_DICT.items():
        # Extract the second value from the dict list for df_model_info lookup
        model_info_key = values[1]
        # Extract the first value from the dict list for df_model_benchmark lookup
        model_benchmark_key = values[0]
        
        # Get the row from df_model_info matching the model_info_key
        info_row = df_model_info[df_model_info['Model'] == model_info_key]
        
        # Get the row from the pivoted df_model_benchmark matching the model_benchmark_key
        benchmark_row = df_model_benchmark_pivot[df_model_benchmark_pivot['model'] == model_benchmark_key]
        
        # Extract the required values from df_model_info and df_model_benchmark
        row = {
            "model": model_key,
            "Publication date": info_row['Publication date'].values[0] if not info_row.empty else None,
            "Organization": info_row['Organization'].values[0] if not info_row.empty else None,
            "Organization categorization": info_row['Organization categorization'].values[0] if not info_row.empty else None,
            "Parameters": info_row['Parameters'].values[0] if not info_row.empty else None,
            "Training compute (FLOP)": info_row['Training compute (FLOP)'].values[0] if not info_row.empty else None,
            "Training time (hours)": info_row['Training time (hours)'].values[0] if not info_row.empty else None,
            "Training compute cost (2023 USD)": info_row['Training compute cost (2023 USD)'].values[0] if not info_row.empty else None,
            "Model accessibility": info_row['Model accessibility'].values[0] if not info_row.empty else None,
            "MATH level 5": benchmark_row['MATH level 5'].values[0] if not benchmark_row.empty else None,
            "GPQA diamond": benchmark_row['GPQA diamond'].values[0] if not benchmark_row.empty else None,
            "OTIS Mock AIME 2024-2025": benchmark_row['OTIS Mock AIME 2024-2025'].values[0] if not benchmark_row.empty else None,
            "FrontierMath-2025-02-28-Public": benchmark_row['FrontierMath-2025-02-28-Public'].values[0] if not benchmark_row.empty else None,
            "FrontierMath-2025-02-28-Private": benchmark_row['FrontierMath-2025-02-28-Private'].values[0] if not benchmark_row.empty else None,
            "SWE-Bench verified": benchmark_row['SWE-Bench verified'].values[0] if not benchmark_row.empty else None
        }
        
        # Append the row to the data list
        data.append(row)
    
    # Create the new DataFrame
    df_model_bench = pd.DataFrame(data, columns=columns)
    
    # Update the publication date and FLOPs that are not available in latest csv file
    manual_updates = {
        "gemini-2.0-flash": {"Publication date": "2025-02-05", 
                             "Training compute (FLOP)": 2.43e+25,
                             "Organization": 'Google DeepMind'},
        "gemini-2.5-pro-preview-03-25": {"Publication date": "2025-03-01", 
                                          "Training compute (FLOP)": 5.6e+25,
                                          "Organization": 'Google DeepMind'},
        "o3-2025-04-16": {"Publication date": "2025-01-31", 
                          "Training compute (FLOP)": 8e+25, # from LessWrong
                          "Organization": 'OpenAI'},
        "gpt-3.5-turbo-0125": {"Training compute (FLOP)": 2.58e+24}
    }
    
    # Apply the manual updates
    for model, updates in manual_updates.items():
        for field, value in updates.items():
            df_model_bench.loc[df_model_bench['model'] == model, field] = value
    
    return df_model_bench

def calculate_scores_by_category(all_exams, df_model_bench):
    """
    Calculate category-specific performance metrics for each model
    """
    # Convert publication date to datetime
    df_model_bench['Publication date'] = pd.to_datetime(df_model_bench['Publication date'], errors='coerce')
    
    # Sort the model bench by publication date
    df_model_bench_sorted = df_model_bench.sort_values(by='Publication date', na_position='last')
    
    # Calculate average score for each model across all tasks (handling NaN values)
    avg_scores = {}
    std_scores = {}
    occupation_categories = all_exams['occupation_category'].unique()
    
    for model_name, score_col in MODEL_MAPPING.items():
        # Use nanmean and nanstd to properly handle missing values
        avg_scores[model_name] = all_exams[score_col].fillna(0).mean(skipna=True)
        std_scores[model_name] = all_exams[score_col].fillna(0).std(skipna=True)
    
    # Create dataframe with average scores and model info
    avg_scores_df = pd.DataFrame(list(avg_scores.items()), columns=['model', 'avg_score'])
    avg_scores_df['std_score'] = avg_scores_df['model'].map(std_scores)
    avg_scores_df = pd.merge(
        avg_scores_df, 
        df_model_bench[['model', 'Publication date', 'Training compute (FLOP)']], 
        on='model', how='left'
    )
    
    # Create a dataframe to hold category-specific scores
    category_scores_data = []
    
    for model_name, score_col in MODEL_MAPPING.items():
        for category in occupation_categories:
            if pd.notna(category):  # Skip NaN categories
                category_data = all_exams[all_exams['occupation_category'] == category]
                
                # Fill NA values with 0
                category_data[score_col] = category_data[score_col].fillna(0)
                
                # Calculate metrics for this model and category
                mean_score = category_data[score_col].mean(skipna=True)
                std_score = category_data[score_col].std(skipna=True)
                min_score = category_data[score_col].min(skipna=True)
                max_score = category_data[score_col].max(skipna=True)
                q1_score = category_data[score_col].quantile(0.25)
                q3_score = category_data[score_col].quantile(0.75)
                
                # Only add if there are valid scores
                if not pd.isna(mean_score):
                    category_scores_data.append({
                        'model': model_name,
                        'display_name': REVERSE_MAPPING.get(score_col, model_name),
                        'category': category,
                        'mean_score': mean_score,
                        'std_score': std_score,
                        'min_score': min_score,
                        'max_score': max_score,
                        'q1_score': q1_score,
                        'q3_score': q3_score,
                        'count': category_data[score_col].count()
                    })
    
    # Create dataframe from the collected data
    category_scores_df = pd.DataFrame(category_scores_data)
    
    # Add publication date and FLOP info
    category_scores_df = pd.merge(
        category_scores_df,
        df_model_bench[['model', 'Publication date', 'Training compute (FLOP)', 'MATH level 5', 'Organization']],
        on='model',
        how='left'
    )
    
    # Update the main model benchmark dataframe with average scores
    for model_name in df_model_bench['model']:
        if model_name in avg_scores:
            # Add columns for overall average score
            df_model_bench.loc[df_model_bench['model'] == model_name, 'avg_all_tasks_score'] = avg_scores[model_name]
            df_model_bench.loc[df_model_bench['model'] == model_name, 'std_all_tasks_score'] = std_scores[model_name]
            
            # Add columns for category-specific average scores
            for category in occupation_categories:
                if pd.notna(category):
                    category_subset = category_scores_df[
                        (category_scores_df['model'] == model_name) & 
                        (category_scores_df['category'] == category)
                    ]
                    if not category_subset.empty:
                        col_name = f'avg_score_{category}'
                        df_model_bench.loc[df_model_bench['model'] == model_name, col_name] = category_subset['mean_score'].values[0]
    
    return category_scores_df, df_model_bench, avg_scores_df

def plot_category_performance(all_exams, df_model_bench, category_scores_df):
    """
    Create bar plot of model performance by category
    
    Parameters:
    -----------
    all_exams : DataFrame
        DataFrame containing all exam data
    df_model_bench : DataFrame
        DataFrame containing model benchmark data
    category_scores_df : DataFrame
        DataFrame containing category-specific scores
    """
    # Get proper model ordering for plots based on publication date
    models_with_dates = df_model_bench.dropna(subset=['Publication date']).copy()
    model_order = models_with_dates.sort_values('Publication date')['model'].tolist()
    
    # For models without dates, add them at the end
    models_without_dates = [m for m in df_model_bench['model'] if m not in model_order and m in MODEL_MAPPING]
    model_order.extend(models_without_dates)
    
    # Create model display names dictionary
    model_display_names = {model: get_display_name(model) for model in model_order}
    
    plt.figure(figsize=(15, 8))
    
    # Prepare data for plotting
    plot_data = []
    occupation_categories = all_exams['occupation_category'].unique()
    
    for model in model_order:
        for category in occupation_categories:
            if pd.notna(category):
                subset = category_scores_df[
                    (category_scores_df['model'] == model) & 
                    (category_scores_df['category'] == category)
                ]
                if not subset.empty:
                    plot_data.append({
                        'model': model_display_names[model],
                        'category': category,
                        'score': subset['mean_score'].values[0]
                    })
    
    plot_df = pd.DataFrame(plot_data)
    
    # Only proceed if we have data
    if not plot_df.empty:
        # Convert to pivot format for grouped bar chart
        pivot_df = plot_df.pivot(index='model', columns='category', values='score')
        
        # Plot
        ax = pivot_df.plot(kind='bar', figsize=(15, 8), width=0.8, color=COLOR_PALETTE)
        
        plt.title('Model Performance by Occupation Category', fontsize=14)
        plt.xlabel('Model', fontsize=12)
        plt.ylabel('Average Score (0-100)', fontsize=12)
        plt.grid(True, alpha=0.3, axis='y')
        plt.legend(title='Occupation Category', fontsize=10)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        # Add value labels on bars
        for container in ax.containers:
            ax.bar_label(container, fmt='%.1f', fontsize=8)
        
        return plt
    
    return None

def create_marker_legend_handles():
    """
    Create marker legend handles for organization markers
    """
    return [
        mlines.Line2D([], [], color='black', marker='o', linestyle='None', markersize=10, label='Anthropic'),
        mlines.Line2D([], [], color='black', marker='s', linestyle='None', markersize=10, label='OpenAI'),
        mlines.Line2D([], [], color='black', marker='D', linestyle='None', markersize=10, label='Google DeepMind'),
        mlines.Line2D([], [], color='black', marker='^', linestyle='None', markersize=10, label='DeepSeek')
    ]

def plot_scatter_performance_vs_time_by_category(category_scores_df):
    """
    Create scatter plot of performance vs time by category
    """
    # We'll use category-level data instead of average scores
    time_data = category_scores_df.dropna(subset=['Publication date']).copy()
    sns.set_context("talk", font_scale=1.3)

    if time_data.empty:
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Track which models we've already labeled to avoid duplicates
    labeled_models = set()
    texts = [] 
    
    # Plot each category with a different color
    categories = time_data['category'].unique()
    marker_legend_handles = create_marker_legend_handles()
    
    for i, category in enumerate(categories):
        category_subset = time_data[time_data['category'] == category]
        
        # Group by model to handle error bars
        models = category_subset['model'].unique()
        for model in models:
            model_data = category_subset[category_subset['model'] == model]
            
            # Determine the marker type based on the organization
            organization = model_data['Organization'].iloc[0] if not model_data.empty else "Other"
            marker = ORGANIZATION_MARKERS.get(organization, "o")  # Default to circle if not found
            
            # Check if we should label this model
            should_label = model not in labeled_models and category == 'computer_and_mathematical_occupations'
            if should_label:
                labeled_models.add(model)
            
            # Scatter plot for this category and model
            plt.scatter(
                model_data['Publication date'], 
                model_data['mean_score'],
                s=150,
                c=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                marker=marker,  # Use the marker type based on the organization
                label=category.replace('_', ' ').title() if model == models[0] else "",  # Only label once per category
                alpha=0.7
            )
            
            # Add error bars for standard deviation
            plt.errorbar(
                model_data['Publication date'], 
                model_data['mean_score'],
                yerr=model_data['std_score'],
                fmt='none',
                ecolor='gray',
                capsize=5,
                alpha=0.5
            )
            
            # Add model labels, but only once per model
            if should_label:
                for _, row in model_data.iterrows():
                    text = plt.text(
                        row['Publication date'], 
                        row['mean_score'], 
                        get_display_name(row['model']),
                        fontsize=12
                    )
                    texts.append(text)  # Add text to the list for adjustment
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5), fontsize=12)
    
    # Add trend lines for each category
    for i, category in enumerate(categories):
        category_subset = time_data[time_data['category'] == category]
        
        if len(category_subset) > 1:
            # Convert dates to numeric for trend line
            x_numeric = np.array([(d - pd.Timestamp('2022-01-01')).days for d in category_subset['Publication date']])
            y = category_subset['mean_score'].values
            
            # Only add trend line if we have enough data points
            if len(x_numeric) > 1:
                z = np.polyfit(x_numeric, y, 1)
                p = np.poly1d(z)
                
                # Generate points for trend line
                x_range = np.linspace(min(x_numeric), max(x_numeric), 100)
                x_dates = [pd.Timestamp('2022-01-01') + pd.Timedelta(days=int(x)) for x in x_range]
                
                plt.plot(x_dates, p(x_range), '--', color=COLOR_PALETTE[i % len(COLOR_PALETTE)], alpha=0.7)
    
    plt.xlabel('Publication Date', fontsize=14)
    plt.ylabel('Average Score by Category', fontsize=14)
    plt.title('Model Performance Over Time by Occupation Category', fontsize=16)
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(MonthLocator(interval=3))
    plt.xticks(rotation=45, fontsize=12)
    plt.yticks(fontsize=12)
    
    # Despine - remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add the marker legend for organizations
    organization_legend = plt.legend(handles=marker_legend_handles, title='Organization', fontsize=12, loc='lower right')

    # Add the color legend for categories
    category_legend = plt.legend(title='Occupation Category', fontsize=12, loc='upper left')

    # Ensure both legends are displayed
    plt.gca().add_artist(organization_legend)
    
    plt.tight_layout()
    
    return plt

def plot_scatter_performance_vs_compute_by_category(category_scores_df):
    """
    Create scatter plot of performance vs compute by category
    """
    # We'll use category-level data instead of average scores
    compute_data = category_scores_df.dropna(subset=['Training compute (FLOP)']).copy()
    
    if compute_data.empty:
        return None
    
    plt.figure(figsize=(12, 8))
    
    # Track which models we've already labeled to avoid duplicates
    labeled_models = set()
    texts = [] 
    
    # Plot each category with a different color
    categories = compute_data['category'].unique()
    marker_legend_handles = create_marker_legend_handles()
    
    for i, category in enumerate(categories):
        category_subset = compute_data[compute_data['category'] == category]
        
        # Group by model to handle error bars
        models = category_subset['model'].unique()
        for model in models:
            model_data = category_subset[category_subset['model'] == model]
            
            # Determine the marker type based on the organization
            organization = model_data['Organization'].iloc[0] if not model_data.empty else "Other"
            marker = ORGANIZATION_MARKERS.get(organization, "o")  # Default to circle if not found
            
            # Check if we should label this model
            should_label = model not in labeled_models
            if should_label:
                labeled_models.add(model)
            
            # Scatter plot for this category and model
            plt.scatter(
                model_data['Training compute (FLOP)'], 
                model_data['mean_score'],
                s=150,
                c=COLOR_PALETTE[i % len(COLOR_PALETTE)],
                marker=marker,  # Use the marker type based on the organization
                label=category.replace('_', ' ').title() if model == models[0] else "",  # Only label once per category
                alpha=0.7
            )
            
            # Add error bars for standard deviation
            plt.errorbar(
                model_data['Training compute (FLOP)'], 
                model_data['mean_score'],
                yerr=model_data['std_score'],
                fmt='none',
                ecolor='gray',
                capsize=5,
                alpha=0.5
            )
            
            # Add model labels, but only once per model
            if should_label:
                for _, row in model_data.iterrows():
                    text = plt.text(
                        row['Training compute (FLOP)'], 
                        row['mean_score'], 
                        get_display_name(row['model']),
                        fontsize=12
                    )
                    texts.append(text)  # Add text to the list for adjustment
    
    adjust_text(texts, arrowprops=dict(arrowstyle='-', color='gray', alpha=0.5), fontsize=12)
    
    plt.xlabel('Training Compute (FLOP)', fontsize=14)
    plt.ylabel('Average Score by Category', fontsize=14)
    plt.title('Model Performance vs Training Compute by Occupation Category', fontsize=16)
    
    # Log scale for x-axis since compute varies by orders of magnitude
    plt.xscale('log')
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    
    # Format x-axis to show scientific notation nicely
    formatter = ticker.LogFormatter(10, labelOnlyBase=False)
    plt.gca().xaxis.set_major_formatter(formatter)
    
    # Despine - remove top and right spines
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # Add legend
    organization_legend = plt.legend(handles=marker_legend_handles, title='Organization', fontsize=12, loc='lower right')

    # Add the color legend for categories
    category_legend = plt.legend(title='Occupation Category', fontsize=12, loc='upper left')

    # Ensure both legends are displayed
    plt.gca().add_artist(organization_legend)
    
    plt.tight_layout()
    
    return plt

def analyze_correlations(category_scores_df):
    """
    Calculate correlations, R², and p-values for model performance vs. time and vs. compute
    across different occupation categories.
    
    Returns a DataFrame with the analysis results.
    """
    results = []
    
    # Categories to analyze
    categories = category_scores_df['category'].unique()
    
    # Analyze correlation with publication date
    time_data = category_scores_df.dropna(subset=['Publication date']).copy()
    
    # Convert publication dates to numeric values (days since a reference date)
    if not time_data.empty:
        time_data['date_numeric'] = [(d - pd.Timestamp('2022-01-01')).days for d in time_data['Publication date']]
    
    # Analyze correlation with compute
    compute_data = category_scores_df.dropna(subset=['Training compute (FLOP)']).copy()
    
    # Transform compute to log scale
    if not compute_data.empty:
        compute_data['log_compute'] = np.log10(compute_data['Training compute (FLOP)'])
    
    # Analyze by category
    for category in categories:
        # Time analysis
        if not time_data.empty:
            cat_time_data = time_data[time_data['category'] == category]
            
            if len(cat_time_data) > 1:  # Need at least 2 points for correlation
                x = cat_time_data['date_numeric'].values
                y = cat_time_data['mean_score'].values
                
                # Pearson correlation
                r_time, p_time = stats.pearsonr(x, y)
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared_time = r_value**2
                
                # Store results
                results.append({
                    'Category': category.replace('_', ' ').title(),
                    'Metric': 'Time (Publication Date)',
                    'Correlation': r_time,
                    'R-squared': r_squared_time,
                    'p-value': p_time,
                    'Slope': slope,
                    'Intercept': intercept
                })
        
        # Compute analysis
        if not compute_data.empty:
            cat_compute_data = compute_data[compute_data['category'] == category]
            
            if len(cat_compute_data) > 1:  # Need at least 2 points for correlation
                x = cat_compute_data['log_compute'].values  # Using log scale
                y = cat_compute_data['mean_score'].values
                
                # Pearson correlation
                r_compute, p_compute = stats.pearsonr(x, y)
                
                # Linear regression
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
                r_squared_compute = r_value**2
                
                # Store results
                results.append({
                    'Category': category.replace('_', ' ').title(),
                    'Metric': 'Log Training Compute (FLOP)',
                    'Correlation': r_compute,
                    'R-squared': r_squared_compute,
                    'p-value': p_compute,
                    'Slope': slope,
                    'Intercept': intercept
                })
    
    # Convert to DataFrame
    results_df = pd.DataFrame(results)
    
    # Sort by category and metric
    results_df = results_df.sort_values(['Category', 'Metric'])
    
    # Add overall correlations across all categories (pooled data)
    if not time_data.empty and len(time_data) > 1:
        x = time_data['date_numeric'].values
        y = time_data['mean_score'].values
        r_time, p_time = stats.pearsonr(x, y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared_time = r_value**2
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Category': 'ALL CATEGORIES',
            'Metric': 'Time (Publication Date)',
            'Correlation': r_time,
            'R-squared': r_squared_time,
            'p-value': p_time,
            'Slope': slope,
            'Intercept': intercept
        }])])
    
    if not compute_data.empty and len(compute_data) > 1:
        x = compute_data['log_compute'].values
        y = compute_data['mean_score'].values
        r_compute, p_compute = stats.pearsonr(x, y)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        r_squared_compute = r_value**2
        
        results_df = pd.concat([results_df, pd.DataFrame([{
            'Category': 'ALL CATEGORIES',
            'Metric': 'Log Training Compute (FLOP)',
            'Correlation': r_compute,
            'R-squared': r_squared_compute,
            'p-value': p_compute,
            'Slope': slope,
            'Intercept': intercept
        }])])
    
    # Format the numeric columns
    results_df['Correlation'] = results_df['Correlation'].map('{:.4f}'.format)
    results_df['R-squared'] = results_df['R-squared'].map('{:.4f}'.format)
    results_df['p-value'] = results_df['p-value'].map('{:.4f}'.format)
    results_df['Slope'] = results_df['Slope'].map('{:.6f}'.format)
    results_df['Intercept'] = results_df['Intercept'].map('{:.4f}'.format)
    
    return results_df

def create_alluvial_plot(all_exams, model_family, family_name):
    """
    Create an alluvial-style plot for model family performance evolution
    
    Parameters:
    -----------
    all_exams : DataFrame
        DataFrame containing all exam data
    model_family : dict
        Dictionary containing model family mapping
    family_name : str
        Name of the model family
    """
    # Get the model order from the family orders dictionary
    order = MODEL_FAMILY_ORDERS[family_name.lower()]
    
    # Sort by version
    keys = sorted(model_family.keys(), key=lambda k: order.get(k, 999))
    cols = keys
    names = [model_family[k] for k in cols]

    # Style parameters
    fs_title, fs_lab, fs_tick = 22, 18, 16
    lw_trend, lw_avg = 2.5, 6
    ms_avg = 12

    fig, ax = plt.subplots(figsize=(14, 10))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Plot each task
    for _, row in all_exams[cols].iterrows():
        vals = row.values
        diffs = np.diff(vals)
        if (diffs >= 0).all() and (diffs > 0).any():
            color = COLOR_PALETTE[4]    # consistently up
            label = 'Consistently ↑'
        elif (diffs <= 0).all() and (diffs < 0).any():
            color = COLOR_PALETTE[1]    # consistently down
            label = 'Consistently ↓'
        elif len(diffs) == 2 and diffs[0] > 0 and diffs[1] < 0:
            color = COLOR_PALETTE[2]    # up → down
            label = 'Up → Down'
        elif len(diffs) == 2 and diffs[0] < 0 and diffs[1] > 0:
            color = COLOR_PALETTE[0]    # down → up
            label = 'Down → Up'
        else:
            continue
        ax.plot(names, vals, color=color, alpha=0.4, linewidth=lw_trend, label='_nolegend_')

    # Average line in black
    avg = [all_exams[c].mean() for c in cols]
    ax.plot(names, avg,
            color='black',
            linewidth=lw_avg,
            marker='o', markersize=ms_avg,
            label='Average score')
    for i, v in enumerate(avg):
        ax.annotate(f'{v:.1f}', (names[i], v),
                    textcoords="offset points", xytext=(0, 12),
                    ha='center', fontsize=fs_lab, fontweight='bold')

    # Custom legend
    handles = [
        plt.Line2D([], [], color=COLOR_PALETTE[4], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[1], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[2], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[0], lw=lw_trend),
        plt.Line2D([], [], color='black', lw=lw_avg, marker='o', markersize=ms_avg)
    ]
    labels = [
        'Consistently ↑',
        'Consistently ↓',
        'Up → Down',
        'Down → Up',
        'Average score'
    ]
    ax.legend(handles, labels,
              loc='upper center', bbox_to_anchor=(0.5, -0.12),
              ncol=5, fontsize=fs_lab)

    ax.set_title(f'{family_name} Family Performance Evolution', fontsize=fs_title, pad=20)
    ax.set_ylabel('Score (0–100)', fontsize=fs_lab)
    ax.tick_params(axis='x', labelrotation=15, labelsize=fs_tick)
    ax.tick_params(axis='y', labelsize=fs_tick)
    ax.set_ylim(0, 100)
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    plt.tight_layout()
    return fig

def plot_family_with_hist(all_exams, model_family, family_name, bins=40):
    """
    Create a combined plot with histogram of changes and alluvial plot for a model family
    
    Parameters:
    -----------
    all_exams : DataFrame
        DataFrame containing all exam data
    model_family : dict
        Dictionary containing model family mapping
    family_name : str
        Name of the model family
    bins : int, optional
        Number of bins for the histogram, by default 40
    """
    # Get the model order from the family orders dictionary
    order = MODEL_FAMILY_ORDERS[family_name.lower()]
    
    cols = sorted(model_family.keys(), key=lambda k: order[k])
    names = [model_family[c] for c in cols]

    # Compute diffs and flatten
    diff_df = all_exams[cols].diff(axis=1).iloc[:, 1:]
    changes = diff_df.values.flatten()

    fig, (ax_hist, ax_allu) = plt.subplots(
        2, 1,
        figsize=(14, 12),
        gridspec_kw={'height_ratios': [1, 2], 'hspace': 0.3}
    )

    # Histogram with palette[3]
    ax_hist.hist(changes, bins=bins, color=COLOR_PALETTE[3], edgecolor='black', alpha=0.8)
    ax_hist.set_title(f'{family_name} Task Score Changes Distribution', fontsize=18)
    ax_hist.set_xlabel('Score Change (points)')
    ax_hist.set_ylabel('Count')
    ax_hist.grid(axis='y', linestyle='--', alpha=0.3)

    # Alluvial-style plot
    lw_trend, lw_avg = 2.5, 6
    ms_avg = 12

    for _, row in all_exams[cols].iterrows():
        vals = row.values
        diffs = np.diff(vals)
        if (diffs >= 0).all() and (diffs > 0).any():
            color = COLOR_PALETTE[4]
        elif (diffs <= 0).all() and (diffs < 0).any():
            color = COLOR_PALETTE[1]
        elif len(diffs) == 2 and diffs[0] > 0 and diffs[1] < 0:
            color = COLOR_PALETTE[2]
        elif len(diffs) == 2 and diffs[0] < 0 and diffs[1] > 0:
            color = COLOR_PALETTE[0]
        else:
            continue
        ax_allu.plot(names, vals, color=color, alpha=0.4, linewidth=lw_trend)

    # Average line
    avg = [all_exams[c].mean() for c in cols]
    ax_allu.plot(
        names, avg,
        color='black', linewidth=lw_avg,
        marker='o', markersize=ms_avg
    )
    for i, v in enumerate(avg):
        ax_allu.annotate(f'{v:.1f}', (names[i], v),
                         textcoords="offset points", xytext=(0, 12),
                         ha='center', fontsize=14, fontweight='bold')

    # Custom legend
    handles = [
        plt.Line2D([], [], color=COLOR_PALETTE[4], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[1], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[2], lw=lw_trend),
        plt.Line2D([], [], color=COLOR_PALETTE[0], lw=lw_trend),
        plt.Line2D([], [], color='black', lw=lw_avg, marker='o', markersize=ms_avg)
    ]
    labels = [
        'Consistently ↑',
        'Consistently ↓',
        'Up → Down',
        'Down → Up',
        'Average score'
    ]
    ax_allu.legend(
        handles, labels,
        loc='upper center',
        bbox_to_anchor=(0.5, -0.15),
        ncol=5,
        fontsize=14
    )

    ax_allu.set_title(f'{family_name} Family Performance Evolution', fontsize=20, pad=20)
    ax_allu.set_ylabel('Score (0–100)', fontsize=16)
    ax_allu.set_ylim(0, 100)
    ax_allu.tick_params(axis='x', rotation=15, labelsize=12)
    ax_allu.tick_params(axis='y', labelsize=12)
    ax_allu.grid(axis='y', linestyle='--', alpha=0.3)

    plt.tight_layout()
    return fig

def analyze_top_bottom_tasks(all_exams):
    """
    Analyze top and bottom tasks by model performance
    
    Parameters:
    -----------
    all_exams : DataFrame
        DataFrame containing all exam data
        
    Returns:
    --------
    tuple
        Tuple containing top10_overall, bottom10_overall, top10_avg_improve, bottom10_avg_decrease
    """
    # Get the latest model columns for each family
    mid_last = {
        'claude': (MODEL_FAMILY_VERSIONS['claude']['middle'], MODEL_FAMILY_VERSIONS['claude']['last']),
        'gemini': (MODEL_FAMILY_VERSIONS['gemini']['middle'], MODEL_FAMILY_VERSIONS['gemini']['last']),
        'gpt': (MODEL_FAMILY_VERSIONS['gpt']['middle'], MODEL_FAMILY_VERSIONS['gpt']['last']),
    }
    
    # Unpack into lists
    top_cols = [last for _, (_, last) in mid_last.items()]
    mid_cols = [mid for _, (mid, _) in mid_last.items()]
    
    # 1) Compute average latest score
    all_exams['avg_latest_score'] = all_exams[top_cols].mean(axis=1)
    
    # Pick overall top 10 & bottom 10 by that average
    top10_overall = all_exams[['task_id', 'task_description', 'avg_latest_score']] \
        .nlargest(10, 'avg_latest_score') \
        .assign(rank=lambda d: range(1, 11))
    bottom10_overall = all_exams[['task_id', 'task_description', 'avg_latest_score']] \
        .nsmallest(10, 'avg_latest_score') \
        .assign(rank=lambda d: range(1, 11))
    
    # 2) Compute per‐family improvements, then average them
    all_exams['imp_claude'] = (
        all_exams[mid_last['claude'][1]] - all_exams[mid_last['claude'][0]]
    )
    all_exams['imp_gemini'] = (
        all_exams[mid_last['gemini'][1]] - all_exams[mid_last['gemini'][0]]
    )
    all_exams['imp_gpt'] = (
        all_exams[mid_last['gpt'][1]] - all_exams[mid_last['gpt'][0]]
    )
    
    all_exams['avg_improvement'] = all_exams[
        ['imp_claude', 'imp_gemini', 'imp_gpt']
    ].mean(axis=1)
    
    # Pick top 10 improvements & bottom 10 decreases by that average delta
    top10_avg_improve = all_exams[['task_id', 'task_description', 'avg_improvement']] \
        .nlargest(10, 'avg_improvement') \
        .assign(rank=lambda d: range(1, 11))
    bottom10_avg_decrease = all_exams[['task_id', 'task_description', 'avg_improvement']] \
        .nsmallest(10, 'avg_improvement') \
        .assign(rank=lambda d: range(1, 11))
    
    return top10_overall, bottom10_overall, top10_avg_improve, bottom10_avg_decrease

def format_top_bottom_tables(top10_overall, bottom10_overall, top10_avg_improve, bottom10_avg_decrease, all_exams):
    """
    Format the top and bottom task tables for LaTeX
    """
    # Pull in the two meta fields
    meta = all_exams[['task_id', 'occupation', 'occupation_category']]
    
    # Custom human-readable map
    category_map = {
        'business_and_financial_operations': 'Business and Finance',
        'computer_and_mathematical': 'Computer and Mathematics',
        # Add more explicit overrides here if needed
    }
    
    tables = [
        (top10_overall, 'Top 10 Tasks by Average Latest Score', 'tab:top10_overall', 'avg_latest_score'),
        (bottom10_overall, 'Bottom 10 Tasks by Average Latest Score', 'tab:bottom10_overall', 'avg_latest_score'),
        (top10_avg_improve, 'Top 10 Tasks by Average Improvement', 'tab:top10_avg_improve', 'avg_improvement'),
        (bottom10_avg_decrease, 'Bottom 10 Tasks by Average Decrease', 'tab:bottom10_avg_decrease', 'avg_improvement'),
    ]
    
    # Desired column format
    col_fmt = 'c p{6.0cm} p{2.0cm} p{2.0cm} p{1.0cm}'
    
    formatted_tables = []
    
    for df, caption, label, valcol in tables:
        # Re-attach occupation info
        df_full = df.merge(meta, on='task_id', how='left')
        
        # Select & rename columns
        latex_df = df_full[[
            'rank',
            'task_description',
            'occupation',
            'occupation_category',
            valcol
        ]].rename(columns={
            'rank': 'Rank',
            'task_description': 'Task Description',
            'occupation': 'Occupation',
            'occupation_category': 'Broad Category',
            valcol: valcol.replace('_', ' ').title()
        })
        
        # Map & title-case broad category
        latex_df['Broad Category'] = (
            latex_df['Broad Category']
            .replace(category_map)
            .str.replace('_', ' ')
            .str.title()
        )
        
        # Generate the plain tabular body
        raw_tex = latex_df.to_latex(
            index=False,
            longtable=False,
            column_format=col_fmt,
            float_format="%.2f",
            escape=True
        ).splitlines()
        
        # Extract only the interior of the tabular (skip begin/end)
        body = "\n".join(raw_tex[1:-1])
        
        # Assemble complete table float
        table_tex = "\n".join([
            "\\begin{table}[ht]",
            "  \\tiny",
            "  \\centering",
            f"  \\begin{{tabular}}{{{col_fmt}}}",
            body,
            "  \\end{tabular}",
            f"  \\caption{{{caption}}}",
            f"  \\label{{{label}}}",
            "\\end{table}"
        ])
        
        formatted_tables.append(table_tex)
    
    return formatted_tables

# Helper function to get display name for a model
def get_display_name(model_name):
    """
    Get the display name for a model
    """
    score_col = MODEL_MAPPING.get(model_name)
    if score_col:
        return REVERSE_MAPPING.get(score_col, model_name)
    return model_name

def main():
    """
    Main function to run all analyses
    """
    # Load data
    all_exams = load_exam_data()
    exam_results = load_exam_results()
    
    # Replace all_exams with exam_results for consistency
    all_exams = exam_results
    
    # Fill NA values with 0 as mentioned in the requirements
    all_exams = all_exams.fillna(0)
    
    # Load model info
    df_model_info, df_model_benchmark = load_model_info()
    
    # Create model benchmark dataframe
    df_model_bench = create_model_benchmark_dataframe(df_model_info, df_model_benchmark)
    
    # Calculate scores by category
    category_scores_df, df_model_bench, avg_scores_df = calculate_scores_by_category(all_exams, df_model_bench)
    
    # Save category scores to csv
    category_scores_df.to_csv('../../scores_time_category.csv', index=False)
    
    # Plot category performance
    plot1 = plot_category_performance(all_exams, df_model_bench, category_scores_df)
    plt.savefig('../../results/figures/bar_plot_performance_by_category_na0_large_font.png', bbox_inches='tight')
    plt.show()
    
    # Plot performance vs time by category
    plot2 = plot_scatter_performance_vs_time_by_category(category_scores_df)
    plt.savefig('../../results/figures/scatter_performace_vs_time_by_occcategory_na0_large_font.png', bbox_inches='tight')
    plt.show()
    
    # # Plot performance vs compute by category
    # plot3 = plot_scatter_performance_vs_compute_by_category(category_scores_df)
    # plt.savefig('../../results/figures/scatter_performace_vs_compute_by_occcategory_na0.png', bbox_inches='tight')
    # plt.show()
    
    # # Analyze correlations
    # results_df = analyze_correlations(category_scores_df)
    # print("Correlation Analysis Results:")
    # print(results_df)
    
    # # Create alluvial plots for each model family
    # for family_name, model_family in [
    #     ("Claude", MODEL_FAMILIES["claude"]), 
    #     ("Gemini", MODEL_FAMILIES["gemini"]), 
    #     ("GPT", MODEL_FAMILIES["gpt"])
    # ]:
    #     # Simple alluvial plot
    #     fig = create_alluvial_plot(all_exams, model_family, family_name)
    #     plt.savefig(f'../../results/figures/{family_name.lower()}_evolution.png', dpi=300)
    #     plt.show()
        
    #     # Alluvial plot with histogram
    #     fig = plot_family_with_hist(all_exams, model_family, family_name)
    #     plt.savefig(f'../../results/figures/{family_name.lower()}_evolution_with_hist.png', dpi=300)
    #     plt.show()
    
    # # Analyze top and bottom tasks
    # # Call analyze_top_bottom_tasks with all_exams as an argument
    # top10_overall, bottom10_overall, top10_avg_improve, bottom10_avg_decrease = analyze_top_bottom_tasks(all_exams)
    
    # # Format the top and bottom task tables for LaTeX
    # formatted_tables = format_top_bottom_tables(
    #     top10_overall, bottom10_overall, top10_avg_improve, bottom10_avg_decrease, all_exams
    # )
    
    # # Print the formatted tables
    # for table in formatted_tables:
    #     print(table)
    #     print("\n" + "-"*80 + "\n")
    
    # print("Analysis complete!")


main()


