import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

try:
    DATA_DIR = str(pathlib.Path(__file__).parent / "data")
    PLOT_DIR = str(pathlib.Path(__file__).parent / "plots")
except:
    DATA_DIR = "./scenegraph/analysis/data"
    PLOT_DIR = "./scenegraph/analysis/plots"


## QA EXPERIMENT ##
data_qa = pd.read_csv(DATA_DIR + "/data_qa.csv")
data_qa_corrected = pd.read_csv(DATA_DIR + "/data_qa_corrected.csv")

# add column corrected
data_qa["corrected"] = False
data_qa_corrected["corrected"] = True

# Combine the datasets
combined_data = pd.concat([data_qa, data_qa_corrected], ignore_index=True)

# Define the outcome based on the conditions provided
combined_data['outcome'] = combined_data.apply(
    lambda row: 'failed parsing' if pd.isna(row['pred_answer']) else 
                'correct' if row['pred_answer'] == row['answer'] else 
                'incorrect', axis=1)

# Define consistent colors for outcomes
colors = {
    'correct': 'green',
    'incorrect': 'red',
    'failed parsing': 'blue'
}

# First group: Filter where corrected is False, aggregate by difficulty
group1 = combined_data[combined_data['corrected'] == False].groupby(['difficulty', 'outcome']).size().unstack(fill_value=0)
group1_percent = group1.div(group1.sum(axis=1), axis=0) * 100

# Second group: Filter where corrected is False, aggregate by type
group2 = combined_data[combined_data['corrected'] == False].groupby(['type', 'outcome']).size().unstack(fill_value=0)
group2_percent = group2.div(group2.sum(axis=1), axis=0) * 100

# Third group: Aggregate by corrected
group3 = combined_data.groupby(['corrected', 'outcome']).size().unstack(fill_value=0)
group3_percent = group3.div(group3.sum(axis=1), axis=0) * 100

# Create three subplots with shared y-axis
fig, axes = plt.subplots(1, 3, figsize=(12, 5), sharey=True)

# First subplot: Aggregated by difficulty
group1_percent.plot(kind='bar', stacked=True, ax=axes[0], color=[colors[outcome] for outcome in group1.columns], legend=False)
axes[0].set_title('Aggregated by Difficulty')
axes[0].set_xlabel('Difficulty')
axes[0].set_ylabel('Percentage (%)')
axes[0].tick_params(axis='x', rotation=0)  # Set x-axis labels to horizontal

# Second subplot: Aggregated by type
group2_percent.plot(kind='bar', stacked=True, ax=axes[1], color=[colors[outcome] for outcome in group2.columns], legend=False)
axes[1].set_title('Aggregated by Type')
axes[1].set_xlabel('Type')
axes[1].tick_params(axis='x', rotation=0)  # Set x-axis labels to horizontal

# Third subplot: Aggregated by corrected status
group3_percent.plot(kind='bar', stacked=True, ax=axes[2], color=[colors[outcome] for outcome in group3.columns], legend=True)
axes[2].set_title('Aggregated by Corrected Status')
axes[2].set_xlabel('Corrected')
axes[2].tick_params(axis='x', rotation=0)  # Set x-axis labels to horizontal

# Set a single shared legend
handles = [plt.Line2D([0], [0], color=colors[outcome], lw=4) for outcome in ['correct', 'incorrect', 'failed parsing']]
labels = ['Correct', 'Incorrect', 'Failed Parsing']
#handles, labels = axes[0].get_legend_handles_labels()
#breakpoint()
# Add the figure-level legend
fig.legend(handles, labels, title='Outcome', loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
# Set a tight layout for better readability
#plt.tight_layout()

# save
plt.savefig(PLOT_DIR + "/qa.png")

# Show the plot
plt.show()


## MAIN EXPERIMENT ##
data = pd.read_csv(DATA_DIR + "/data_main.csv")

data.head()

# simplfy env columns# Example translation dictionary
translation_dict = {
    'MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0': 'CleaningUpTheKitchenOnly',
    "MiniGrid-CollectMisplacedItems-16x16-N2-v0": "CollectMisplacedItems",
    "MiniGrid-OrganizingFileCabinet-16x16-N2-v0": "OrganizingFileCabinet"
}

# Applying the translation to the 'env' column using the dictionary
data['env'] = data['env'].replace(translation_dict)

# ANALYZE FAILURE CONDITIONS

# Filter the dataset to include only rows where the final output is not 'Success'
non_success_data = data[data['final_output'] != 'Success']

# Recheck the distribution of final outputs in the non-success data
non_success_final_output_counts = non_success_data['final_output'].value_counts()

# Plot the pie chart for non-success final outputs
plt.figure(figsize=(7,7))
plt.pie(non_success_final_output_counts, labels=non_success_final_output_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Reasons for Task Abortion')
plt.axis('equal')
# save
plt.savefig(PLOT_DIR + "/failure_reasons.png")
plt.show()


# ANALYZE PASSING CONDS PER TASK

# First, group the data by 'env', 'prompt_difficulty', and 'success_conds'
# Calculate standard deviation for passing conditions
grouped_data_with_std = data.groupby(['title', 'prompt_difficulty']).agg(
    total_success_conds=('success_conds', 'mean'),
    total_passing_conds=('passing_conds', 'mean'),
    passing_conds_std=('passing_conds', 'std')  # Standard deviation for error bars
).reset_index()

# Plotting the data with error bars
plt.figure(figsize=(14, 8))

# Slimmer bars with a bit of space between them
bar_width = 0.1
positions = [-bar_width * 1.5, 0, bar_width * 1.5]

# Plotting the bars for each environment and difficulty
titles = grouped_data_with_std['title'].unique()

new_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}  # Blue, Orange, Green
for i, title in enumerate(titles):
    # Filter the data for the current environment
    title_data = grouped_data_with_std[grouped_data_with_std['title'] == title]
    
    # Plot bars for each difficulty level, tightly grouped around the current environment
    for j, row in title_data.iterrows():
        difficulty_idx = row['prompt_difficulty'] - 1  # Offset by difficulty
        plt.bar(i + positions[difficulty_idx], row['total_success_conds'], width=bar_width, color='lightblue', label='Success Conditions' if i == 0 and difficulty_idx == 0 else "")
        plt.bar(i + positions[difficulty_idx], row['total_passing_conds'], width=bar_width, color=new_colors[row['prompt_difficulty']],
                yerr=row['passing_conds_std'], capsize=5)  # Adding error bars for passing conditions

# Customize the plot
plt.title('Success conditions and passed success conditions onditions by task and difficulty')
plt.xlabel('Task')
plt.ylabel('Number of success conditions')
plt.xticks(range(len(titles)), titles, rotation=45, ha='right')  # Slanting the labels for easier reading
plt.grid(True, axis='y')

# Custom legend for difficulty levels with new color palette
custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for color in new_colors.values()]
plt.legend(custom_legend, [f'Difficulty {lvl}' for lvl in new_colors.keys()], title='High-level task difficulty:')

plt.tight_layout()
plt.savefig(PLOT_DIR + "/success_conditions.png")
plt.show()

# ANALYZE STEPS UNTIL FAILURE PER ENV, DIFFICULTY LEVEL
# Now, group the data by 'env' and 'prompt_difficulty' to calculate the steps until failure
grouped_steps_env = non_success_data.groupby('env').agg(
    steps_until_failure_mean=('num_steps_completed', 'mean'),
    steps_until_failure_std=('num_steps_completed', 'std')
).reset_index()

grouped_steps_difficulty = non_success_data.groupby('prompt_difficulty').agg(
    steps_until_failure_mean=('num_steps_completed', 'mean'),
    steps_until_failure_std=('num_steps_completed', 'std')
).reset_index()

# Convert prompt difficulty to string for easier categorization
grouped_steps_difficulty['category'] = grouped_steps_difficulty['prompt_difficulty'].astype(str) + " (Difficulty)"
grouped_steps_env['category'] = grouped_steps_env['env'] + " (Environment)"

# Concatenate both dataframes for unified plotting
combined_data = pd.concat([grouped_steps_env[['category', 'steps_until_failure_mean', 'steps_until_failure_std']],
                           grouped_steps_difficulty[['category', 'steps_until_failure_mean', 'steps_until_failure_std']]])

# Define different colors for environment and difficulty bars
colors = ['#1f77b4' if 'Environment' in cat else '#ff7f0e' for cat in combined_data['category']]

# Plotting the combined data with slimmer bars and tighter grouping
plt.figure(figsize=(14, 8))

# Plot bars with error bars (whiskers)
bar_width = 0.4
plt.bar(combined_data['category'], combined_data['steps_until_failure_mean'], 
        yerr=combined_data['steps_until_failure_std'], capsize=5, color=colors, width=bar_width)

# Customize the plot
plt.title('Steps until Failure grouped by Environment, Difficulty')
plt.xlabel('Category')
plt.ylabel('Steps Until Failure')
plt.xticks(rotation=45, ha='right')  # Slant the labels for easier readability
plt.grid(True, axis='y')

plt.tight_layout()
plt.savefig(PLOT_DIR + "/steps_to_failure.png")
plt.show()
