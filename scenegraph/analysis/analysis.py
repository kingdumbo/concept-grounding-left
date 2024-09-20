import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pathlib

try:
    DATA_DIR = str(pathlib.Path(__file__).parent / "data")
except:
    DATA_DIR = "./scenegraph/analysis/data"

## QA EXPERIMENT ##
data = pd.read_csv(DATA_DIR + "/data_qa.csv")

# Modify the 'outcome' column logic to handle NaN values in 'pred_answer'
data['outcome'] = data.apply(lambda row: 'failed' if pd.isna(row['pred_answer']) 
                             else 'correct' if row['pred_answer'] == row['answer'] 
                             else 'incorrect', axis=1)

# First, we aggregate the data by 'difficulty' and 'outcome'
difficulty_agg = data.groupby(['difficulty', 'outcome']).size().unstack(fill_value=0)

# Then, aggregate the data by 'type' and 'outcome'
type_agg = data.groupby(['type', 'outcome']).size().unstack(fill_value=0)

# Combine the data into one DataFrame, treating both 'difficulty' and 'type' as categories
combined_agg = pd.concat([difficulty_agg, type_agg], keys=['Difficulty', 'Type']).reset_index(level=0).rename(columns={'level_0': 'Category'})

# Modify the x-axis ticks to slant and append labels to 'difficulty' and 'type'
combined_agg.index = combined_agg.index.map(lambda x: f"{x} (difficulty)" if combined_agg.loc[x, 'Category'] == 'Difficulty' else f"{x} (type)")

# Create a single plot with the bars for 'Type' first and 'Difficulty' next, with the new labels
fig, ax = plt.subplots(figsize=(10, 6))

# Plot the combined data as a stacked bar chart
combined_agg.plot(kind='bar', stacked=True, ax=ax)

# Slant the ticks on the x-axis
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')

# Adjust labels and title
ax.set_title('Outcome Aggregated by Difficulty and Type')
ax.set_ylabel('Count')
ax.set_xlabel('Categories (Type and Difficulty)')

# Adjust the layout and show the plot
plt.tight_layout()
plt.show()

## MAIN EXPERIMENT ##
data = pd.read_csv(DATA_DIR + "/data_main.csv")

data.head()

# simplfy env columns# Example translation dictionary
translation_dict = {
    'MiniGrid-CleaningUpTheKitchenOnly-16x16-N2-v0': 'Kitchen',
    'MiniGrid-AnotherEnvironment-16x16-N2-v0': 'Another Env',
    # Add more translations as needed
}

# Applying the translation to the 'env' column using the dictionary
data['env'] = data['env'].replace(translation_dict)

# ANALYZE FAILURE CONDITIONS

# Filter the dataset to include only rows where the final output is not 'Success'
non_success_data = data[data['final_output'] != 'Success']

# Recheck the distribution of final outputs in the non-success data
non_success_final_output_counts = non_success_data['final_output'].value_counts()

# Plot the pie chart for non-success final outputs
plt.figure(figsize=(8, 8))
plt.pie(non_success_final_output_counts, labels=non_success_final_output_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Reasons for Task Abortion')
plt.axis('equal')
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
plt.title('Success Conditions and Passing Conditions by Task and Difficulty')
plt.xlabel('Task')
plt.ylabel('Number of Conditions')
plt.xticks(range(len(titles)), titles, rotation=45, ha='right')  # Slanting the labels for easier reading
plt.grid(True, axis='y')

# Custom legend for difficulty levels with new color palette
custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for color in new_colors.values()]
plt.legend(custom_legend, [f'Difficulty {lvl}' for lvl in new_colors.keys()], title='Passing Conditions Difficulty')

plt.tight_layout()
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
plt.show()
