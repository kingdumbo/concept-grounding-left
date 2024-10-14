import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pathlib

try:
    DATA_DIR = str(pathlib.Path(__file__).parent / "data")
    PLOT_DIR = str(pathlib.Path(__file__).parent / "plots")
except:
    DATA_DIR = "./scenegraph/analysis/data"
    PLOT_DIR = "./scenegraph/analysis/plots"

# filenames
qa = "/gpt-4o_data_qa.csv"
qa_corrected = "/gpt-4o_data_qa_corrected.csv"
main = "/gpt-4o_data_main.csv"
prefix = "4o"

## QA EXPERIMENT ##
data_qa = pd.read_csv(DATA_DIR + qa)
data_qa_corrected = pd.read_csv(DATA_DIR + qa_corrected)

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

# First group: Filter where corrected is True, aggregate by difficulty
group1 = combined_data[combined_data['corrected'] == False].groupby(['difficulty', 'outcome']).size().unstack(fill_value=0)
group1_percent = group1.div(group1.sum(axis=1), axis=0) * 100

# Second group: Filter where corrected is True, aggregate by type
group2 = combined_data[combined_data['corrected'] == False].groupby(['type', 'outcome']).size().unstack(fill_value=0)
group2_percent = group2.div(group2.sum(axis=1), axis=0) * 100

# Third group: Aggregate by corrected
group3 = combined_data.groupby(['corrected', 'outcome']).size().unstack(fill_value=0)
group3_percent = group3.div(group3.sum(axis=1), axis=0) * 100

# First plot: Two subplots side by side (group1 and group2)
fig, axes = plt.subplots(1, 2, figsize=(7, 4), sharey=True)

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

# Retrieve handles and labels from the plot directly
handles, labels = axes[1].get_legend_handles_labels()

# Set a legend for this plot
axes[1].legend(handles, labels, title='Outcome', loc='lower right', ncol=1)

# Adjust layout and show the first plot
plt.tight_layout()
plt.savefig(PLOT_DIR + f"/{prefix}_qa_type_difficulty.png")
plt.show()

# Second plot: One subplot (group3)
fig, ax = plt.subplots(figsize=(4, 4))

# Third subplot: Aggregated by corrected status
group3_percent.plot(kind='bar', stacked=True, ax=ax, color=[colors[outcome] for outcome in group3.columns])
ax.set_title('Aggregated by Corrected Status')
ax.set_xlabel('Corrected')
ax.set_ylabel('Percentage (%)')
ax.tick_params(axis='x', rotation=0)  # Set x-axis labels to horizontal

# Retrieve handles and labels from the plot directly
handles, labels = ax.get_legend_handles_labels()

# Set a legend for this plot
ax.legend(handles, labels, title='Outcome', loc='lower right', ncol=1)

# Adjust layout and show the second plot
plt.tight_layout()
plt.savefig(PLOT_DIR + f"/{prefix}_qa_correction.png")
plt.show()


## MAIN EXPERIMENT ##
data = pd.read_csv(DATA_DIR + main)

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

# Get unique environments from the dataset
environments = non_success_data['env'].unique()

# Prepare the figure
plt.figure(figsize=(20, 7))  # Adjust the figure size for three subplots

# Set the overall figure title
plt.suptitle("Reasons for Task Abortion per Environment", fontsize=16, y=0.85)

# Collect unique final outputs to ensure consistency across pie charts
unique_final_outputs = sorted(non_success_data['final_output'].unique())

# Define colors for consistency across all pies using a colormap
color_map = plt.get_cmap("tab20")
colors = {final_output: color_map(i / len(unique_final_outputs)) for i, final_output in enumerate(unique_final_outputs)}

# Custom autopct function to filter out 0.0% labels
def autopct_filter_zero(pct):
    return ('%1.1f%%' % pct) if pct > 0 else ''  # Only show percentages greater than 0

for i, env in enumerate(environments, 1):
    # Filter data for each environment
    env_data = non_success_data[non_success_data['env'] == env]
    
    # Count the occurrences of final outputs for this environment
    env_final_output_counts = env_data['final_output'].value_counts()

    # Ensure that the colors correspond to the same final_output across all pie charts
    env_final_output_counts = env_final_output_counts.reindex(unique_final_outputs, fill_value=0)
    
    # Create a subplot for each pie chart
    plt.subplot(1, 3, i)
    plt.pie(env_final_output_counts, labels=None, 
            colors=[colors[fo] for fo in unique_final_outputs],  # Ensure same order of colors
            autopct=autopct_filter_zero, startangle=90, labeldistance=2)
    plt.title(f'{env}', fontsize=12, y=0.8)  # Set environment as the subplot title
    plt.axis('equal')

# Create the global legend using the consistent colors
patches = [mpatches.Patch(color=colors[label], label=label) for label in unique_final_outputs]
plt.legend(handles=patches, title="Final Output", bbox_to_anchor=(1.05, 0.5), loc="center left", fontsize=10)

# Adjust layout to prevent overlap
plt.tight_layout(rect=[0, 0, 0.85, 0.95])  # Adjust for legend placement and title

# Save the plot
plt.savefig(PLOT_DIR + f"/{prefix}_failure_reasons_per_env_with_legend.png")
plt.show()

# ANALYZE PASSING CONDS PER TASK

# Group the data by 'env', 'title', and 'prompt_difficulty'
grouped_data_with_std = data.groupby(['env', 'title', 'prompt_difficulty']).agg(
    total_success_conds=('success_conds', 'mean'),
    total_passing_conds=('passing_conds', 'mean'),
    # passing_conds_std=('passing_conds', 'std')  # Uncomment if you have std data for error bars
).reset_index()

# Plotting the data with error bars
plt.figure(figsize=(14, 8))

# Slimmer bars with a bit of space between them
bar_width = 0.1
positions = [-bar_width * 1.5, 0, bar_width * 1.5]  # Offset for difficulty levels

# Get unique environments and titles within each environment
environments = grouped_data_with_std['env'].unique()

# Create x-tick positions for each environment and title
xticks_positions = []
xticks_labels = []
env_positions = []  # Store the position of each environment for labeling
offset = 0  # To manage x-tick placement

new_colors = {1: '#1f77b4', 2: '#ff7f0e', 3: '#2ca02c'}  # Blue, Orange, Green

for env in environments:
    # Filter the data for the current environment
    env_data = grouped_data_with_std[grouped_data_with_std['env'] == env]
    titles = env_data['title'].unique()

    # Store the position for the environment label
    env_positions.append((offset + (len(titles) - 1) / 2))  # Center position for the environment label

    for i, title in enumerate(titles):
        # Append x-tick label for each title within the environment
        xticks_positions.append(offset + i)
        xticks_labels.append(f'{title}')  # Only the title for the x-tick label

        # Filter the data for the current title within the environment
        title_data = env_data[env_data['title'] == title]

        # Plot bars for each difficulty level, grouped around the current environment-title pair
        for j, row in title_data.iterrows():
            difficulty_idx = row['prompt_difficulty'] - 1  # Offset by difficulty
            plt.bar(offset + i + positions[difficulty_idx], row['total_success_conds'], width=bar_width, color='lightblue', label='Success Conditions' if offset == 0 and difficulty_idx == 0 else "")
            plt.bar(offset + i + positions[difficulty_idx], row['total_passing_conds'], width=bar_width, color=new_colors[row['prompt_difficulty']],
                    # yerr=row['passing_conds_std'],  # Add this back if you have std data
                    capsize=5)  # Adding error bars for passing conditions

    offset += len(titles) + 1  # Add space between different environments

# Customize the plot
plt.title('Completed success conditions by task and prompt difficulty')
plt.xlabel('Task')
plt.ylabel('Number of success conditions')
plt.xticks(xticks_positions, xticks_labels, rotation=45, ha='right')  # Slanting the labels for easier reading
plt.grid(True, axis='y')

# Custom legend for difficulty levels with new color palette
custom_legend = [plt.Line2D([0], [0], color=color, lw=4) for color in new_colors.values()]
plt.legend(custom_legend, [f'Difficulty {lvl}' for lvl in new_colors.keys()], title='High-level task difficulty:')

# Add environment labels above the grouped tasks
for env, pos in zip(environments, env_positions):
    plt.text(pos, plt.gca().get_ylim()[1] - 2, env, ha='center', fontsize=10)

# Optionally, add vertical lines to separate the environments
for pos in env_positions:
    #plt.axvline(x=pos + (bar_width * 2), color='gray', linestyle='--')
    pass

plt.tight_layout()
plt.savefig(PLOT_DIR + f"/{prefix}_success_conditions_by_env.png")
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
grouped_steps_difficulty['category'] = grouped_steps_difficulty['prompt_difficulty'].astype(str)
grouped_steps_env['category'] = grouped_steps_env['env'] 

# Define different colors for environment and difficulty bars
colors_env = ['#1f77b4'] * len(grouped_steps_env)  # Blue for environment
colors_difficulty = ['#ff7f0e'] * len(grouped_steps_difficulty)  # Orange for difficulty

# Create subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 4), sharey=True)  # Share the y-axis

# Plot for environment
ax1.bar(grouped_steps_env['category'], grouped_steps_env['steps_until_failure_mean'], 
        yerr=grouped_steps_env['steps_until_failure_std'], capsize=5, color=colors_env, width=0.4)
ax1.set_title('Steps until Failure by Environment')
ax1.set_xlabel('Environment')
ax1.set_ylabel('Steps Until Failure')

# Adjust tick position to align with the end of the label
ax1.set_xticks(ax1.get_xticks())  # Ensure ticks are set properly
ax1.tick_params(axis='x', rotation=45)
for tick in ax1.get_xticklabels():
    tick.set_ha('right')  # Align the labels so the end of the text is at the tick

ax1.grid(True, axis='y')

# Plot for difficulty
ax2.bar(grouped_steps_difficulty['category'], grouped_steps_difficulty['steps_until_failure_mean'], 
        yerr=grouped_steps_difficulty['steps_until_failure_std'], capsize=5, color=colors_difficulty, width=0.4)
ax2.set_title('Steps until Failure by Difficulty')
ax2.set_xlabel('Difficulty')

# Adjust tick position to align with the end of the label
ax2.set_xticks(ax2.get_xticks())  # Ensure ticks are set properly
ax2.tick_params(axis='x', rotation=0)
for tick in ax2.get_xticklabels():
    tick.set_ha('center')  # Align the labels so the end of the text is at the tick

ax2.grid(True, axis='y')

# Adjust layout to avoid overlap
plt.tight_layout()

# Save the plot
plt.savefig(PLOT_DIR + f"/{prefix}_steps_to_failure_by_env_and_difficulty.png")
plt.show()

