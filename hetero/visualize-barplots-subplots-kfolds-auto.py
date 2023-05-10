import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# read k_folds.pkl
import pickle
with open('k_folds.pkl', 'rb') as f:
    k_folds = pickle.load(f)

def sort_keys(dictionary):
    sorted_dict = {}
    for key, value in dictionary.items():
        sorted_key = tuple(sorted(key))
        sorted_dict[sorted_key] = value
    return sorted_dict

# Sort the keys for all variables
auc_rocs_train = sort_keys(k_folds[0])
accs_train = sort_keys(k_folds[1])
auc_rocs_test = sort_keys(k_folds[2])
accs_test = sort_keys(k_folds[3])

keys = list(auc_rocs_train.keys())
for key in keys:
    if len(key) == 5:
        edge_types = key

num_edge_types = len(edge_types)

# check
fold_counts = {i: 0 for i in range(5)}

for key in keys:
    for fold in range(5):
        if fold in auc_rocs_train[key]:
            fold_counts[fold] += 1

print("Occurrences of each fold:", fold_counts)

from math import factorial
def get_shapley_values(res, fold):
    shapley_values = {}
    n = len(edge_types)

    for ei in edge_types:
        shapley_value = 0
        for key in keys:
            if ei not in key:
                S = set(key)
                S_with_ei = S | {ei}
                size_S = len(S)

                # Convert sets back to sorted tuples
                S_tuple = tuple(sorted(S))
                S_with_ei_tuple = tuple(sorted(S_with_ei))
                
                weight = (factorial(size_S) * factorial(n - size_S - 1)) / factorial(n)
                marginal_contribution = res[S_with_ei_tuple].get(fold, 0) - res[S_tuple].get(fold, 0)

                shapley_value += weight * marginal_contribution

        shapley_values[ei] = shapley_value

    return shapley_values

# Compute Shapley values for each fold and score type
shapley_values_dict = {}

score_types = {
    'auc_rocs_train': auc_rocs_train,
    'accs_train': accs_train,
    'auc_rocs_test': auc_rocs_test,
    'accs_test': accs_test
}

for score_type, score_data in score_types.items():
    shapley_values_folds = {}
    for fold in range(5):
        shapley_values_folds[fold] = get_shapley_values(score_data, fold)
    shapley_values_dict[score_type] = shapley_values_folds

print("Shapley values for each fold and score type:", shapley_values_dict)

import pandas as pd

# Initialize a list to store the data for the DataFrame
shapley_data = []

for score_type, shapley_values_folds in shapley_values_dict.items():
    for fold, shapley_values in shapley_values_folds.items():
        for edge_type, shapley_value in shapley_values.items():
            shapley_data.append({
                'score_type': score_type,
                'fold': fold,
                'edge_type': edge_type,
                'shapley_value': shapley_value
            })

# Create the DataFrame
shapley_df = pd.DataFrame(shapley_data)
print(shapley_df.head())

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Set the plot style
plt.style.use("seaborn-whitegrid")

# Group the data by 'score_type' and 'edge_type'
grouped_df = shapley_df.groupby(['score_type', 'edge_type']).agg(
    shapley_mean=('shapley_value', 'mean'),
    shapley_std=('shapley_value', 'std')
).reset_index()

# Print the grouped data
print(grouped_df)

import matplotlib.pyplot as plt
import numpy as np

# Set the plot style
plt.style.use("seaborn-whitegrid")

# Create a figure and axis
fig, ax = plt.subplots()

# Set the width of the bars
bar_width = 0.15

# Set the positions for the bars
positions = np.arange(len(grouped_df['edge_type'].unique()))

# Loop through each score type and plot the bars with error bars
for i, score_type in enumerate(grouped_df['score_type'].unique()):
    # Filter the data for the current score type
    score_data = grouped_df[grouped_df['score_type'] == score_type]
    
    # Plot the bars with error bars
    ax.bar(positions + i * bar_width,
           score_data['shapley_mean'],
           yerr=score_data['shapley_std'],
           width=bar_width,
           label=score_type)

# Set the x-ticks and x-tick labels
ax.set_xticks(positions + bar_width * (len(grouped_df['score_type'].unique()) - 1) / 2)
ax.set_xticklabels(grouped_df['edge_type'].unique(), rotation=45, ha='right')

# Set the labels and title
ax.set_ylabel("Shapley Value")
ax.set_xlabel("Edge Type")
ax.set_title("Shapley Values Comparison Across Different Score Types")

# Add the legend
ax.legend(title="Score Type", title_fontsize="13", loc="upper left", bbox_to_anchor=(1, 1))

# Show the plot
plt.tight_layout()
plt.show()
