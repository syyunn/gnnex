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

buy_sell_edge = ('congressperson', 'buy-sell', 'ticker')

filtered_auc_rocs_train = {key: value for key, value in auc_rocs_train.items() if buy_sell_edge not in key}
filtered_accs_train = {key: value for key, value in accs_train.items() if buy_sell_edge not in key}

filtered_auc_rocs_test = {key: value for key, value in auc_rocs_test.items() if buy_sell_edge not in key}
filtered_accs_test = {key: value for key, value in accs_test.items() if buy_sell_edge not in key}

# Filter dictionaries for keys with 3 edges
accs_train_filtered = {key: value for key, value in filtered_accs_train.items() if len(key) >= 3}
auc_rocs_train_filtered = {key: value for key, value in filtered_auc_rocs_train.items() if len(key) >= 3}

accs_test_filtered = {key: value for key, value in filtered_accs_test.items() if len(key) >= 3}
auc_rocs_test_filtered = {key: value for key, value in filtered_auc_rocs_test.items() if len(key) >= 3}



# Create DataFrames with the filtered keys
accs_test_filtered_list = [{k: v[i] for k, v in accs_test_filtered.items()} for i in range(5)]
auc_rocs_test_filtered_list = [{k: v[i] for k, v in auc_rocs_test_filtered.items()} for i in range(5)]

accs_train_filtered_list = [{k: v[i] for k, v in accs_train_filtered.items()} for i in range(5)]
auc_rocs_train_filtered_list = [{k: v[i] for k, v in auc_rocs_train_filtered.items()} for i in range(5)]


all_edge_types = {('congressperson', 'assignment', 'committee'), ('ticker', 'classified', 'naics'), ('bill', 'assigned_to', 'committee'), ('ticker', 'lobbies_on', 'bill')}

# Create labels for the models showing the missing edge
edge_types_labels = []
for key in accs_test_filtered.keys():
    missing_edges = all_edge_types - set(key)
    missing_edge_str = ', '.join(sorted('-'.join(edge) for edge in missing_edges))
    if missing_edge_str == '':
        edge_types_labels.append('All Edge types')
    else:
        edge_types_labels.append(f"Missing: {missing_edge_str}")

accs_test_df = pd.concat([pd.DataFrame([d]) for d in accs_test_filtered_list], ignore_index=True)
auc_rocs_test_df = pd.concat([pd.DataFrame([d]) for d in auc_rocs_test_filtered_list], ignore_index=True)

accs_train_df = pd.concat([pd.DataFrame([d]) for d in accs_train_filtered_list], ignore_index=True)
auc_rocs_train_df = pd.concat([pd.DataFrame([d]) for d in auc_rocs_train_filtered_list], ignore_index=True)

# Compute means and standard deviations for train
accs_train_mean = accs_train_df.mean(axis=0)
accs_train_std = accs_train_df.std(axis=0)
auc_rocs_train_mean = auc_rocs_train_df.mean(axis=0)
auc_rocs_train_std = auc_rocs_train_df.std(axis=0)

# Compute means and standard deviations
accs_test_mean = accs_test_df.mean(axis=0)
accs_test_std = accs_test_df.std(axis=0)
auc_rocs_test_mean = auc_rocs_test_df.mean(axis=0)
auc_rocs_test_std = auc_rocs_test_df.std(axis=0)

# Plot train and test accuracies with error bars
x = range(len(accs_test_mean))
fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size
width = 0.3  # Width of each bar

# Train Accuracy
ax.errorbar([i - width/2 for i in x], accs_train_mean, yerr=accs_train_std, fmt='o-', label='Train Accuracy', capsize=3)

# Test Accuracy
ax.errorbar([i + width/2 for i in x], accs_test_mean, yerr=accs_test_std, fmt='o-', label='Test Accuracy', capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(edge_types_labels, rotation=45, fontsize=12)  # Update rotation and fontsize here
ax.set_xlabel("Edge Types")
ax.set_ylabel("Accuracy")
ax.set_title("Train and Test Accuracy with Error Bars")
ax.legend()
fig.tight_layout()
plt.show()

# Plot train and test AUC-ROC with error bars
x = range(len(auc_rocs_test_mean))
fig, ax = plt.subplots(figsize=(10, 6))  # Increase figure size

# Train AUC-ROC
ax.errorbar([i - width/2 for i in x], auc_rocs_train_mean, yerr=auc_rocs_train_std, fmt='o-', label='Train AUC-ROC', capsize=3)

# Test AUC-ROC
ax.errorbar([i + width/2 for i in x], auc_rocs_test_mean, yerr=auc_rocs_test_std, fmt='o-', label='Test AUC-ROC', capsize=3)

ax.set_xticks(x)
ax.set_xticklabels(edge_types_labels, rotation=45, fontsize=12)  # Update rotation and fontsize here
ax.set_xlabel("Missing Edge Type")
ax.set_ylabel("AUC-ROC")
ax.set_title("Train and Test AUC-ROC with Error Bars")
ax.legend()
fig.tight_layout()
plt.show()
