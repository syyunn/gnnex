import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('log_results.csv')

# Find the highest values among epochs in each group
df_max_epoch = df.groupby(['edge_type_removed', 'train_test', 'fold']).max('epoch').reset_index()

# Set the figure size (width, height) in inches
plt.figure(figsize=(15, 9))  # Adjust the width and height values as needed

# Create three separate subplots
plt.subplot(3, 1, 1)
sns.pointplot(data=df_max_epoch[df_max_epoch['edge_type_removed'].str.count('&') == 0], x='edge_type_removed', y='auc_roc', hue='train_test', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Edge Type Removed')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC for No "&" in Edge Type Removed')
plt.legend(title='Train/Test')

plt.subplot(3, 1, 2)
sns.pointplot(data=df_max_epoch[df_max_epoch['edge_type_removed'].str.count('&') == 1], x='edge_type_removed', y='auc_roc', hue='train_test', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Edge Type Removed')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC for One "&" in Edge Type Removed')
plt.legend(title='Train/Test', loc='center left', bbox_to_anchor=(1, 0.5))

plt.subplot(3, 1, 3)
sns.pointplot(data=df_max_epoch[df_max_epoch['edge_type_removed'].str.count('&') == 2], x='edge_type_removed', y='auc_roc', hue='train_test', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Edge Type Removed')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC for Two "&"s in Edge Type Removed')
plt.legend(title='Train/Test', loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust the layout to prevent overlapping titles
plt.subplots_adjust(hspace=0.5)

# Show the plot
plt.show()
