import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('log_results.csv')

# Find the highest values among epochs in each group
df_max_epoch = df.groupby(['edge_type_removed', 'train_test', 'fold']).max('epoch').reset_index()

# Split the data into two groups based on edge_type_removed
df_buy_sell = df_max_epoch[df_max_epoch['edge_type_removed'].str.contains('buy-sell &')]
df_other = df_max_epoch[~df_max_epoch['edge_type_removed'].str.contains('buy-sell &')]

# Set the figure size (width, height) in inches
plt.figure(figsize=(15, 6))  # Adjust the width value as needed

# Create the first subplot
plt.subplot(1, 2, 1)
sns.pointplot(data=df_buy_sell, x='train_test', y='auc_roc', hue='fold', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Train/Test')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC for "buy-sell &" Edge Type Removed')
plt.legend(title='Fold')

# Create the second subplot
plt.subplot(1, 2, 2)
sns.pointplot(data=df_other, x='train_test', y='auc_roc', hue='edge_type_removed', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Train/Test')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC for Other Edge Types Removed')
plt.legend(title='Edge Type Removed', loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust the layout to prevent overlapping titles
plt.subplots_adjust(wspace=0.5)

# Show the plot
plt.show()
