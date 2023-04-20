import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Read the CSV file
df = pd.read_csv('log_results.csv')

# Find the highest values among epochs in each group
df_max_epoch = df.groupby(['edge_type_removed', 'train_test', 'fold']).max('epoch').reset_index()


# Set the figure size (width, height) in inches
plt.figure(figsize=(15, 6))  # Adjust the width value as needed

# Plot the point plot with error bars
sns.pointplot(data=df_max_epoch, x='edge_type_removed', y='auc_roc', hue='train_test', ci='sd', capsize=0.1, dodge=True, linestyles='')
plt.xlabel('Edge Type Removed')
plt.ylabel('AUC-ROC')
plt.title('AUC-ROC by Edge Type Removed with Error Bars')
plt.legend(title='Train/Test')
plt.show()
