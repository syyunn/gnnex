import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# Number of data points
n = 100000

# Define the number of bins for Z
num_bins = 100

# Generate confounder Z
Z = np.random.normal(loc=0, scale=1, size=n)

# Define the logistic function to model the probability of treatment
def logistic(z):
    return 1 / (1 + np.exp(-z))

# Generate binary treatment W based on Z using the logistic function
prob_W = logistic(0.5 * Z)
W = np.random.binomial(n=1, p=prob_W)

# Define potential outcomes Y(1) and Y(0) based on Z and random noise
Y1 = 3 * Z + np.random.normal(loc=0, scale=1, size=n)
Y0 = 2 * Z + np.random.normal(loc=0, scale=1, size=n)

# Create a DataFrame to store the generated data
data = pd.DataFrame({'W': W, 'Z': Z, 'Y1': Y1, 'Y0': Y0})


# Create bins for Z
data['Z_bins'] = pd.cut(data['Z'], bins=num_bins, labels=False)

# Group the data by the bins of Z and compute the correlation between W and Y1 and between W and Y0 within each bin
correlations_Y1 = data.groupby('Z_bins').apply(lambda x: np.corrcoef(x['W'], x['Y1'])[0, 1])
correlations_Y0 = data.groupby('Z_bins').apply(lambda x: np.corrcoef(x['W'], x['Y0'])[0, 1])

# Print the correlations within each bin
print("Correlation between W and Y(1) within each bin:")
print(correlations_Y1)
print("Correlation between W and Y(0) within each bin:")
print(correlations_Y0)

# Plot the correlations within each bin
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(range(num_bins), correlations_Y1)
plt.xlabel('Bins of Z')
plt.ylabel('Correlation between W and Y(1)')
plt.title('Correlation between W and Y(1) within Bins of Z')
plt.xticks(range(num_bins), [f'Bin {i+1}' for i in range(num_bins)])

plt.subplot(1, 2, 2)
plt.bar(range(num_bins), correlations_Y0)
plt.xlabel('Bins of Z')
plt.ylabel('Correlation between W and Y(0)')
plt.title('Correlation between W and Y(0) within Bins of Z')
plt.xticks(range(num_bins), [f'Bin {i+1}' for i in range(num_bins)])

plt.tight_layout()
plt.show()
