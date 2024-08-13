"""
This script processes ECG5000 dataset files to analyze and visualize ECG signals. The steps performed are as follows:

1. **Data Preparation:**
   - Reads ECG training and test data from text files.
   - Merges the training and test datasets into a single DataFrame.
   - Saves the combined dataset to a new CSV file.
   - Separates the combined data into normal and abnormal categories based on label values and saves these subsets as separate CSV files.

2. **Data Visualization:**
   - Plots all ECG signals from the combined dataset.
   - Plots ECG signals specifically for normal data.
   - Selects and plots individual normal and abnormal ECG signals for comparison.
   - Generates density plots to visualize the distribution of ECG signal amplitudes for normal and abnormal data.


The code includes various types of plots to facilitate the analysis and understanding of the ECG signals, providing insights into both individual and comparative signal behaviors.
"""


import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

# Read training and test data from text files
Train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
Test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt", delimiter='\s+', header=None)

# Merge the training and test datasets
df = pd.concat([Train_df, Test_df], ignore_index=True)

# Save the combined dataset to a new CSV file
df.to_csv('../ECG5000_Dataset/Combined_data.csv', index=False, header=False)

# Display basic information and summary statistics of the combined dataset
print(df.head())
print(df.info())
print(df.describe())

# Remove any rows with missing values
df.dropna(inplace=True)

# Separate the data into normal and abnormal categories
normal_data = df.loc[df[0] == 1]
abnormal_data = df.loc[df[0] != 1]

# Save normal and abnormal data to separate CSV files
normal_data.to_csv('../ECG5000_Dataset/normal_data.csv', index=False)
abnormal_data.to_csv('../ECG5000_Dataset/abnormal_data.csv', index=False)

# PLOTTING

# Plot all ECG signals from the combined dataset
ecg_signals = df.iloc[:, 1:]  # Exclude the first column which contains labels
plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('ECG Signals')
plt.show()

# Plot only normal ECG signals
ecg_signals = normal_data.iloc[:, 1:]  # Exclude the first column which contains labels
plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Normal ECG Signals')
plt.show()

# Randomly select and plot one normal and one abnormal ECG signal
normal_signals = normal_data.iloc[:, 1:]  # Exclude the first column
selected_normal_signal = normal_signals.iloc[random.randint(0, len(normal_signals) - 1)]

abnormal_signals = abnormal_data.iloc[:, 1:]  # Exclude the first column
selected_abnormal_signal = abnormal_signals.iloc[random.randint(0, len(abnormal_signals) - 1)]

# Create subplots to compare one normal and one abnormal ECG signal
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot the selected normal ECG signal
axs[0].plot(selected_normal_signal, label='Normal ECG Signal', color='blue')
axs[0].set_xlabel('Time (Sample Points)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Normal ECG Signal')
axs[0].legend()

# Plot the selected abnormal ECG signal
axs[1].plot(selected_abnormal_signal, label='Abnormal ECG Signal', color='red')
axs[1].set_xlabel('Time (Sample Points)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Abnormal ECG Signal')
axs[1].legend()

plt.show()

# Generate density plots for ECG signal amplitudes
plt.figure(figsize=(10, 5))

# Plot density for normal ECG signals
sns.kdeplot(normal_data.iloc[:, 1:].values.flatten(), fill=True, alpha=0.5, color='blue', legend=False)

# Plot density for abnormal ECG signals
sns.kdeplot(abnormal_data.iloc[:, 1:].values.flatten(), fill=True, alpha=0.5, color='red', legend=False)

plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.show()

# Combine and flatten all normal and abnormal ECG signal amplitudes
normal_combined = normal_data.iloc[:, 1:].values.flatten()
abnormal_combined = abnormal_data.iloc[:, 1:].values.flatten()

# Plot density plots for combined normal and abnormal signal amplitudes
plt.figure(figsize=(10, 5))
sns.kdeplot(normal_combined, label='Normal ECG Signal', fill=True, alpha=0.5, color='blue')
sns.kdeplot(abnormal_combined, label='Abnormal ECG Signal', fill=True, alpha=0.5, color='red')
plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of Combined ECG Signal Amplitudes')
plt.legend()
plt.show()

