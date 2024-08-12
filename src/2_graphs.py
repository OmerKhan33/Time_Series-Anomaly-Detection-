# EXTRACTING NORMAL AND ABNORMAL DATA
import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

Train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
Test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt", delimiter='\s+', header=None)

# Merge the datasets
df = pd.concat([Train_df, Test_df], ignore_index=True)
df.to_csv('../ECG5000_Dataset/Combined_data.csv', index=False, header=False)
df.head()
df.info()
df.describe()
df.isnull().sum()
df.dropna(inplace=True)

normal_data = df.loc[df[0] == 1]
abnormal_data = df.loc[df[0] != 1]
normal_data.to_csv('../ECG5000_Dataset/normal data.csv', index=False)
abnormal_data.to_csv('../ECG5000_Dataset/abnormal data.csv', index=False)


               #PLOTS
# ECG SIGNALS WITH NORMAL AND ABNORMAL DATA.
ecg_signals = df.iloc[:, 1:]
plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('ECG Signals')
plt.show()

# ECG SIGNALS WITH NORMAL DATA
ecg_signals = normal_data.iloc[:, 1:]

plt.figure(figsize=(20, 30))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Normal ECG Signals')
plt.show()

# Assuming normal_data contains the normal ECG signals
normal_signals = normal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_normal_signal = normal_signals.iloc[random.randint(0, len(normal_signals) - 1)]

# Assuming abnormal_data contains the abnormal ECG signals
abnormal_signals = abnormal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_abnormal_signal = abnormal_signals.iloc[random.randint(0, len(abnormal_signals) - 1)]

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 5))

# Plot the SINGLE normal ECG signal
axs[0].plot(selected_normal_signal, label='Normal ECG Signal', color='blue')
axs[0].set_xlabel('Time (Sample Points)')
axs[0].set_ylabel('Amplitude')
axs[0].set_title('Normal ECG Signal')
axs[0].legend()

# Plot the SINGLE abnormal ECG signal
axs[1].plot(selected_abnormal_signal, label='Abnormal ECG Signal', color='red')
axs[1].set_xlabel('Time (Sample Points)')
axs[1].set_ylabel('Amplitude')
axs[1].set_title('Abnormal ECG Signal')
axs[1].legend()

# Display the plot
plt.show()

# Assuming ecg_signals contains the normal data
normal_signals = normal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_normal_signal = normal_signals.iloc[random.randint(0, len(normal_signals) - 1)]

# Assuming ecg_signals contains the abnormal data
abnormal_signals = abnormal_data.iloc[:, 1:]  # Adjust the slicing as per your data structure
selected_abnormal_signal = abnormal_signals.iloc[random.randint(0, len(abnormal_signals) - 1)]


# Histogram
plt.figure(figsize=(10, 5))
plt.hist(normal_data, bins=75, alpha=0.7, label='Normal ECG Signal')
plt.hist(abnormal_data, bins=75, alpha=0.7, label='Abnormal ECG Signal')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of ECG Signal Amplitudes')
plt.legend()
plt.show()

# Density PLOt
plt.figure(figsize=(10, 5))

# Plot density for normal ECG signals without labels in the legend
sns.kdeplot(normal_data, fill=True, alpha=0.5, color='blue', legend=False)

# Plot density for abnormal ECG signals without labels in the legend
sns.kdeplot(abnormal_data, fill=True, alpha=0.5, color='red', legend=False)

plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.show()


# Combine all columns of normal and abnormal data into single series
normal_combined = normal_data.values.flatten()
abnormal_combined = abnormal_data.values.flatten()

# Plotting the density plots
plt.figure(figsize=(10, 5))
sns.kdeplot(normal_combined, label='Normal ECG Signal', fill=True, alpha=0.5, color='blue')
sns.kdeplot(abnormal_combined, label='Abnormal ECG Signal', fill=True, alpha=0.5, color='red')
plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.legend()
plt.show()



# Create a DataFrame for plotting
boxplot_data = pd.DataFrame({
    'Amplitude': pd.concat([selected_normal_signal, selected_abnormal_signal]),
    'Type': ['Normal'] * len(selected_normal_signal) + ['Abnormal'] * len(selected_abnormal_signal)
})

plt.figure(figsize=(20, 20))

# Basic Boxplot
plt.subplot(1, 2, 1)
sns.boxplot(x='Type', y='Amplitude', data=boxplot_data, palette='Set2')
plt.title('Basic Boxplot of ECG Signal Amplitudes')

# Detailed Boxplot
plt.subplot(1, 2, 2)
sns.boxplot(x='Type', y='Amplitude', data=boxplot_data, palette='Set2', showfliers=False)
sns.swarmplot(x='Type', y='Amplitude', data=boxplot_data, color='k', alpha=0.5, dodge=True)

# Add mean markers
mean_values = boxplot_data.groupby('Type')['Amplitude'].mean()
for i, mean in enumerate(mean_values):
    plt.scatter(x=i, y=mean, color='red', marker='D', s=100, label='Mean' if i == 0 else "", zorder=10)

plt.title('Detailed Boxplot of ECG Signal Amplitudes')
plt.legend()

plt.tight_layout()
plt.show()