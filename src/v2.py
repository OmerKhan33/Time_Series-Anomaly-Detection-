
import pandas as pd
import random
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt", delimiter='\s+', header=None)

combined_data = pd.concat([train_df, test_df], ignore_index=True)
combined_data.to_csv('../ECG5000_Dataset/combined_data.csv', index=False, header=False)
combined_data.head()

# Plot All ECG Signals
ecg_signals = combined_data.iloc[:, 1:]
plt.figure(figsize=(20, 15))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('ECG Signals')
plt.show()

# Extract and Save Normal Data

normal_data = combined_data[combined_data.iloc[:, 0] == 1.0]
normal_data.to_csv('../ECG5000_Dataset/ecg5000_Normal.csv', index=False)
normal_data = pd.read_csv('../ECG5000_Dataset/ecg5000_Normal.csv')

#%% Plot All Normal ECG Signals
ecg_signals = normal_data.iloc[:, 1:]

plt.figure(figsize=(20, 15))
for index, row in ecg_signals.iterrows():
    plt.plot(row, label=f'Signal {index + 1}')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Normal ECG Signals')
plt.show()

#%% Plot One Normal ECG Recording

selected_normal_signal = ecg_signals.iloc[random.randint(0, len(ecg_signals) - 1)]

plt.figure(figsize=(10, 5))
plt.plot(selected_normal_signal, label='ECG Signal')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Single Normal ECG Signal')
plt.legend()
plt.show()

#%% Extract and Save Abnormal Data

abnormal_data = combined_data[combined_data.iloc[:, 0].isin([2.0, 3.0, 4.0, 5.0])]
abnormal_data.to_csv('../ECG5000_Dataset/ecg5000_Abnormal.csv', index=False)
abnormal_data = pd.read_csv('../ECG5000_Dataset/ecg5000_Abnormal.csv')

# Plot One Abnormal ECG Recording

ecg_signals = abnormal_data.iloc[:, 1:]
selected_abnormal_signal = ecg_signals.iloc[random.randint(0, len(ecg_signals) - 1)]

plt.figure(figsize=(10, 5))
plt.plot(selected_abnormal_signal, label='ECG Signal')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Single Abnormal ECG Signal')
plt.legend()
plt.show()

# Plot the selected ECG signals side by side (normal and abnormal)

# Select a specific ECG signal to plot
selected_normal_signal = normal_data.iloc[random.randint(0, len(normal_data) - 1), 1:]
selected_abnormal_signal = abnormal_data.iloc[random.randint(0, len(abnormal_data) - 1), 1:]

plt.figure(figsize=(15, 5))

# Plot normal ECG signal
plt.subplot(1, 2, 1)
plt.plot(selected_normal_signal, label='Normal ECG Signal')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Normal ECG Signal')
plt.legend()

# Plot abnormal ECG signal
plt.subplot(1, 2, 2)
plt.plot(selected_abnormal_signal, label='Abnormal ECG Signal')
plt.xlabel('Time (Sample Points)')
plt.ylabel('Amplitude')
plt.title('Abnormal ECG Signal')
plt.legend()

plt.tight_layout()
plt.show()

# Histogram of ECG Signal Amplitudes
plt.figure(figsize=(10, 5))
plt.hist(selected_normal_signal, bins=75, alpha=0.7, label='Normal ECG Signal')
plt.hist(selected_abnormal_signal, bins=75, alpha=0.7, label='Abnormal ECG Signal')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of ECG Signal Amplitudes')
plt.legend()
plt.show()

# Density Plot of ECG Signal Amplitudes
plt.figure(figsize=(10, 5))
sns.kdeplot(selected_normal_signal, label='Normal ECG Signal', fill=True, alpha=0.5)
sns.kdeplot(selected_abnormal_signal, label='Abnormal ECG Signal', fill=True, alpha=0.5)
plt.xlabel('Amplitude')
plt.ylabel('Density')
plt.title('Density Plot of ECG Signal Amplitudes')
plt.legend()
plt.show()

# Heatmap of Multiple ECG Signals
plt.figure(figsize=(10, 5))
sns.heatmap(normal_data.iloc[:10, :], cmap='viridis')
plt.xlabel('Time (Sample Points)')
plt.ylabel('ECG Signal Index')
plt.title('Heatmap of Normal ECG Signals')
plt.show()

# Basic and Detailed Boxplot of ECG Signal Amplitudes

# Create a DataFrame for plotting
boxplot_data = pd.DataFrame({
    'Amplitude': pd.concat([selected_normal_signal, selected_abnormal_signal]),
    'Type': ['Normal'] * len(selected_normal_signal) + ['Abnormal'] * len(selected_abnormal_signal)
})

plt.figure(figsize=(20, 8))

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
