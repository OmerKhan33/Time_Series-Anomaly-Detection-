import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.ad import ForecastingAnomalyModel, NormScorer
from darts.models import TCNModel

# Reading the TRAIN.txt file
train_df = pd.read_csv("ECG5000_Dataset/ECG5000_TRAIN.txt", delim_whitespace=True, header=None)
print("Training Data:")
print(train_df)

# Reading the TEST.txt file
test_df = pd.read_csv("ECG5000_Dataset/ECG5000_TEST.txt", delim_whitespace=True, header=None)
print("Test Data:")
print(test_df)

# Check the shape of both datasets
print(f"Training Data Shape: {train_df.shape}")
print(f"Test Data Shape: {test_df.shape}")

# Check for null values in both datasets
print(f"Null values in Training Data:\n{train_df.isnull().sum()}")
print(f"Null values in Test Data:\n{test_df.isnull().sum()}")

# Display the first few rows of each dataset to get an idea of the style/format
print("First few rows of Training Data:")
print(train_df.head())
print("First few rows of Test Data:")
print(test_df.head())

# Raise alerts if there are any null values
if train_df.isnull().sum().any():
    print("Alert: There are null values in the Training Data!")
else:
    print("No null values in the Training Data.")

if test_df.isnull().sum().any():
    print("Alert: There are null values in the Test Data!")
else:
    print("No null values in the Test Data.")

# Check for duplicate rows in both datasets
train_duplicates = train_df.duplicated().sum()
test_duplicates = test_df.duplicated().sum()

print(f"Duplicate rows in Training Data: {train_duplicates}")
print(f"Duplicate rows in Test Data: {test_duplicates}")

# Raise alerts if there are any duplicate rows
if train_duplicates > 0:
    print("Alert: There are duplicate rows in the Training Data!")
else:
    print("No duplicate rows in the Training Data.")

if test_duplicates > 0:
    print("Alert: There are duplicate rows in the Test Data!")
else:
    print("No duplicate rows in the Test Data.")

# Explore basic statistics of the datasets
print("Training Data Statistics:")
print(train_df.describe())
print("Test Data Statistics:")
print(test_df.describe())

# Label count for the first column of datasets
print("Label Count for Training Data:")
train_label_counts = train_df[0].value_counts()
print(train_label_counts)

print("Label Count for Test Data:")
test_label_counts = test_df[0].value_counts()
print(test_label_counts)

# Verify which label corresponds to "normal" class
normal_label = 1  # Typically, label 1 is the "normal" class

# Count occurrences of the normal class
train_normal_count = train_label_counts.get(normal_label, 0)
test_normal_count = test_label_counts.get(normal_label, 0)

print(f"Number of normal instances in Training Data: {train_normal_count}")
print(f"Number of normal instances in Test Data: {test_normal_count}")

# Merge the datasets
combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
print("Combined Data Shape:", combined_df.shape)

# Separate normal and anomalous data
normal_data = combined_df[combined_df[0] == normal_label]
anomalous_data = combined_df[combined_df[0] != normal_label]

# Plot one sample from normal and one from anomalous data
plt.figure(figsize=(12, 6))

# Plot normal sample
plt.subplot(1, 2, 1)
plt.plot(normal_data.iloc[0, 1:])
plt.title('Normal Sample')

# Plot anomalous sample
plt.subplot(1, 2, 2)
plt.plot(anomalous_data.iloc[0, 1:])
plt.title('Anomalous Sample')

plt.tight_layout()
plt.show()

# Additional EDA on the combined dataset

# Label count for the combined dataset
print("Label Count for Combined Data:")
combined_label_counts = combined_df[0].value_counts()
print(combined_label_counts)

# 1. Visualizing the Distribution of Classes in Combined Data
plt.figure(figsize=(10, 5))
sns.countplot(x=combined_df[0])
plt.title("Distribution of Classes in Combined Data")
plt.show()

# 2. Plotting Sample Time Series from Combined Data
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(15, 10))
for i in range(3):
    axes[0, i].plot(combined_df.iloc[i, 1:])
    axes[0, i].set_title(f'Combined Data Sample {i+1}')
    axes[1, i].plot(combined_df.iloc[i+3, 1:])
    axes[1, i].set_title(f'Combined Data Sample {i+4}')
plt.tight_layout()
plt.show()

# 3. Correlation Matrix for Combined Data
plt.figure(figsize=(15, 10))
sns.heatmap(combined_df.corr(), cmap="coolwarm", annot=False)
plt.title("Correlation Matrix for Combined Data")
plt.show()

# 4. Feature Statistics for Combined Data
print("Combined Data Feature Statistics:")
print(combined_df.describe())

# 5. Checking Data Range and Outliers in Combined Data
combined_df.iloc[:, 1:].boxplot(figsize=(20, 10))
plt.title("Boxplot of Combined Data Features")
plt.show()

# 6. Pair Plots (if applicable)
sns.pairplot(combined_df.iloc[:, :5])
plt.show()