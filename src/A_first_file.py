"""
This script performs preliminary data exploration and quality checks on two datasets:

1. Training Data (Train_df) - Loaded from "ECG5000_TRAIN.txt"
2. Testing Data (Test_df) - Loaded from "ECG5000_TEST.txt"

The following operations are performed:
- Read the datasets from text files.
- Print basic information about the datasets.
- Check for missing values and print their counts.
- Print the first few rows of each dataset.
- Check and report if there are any duplicate rows.
- Print basic statistics for each dataset.
- Count and print the distribution of labels, specifically focusing on the 'normal' class.

This script helps in understanding the structure and quality of the datasets before proceeding with further analysis or modeling.
"""
import pandas as pd

# reading train.txt file
Train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt")
print("Training Data:")
print(Train_df)

# reading test data set
Test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt")
print("Testing Data:")
print(Test_df)

# Check the shape of both datasets
print(f"The shape of Test : {Test_df.shape} ")
print(f"The shape of Train : {Train_df.shape}")

# Check for null values in both datasets
print(f"The null values in Test are: {Test_df.isnull().sum()}")
print(f"The null values in Train are: {Train_df.isnull().sum()}")

# Print the first few rows of each dataset
print("First few rows of Test Data:")
print(Test_df.head())
print("First few rows of Training Data:")
print(Train_df.head())

# Check and report if there are any null values
if Test_df.isnull().sum().any() > 0:
    print("There are Null values in the Test data set")
else:
    print("There are no NULL values in the Test Data set")

if Train_df.isnull().sum().any() > 0:
    print("There are Null values in the Train dataset")
else:
    print("There are no NULL values in the Train dataset")

# Check for duplicate rows in both datasets
train_duplicates = Train_df.duplicated().sum()
test_duplicates = Test_df.duplicated().sum()

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
print(Train_df.describe())
print("Test Data Statistics:")
print(Test_df.describe())

# Label count for the first column of datasets
train_label_counts = Train_df.iloc[:, 0].value_counts()
print("Label Count for Training Data:")
print(train_label_counts)

test_label_counts = Test_df.iloc[:, 0].value_counts()
print("Label Count for Testing Data:")
print(test_label_counts)

# Assuming label 1 is the "normal" class
normal_label = 1

# Count occurrences of the normal class
train_normal_count = train_label_counts.get(normal_label, 0)
test_normal_count = test_label_counts.get(normal_label, 0)

print(f"Number of normal instances in Training Data: {train_normal_count}")
print(f"Number of normal instances in Test Data: {test_normal_count}")
