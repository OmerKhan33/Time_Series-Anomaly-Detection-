
import pandas as pd
import seaborn as sns


# reading test data set
df_test = pd.read_csv("ECG5000_Dataset/ECG5000_TEST.txt")
print("Testing Data:")
print(df_test)

# reading train.txt file
df_train = pd.read_csv("ECG5000_Dataset/ECG5000_TRAIN.txt")
print("Training Data:")
print(df_train)


# Check for shape values in both datasets
print(f"The shape of Test : {df_test.shape} ")
print(f"The shape of Train : {df_train.shape}")

# Check for null values in both datasets
print(f"The null values in Test are: {df_test.isnull().sum()}")
print(f"The null values in Train are: {df_train.isnull().sum()}")

# print some head values
print("First few rows of Test Data:")
print(df_test.head())
print("First few rows of Training Data:")
print(df_train.head())


if df_test.isnull().sum() > 0:
    print("There are Null values Test data set")
else:
    print("There no NULL values Test Data set")

if df_train.isnull().sum() > 0:
    print("There are Null values in Train dataset")
else:
    print("There no NULL values in Train dataset")

train_duplicates = df_train.duplicated().sum()
test_duplicates = df_test.duplicated().sum()

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
print(df_train.describe())
print("Test Data Statistics:")
print(df_test.describe())

# Label count for the first column of datasets
train_label_counts = df_train[0].value_counts()
print("Label Count for Training Data:")
print(train_label_counts)

test_label_counts = df_test[0].value_counts()
print("Label Count for Testing Data:")
print(test_label_counts)

normal_label = 1  # Typically, label 1 is the "normal" class

# Count occurrences of the normal class
train_normal_count = train_label_counts.get(normal_label, 0)
test_normal_count = test_label_counts.get(normal_label, 0)

print(f"Number of normal instances in Training Data: {train_normal_count}")
print(f"Number of normal instances in Test Data: {test_normal_count}")

# Merge the datasets
combined_df = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
print("Combined Data Shape:", combined_df.shape)

# Separate normal and anomalous data
normal_data = combined_df[combined_df[0] == normal_label]
anomalous_data = combined_df[combined_df[0] != normal_label]