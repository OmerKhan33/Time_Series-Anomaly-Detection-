"""
This script preprocesses time series data for anomaly detection using a Temporal Convolutional Network (TCN) model.
The process includes the following steps:
1. Importing necessary libraries and modules.
2. Loading and preparing normal and abnormal datasets by removing the label column.
3. Splitting the normal dataset into training, validation, and test sets.
4. Scaling the datasets using RobustScaler to reduce the impact of outliers.
5. Converting the scaled data into TimeSeries objects, which are compatible with the Darts library.

This setup is intended for use with the Darts library for time series modeling, particularly in scenarios involving anomaly detection.
"""

import warnings
from src.B_graphs import normal_data, abnormal_data
warnings.filterwarnings('ignore')  # Ignore warnings
import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

# Remove the label column from both normal and abnormal datasets
normal_data.drop(normal_data.columns[0], axis=1, inplace=True)
abnormal_data.drop(abnormal_data.columns[0], axis=1, inplace=True)

# Display the first few rows of the datasets to ensure correct loading
normal_data.head()
abnormal_data.head()

# Train, Validation, and Test Split

# Split the normal data into training and remaining (test + validation) datasets
train_data, temp_data = train_test_split(normal_data, test_size=0.3, random_state=42)

# Split the remaining data into test and validation datasets
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the shapes of the datasets to verify correct splitting
print(f"Training Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# Scaling the Data

# Initialize RobustScaler to reduce the effect of outliers
scaler = RobustScaler()

# Fit the scaler on training data and transform train, validation, and test sets
train_features_scaled = scaler.fit_transform(train_data)
val_features_scaled = scaler.transform(val_data)
test_features_scaled = scaler.transform(test_data)

# Convert the scaled features back to DataFrames for easier manipulation
train_data_scaled = pd.DataFrame(train_features_scaled, columns=train_data.columns)
val_data_scaled = pd.DataFrame(val_features_scaled, columns=val_data.columns)
test_data_scaled = pd.DataFrame(test_features_scaled, columns=test_data.columns)

# Reset the index of the DataFrames to ensure continuous indexing
train_data_scaled.reset_index(drop=True, inplace=True)
val_data_scaled.reset_index(drop=True, inplace=True)
test_data_scaled.reset_index(drop=True, inplace=True)

# Display the first few rows of the scaled datasets to verify scaling
train_data_scaled.head()
val_data_scaled.head()
test_data_scaled.head()

# Print the info of the scaled datasets to check for any issues
train_data_scaled.info()
val_data_scaled.info()
test_data_scaled.info()

# Converting DataFrames to TimeSeries objects for use with the Darts library

# Create TimeSeries objects from the scaled training, validation, and test datasets
train_series = TimeSeries.from_dataframe(train_data_scaled)
val_series = TimeSeries.from_dataframe(val_data_scaled)
test_series = TimeSeries.from_dataframe(test_data_scaled)

# Ensure the series are in the correct format (float32) for modeling
train_series = train_series.astype(np.float32)
val_series = val_series.astype(np.float32)
test_series = test_series.astype(np.float32)

# Display the first few rows of the TimeSeries objects to verify correct conversion
train_series.head(5)
val_series.head(5)
test_series.head(5)
