import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
from darts import TimeSeries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler





#  Remove Label column
normal_data.drop(normal_data.columns[0], axis=1, inplace=True)
abnormal_data.drop(abnormal_data.columns[0], axis=1, inplace=True)

normal_data.head()

abnormal_data.head()

# Train, Validation and Test Split

# Split into training and remaining (test + validation)
train_data, temp_data = train_test_split(normal_data, test_size=0.3, random_state=42)

# Split the remaining data into test and validation
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {val_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

#Scaling
# Initialize RobustScaler
scaler = RobustScaler()

# Fit on training data and transform train, validation, and test sets
train_features_scaled = scaler.fit_transform(train_data)
val_features_scaled = scaler.transform(val_data)
test_features_scaled = scaler.transform(test_data)

# Convert scaled features back to DataFrames if needed
train_data_scaled = pd.DataFrame(train_features_scaled, columns=train_data.columns)
val_data_scaled = pd.DataFrame(val_features_scaled, columns=val_data.columns)
test_data_scaled = pd.DataFrame(test_features_scaled, columns=test_data.columns)

# Remove index
train_data_scaled.reset_index(drop=True, inplace=True)
val_data_scaled.reset_index(drop=True, inplace=True)
test_data_scaled.reset_index(drop=True, inplace=True)

train_data_scaled.head()

val_data_scaled.head()

test_data_scaled.head()

# Printing infos
train_data_scaled.info()
val_data_scaled.info()
test_data_scaled.info()

#Adding TimeSeries
train_series = TimeSeries.from_dataframe(train_data_scaled)
val_series = TimeSeries.from_dataframe(val_data_scaled)
test_series = TimeSeries.from_dataframe(test_data_scaled)
train_series = train_series.astype(np.float32)
val_series = val_series.astype(np.float32)
test_series = test_series.astype(np.float32)

train_series.head(5)

val_series.head(5)

test_series.head(5)