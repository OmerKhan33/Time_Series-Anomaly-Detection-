import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from darts import TimeSeries
from sklearn.preprocessing import RobustScaler
import warnings
warnings.filterwarnings('ignore')



train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt", delimiter='\s+', header=None)

# Remove the label column
normal_data = pd.read_csv('../ECG5000_Dataset/ecg5000_Normal.csv')
normal_data.drop(normal_data.columns[0], axis=1, inplace=True)
normal_data.head()

# Split data into train, valdidation, and test


# Split into training and remaining (test + validation)
train_data, temp_data = train_test_split(normal_data, test_size=0.3, random_state=42)

# Split the remaining data into test and validation
test_data, validation_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Print the shapes of the datasets
print(f"Training Data Shape: {train_data.shape}")
print(f"Validation Data Shape: {validation_data.shape}")
print(f"Test Data Shape: {test_data.shape}")

# Normalize the data using RobustScaler

# Initialize RobustScaler
scaler = RobustScaler()

# Fit on training data and transform train, validation, and test sets
train_features_scaled = scaler.fit_transform(train_data)
validation_features_scaled = scaler.transform(validation_data)
test_features_scaled = scaler.transform(test_data)

# Convert scaled features back to DataFrames if needed
train_data_scaled = pd.DataFrame(train_features_scaled, columns=train_data.columns)
validation_data_scaled = pd.DataFrame(validation_features_scaled, columns=validation_data.columns)
test_data_scaled = pd.DataFrame(test_features_scaled, columns=test_data.columns)

# Print the shapes of the scaled datasets
print(f"Scaled Training Data Shape: {train_data_scaled.shape}")
print(f"Scaled Validation Data Shape: {validation_data_scaled.shape}")
print(f"Scaled Test Data Shape: {test_data_scaled.shape}")

#%% Histogram of scaled data
plt.figure(figsize=(10, 5))
plt.hist(train_data_scaled, bins=75, alpha=0.7, label='Normalised Train')
plt.hist(validation_data_scaled, bins=75, alpha=0.7, label='Normalised Validation')
plt.hist(test_data_scaled, bins=75, alpha=0.7, label='Normalised Test')
plt.xlabel('Amplitude')
plt.ylabel('Frequency')
plt.title('Histogram of ECG Signal Amplitudes')
plt.legend()
plt.show()

# Remove index
train_data_scaled.reset_index(drop=True, inplace=True)
validation_data_scaled.reset_index(drop=True, inplace=True)
test_data_scaled.reset_index(drop=True, inplace=True)
train_data_scaled.head()
validation_data_scaled.head()
test_data_scaled.head()

# Printing infos
train_data_scaled.info()
validation_data_scaled.info()
test_data_scaled.info()

# Convert Datasets to TimeSeries
train_series = TimeSeries.from_dataframe(train_data_scaled)
validation_series = TimeSeries.from_dataframe(validation_data_scaled)
test_series = TimeSeries.from_dataframe(test_data_scaled)
train_series = train_series.astype(np.float32)
validation_series = validation_series.astype(np.float32)

# Displaying the first 5 rows of dataframes after adding TimeSeries
train_series.head(5)
validation_series.head(5)
test_series.head(5)