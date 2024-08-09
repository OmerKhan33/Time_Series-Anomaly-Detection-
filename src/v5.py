import os
from v3 import validation_series, scaler
from v2 import normal_data
import pandas as pd
from darts import TimeSeries
import warnings
warnings.filterwarnings('ignore')


# Preparing Abnormal data like Normal data

# Load abnormal data
abnormal_data = pd.read_csv('../ECG5000_Dataset/ecg5000_abnormal.csv')

# Remove the label column if it exists
abnormal_data.drop(abnormal_data.columns[0], axis=1, inplace=True)
abnormal_data.head()

# Ensure that the abnormal data has the same columns as the training data
abnormal_data.columns = abnormal_data.columns
print(abnormal_data.columns)
print(normal_data.columns)

# Apply the fitted RobustScaler to the abnormal data
abnormal_features_scaled = scaler.transform(abnormal_data)

# Convert scaled features back to DataFrame
abnormal_data_scaled = pd.DataFrame(abnormal_features_scaled, columns=abnormal_data.columns)

# Print the shape of the scaled abnormal dataset
print(f"Scaled Abnormal Data Shape: {abnormal_data_scaled.shape}")

# Remove index
abnormal_data_scaled.reset_index(drop=True, inplace=True)
abnormal_data_scaled.head()

# Convert abnormal data to TimeSeries format
abnormal_series = TimeSeries.from_dataframe(abnormal_data_scaled)

# Display first 5 rows of the TimeSeries
abnormal_series.head(5)



