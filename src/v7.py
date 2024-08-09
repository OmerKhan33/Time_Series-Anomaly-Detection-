import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from src.v4 import ecg_model
from v5 import abnormal_series

# Ensure the data is of type float32
abnormal_series = abnormal_series.astype(np.float32)

# Obtain predictions on Abnormal Data
abnormal_reconstructed = ecg_model.predict(n=len(abnormal_series), series=abnormal_series)

# Convert series to DataFrame if not already
abnormal_series_df = pd.DataFrame(abnormal_series, columns=['Value'])
abnormal_reconstructed_df = pd.DataFrame(abnormal_reconstructed, columns=['Value'])

# Calculate reconstruction errors
reconstruction_errors = (abnormal_series_df - abnormal_reconstructed_df).abs()

# Determine the Anomaly Threshold
threshold = reconstruction_errors.mean().item() + 3 * reconstruction_errors.std().item()
threshold_value = threshold

# Identify anomalies
anomalies = reconstruction_errors > threshold

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(abnormal_series_df.index, abnormal_series_df['Value'], label='Actual', color='blue')
plt.plot(abnormal_reconstructed_df.index, abnormal_reconstructed_df['Value'], label='Reconstructed', color='green')
plt.scatter(abnormal_series_df.index[anomalies.values.flatten()], abnormal_series_df['Value'][anomalies.values.flatten()], color='red', label='Anomalies')
plt.axhline(y=threshold_value, color='r', linestyle='--', label='Anomaly Threshold')
plt.legend()
plt.xlabel('Index')
plt.ylabel('Value')
plt.title('ECG Data Reconstruction and Anomalies')
plt.show()
