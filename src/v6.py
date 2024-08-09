import pandas as pd
import matplotlib.pyplot as plt
from darts.models import TCNModel
from v5 import abnormal_series, normal_data

# Define the path to the saved model
model_save_path = '../ECG5000/MODEL_ECG5000.pth.tar'

# Load the model
loaded_ecg_model = TCNModel.load(model_save_path)
print("Model loaded successfully")

# Ensure that the data is in Pandas Series format
if not isinstance(normal_data, pd.Series):
    normal_series = pd.Series(normal_data)
else:
    normal_series = normal_data

if not isinstance(abnormal_series, pd.Series):
    anomalous_series = pd.Series(abnormal_series)
else:
    anomalous_series = abnormal_series

# Convert the series to float32
normal_series = normal_series.astype('float32')
anomalous_series = anomalous_series.astype('float32')

# Forecast length should match the length of the input data for comparison
forecast_length = len(anomalous_series)

# Make predictions
anomalous_forecast = loaded_ecg_model.predict(n=forecast_length, series=anomalous_series)
normal_forecast = loaded_ecg_model.predict(n=forecast_length, series=normal_series)

# Ensure forecasts are in Pandas Series format
if not isinstance(anomalous_forecast, pd.Series):
    anomalous_forecast = pd.Series(anomalous_forecast)
if not isinstance(normal_forecast, pd.Series):
    normal_forecast = pd.Series(normal_forecast)

# Plotting
plt.figure(figsize=(14, 7))

# Plot for anomalous data
plt.subplot(2, 1, 1)
plt.plot(anomalous_series.index, anomalous_series, label='Actual Anomalous Data', color='blue')
plt.plot(anomalous_series.index, anomalous_forecast, label='Forecasted Anomalous Data', color='green')
plt.title('Anomalous Data Forecasting')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

# Plot for normal data
plt.subplot(2, 1, 2)
plt.plot(normal_series.index, normal_series, label='Actual Normal Data', color='blue')
plt.plot(normal_series.index, normal_forecast, label='Forecasted Normal Data', color='green')
plt.title('Normal Data Forecasting')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
