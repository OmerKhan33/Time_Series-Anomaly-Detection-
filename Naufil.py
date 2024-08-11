import numpy as np
import pandas as pd
from darts import TimeSeries
from sklearn.model_selection import train_test_split
from darts.models import TCNModel
import matplotlib.pyplot as plt
# Load the dataset
dataframe = pd.read_csv('ecg.csv', header=None)
raw_data = dataframe.values

# Separate labels and data
labels = raw_data[:, -1]
data = raw_data[:, :-1]

# Convert the data into Darts TimeSeries objects
series_list = [TimeSeries.from_values(data[i]) for i in range(data.shape[0])]

# Create a DataFrame to hold the labels and the corresponding series
labeled_data = pd.DataFrame({'series': series_list, 'labels': labels})

# Perform a train-test split using sklearn
train_indices, test_indices = train_test_split(
    labeled_data.index, test_size=0.2, random_state=21
)

# Extract the training and testing data
train_series = labeled_data.loc[train_indices, 'series'].tolist()
test_series = labeled_data.loc[test_indices, 'series'].tolist()
train_labels = labeled_data.loc[train_indices, 'labels'].values
test_labels = labeled_data.loc[test_indices, 'labels'].values

# Convert TimeSeries to NumPy arrays for scaling
train_data = np.array([series.values() for series in train_series])
test_data = np.array([series.values() for series in test_series])

# Min-Max Scaling
min_val = np.min(train_data)
max_val = np.max(train_data)

train_data = (train_data - min_val) / (max_val - min_val)
test_data = (test_data - min_val) / (max_val - min_val)

# Convert back to TimeSeries
train_series = [TimeSeries.from_values(data) for data in train_data]
test_series = [TimeSeries.from_values(data) for data in test_data]

# Convert data to float32
train_data = train_data.astype(np.float32)
test_data = test_data.astype(np.float32)

# Convert labels to boolean
train_labels = train_labels.astype(bool)
test_labels = test_labels.astype(bool)

# Extract normal and anomalous data
normal_train_data = train_data[train_labels]
anomalous_train_data = train_data[~train_labels]
normal_test_data = test_data[test_labels]
anomalous_test_data = test_data[~test_labels]
import matplotlib.pyplot as plt
plt.grid()
plt.plot(np.arange(140), normal_train_data[0])
plt.title("A Normal ECG")
plt.show()
plt.grid()
plt.plot(np.arange(140), anomalous_train_data[0])
plt.title("An Anomalous ECG")
plt.show()
normal_train_series = [TimeSeries.from_values(data) for data in normal_train_data]


# Define the TCN model
model = TCNModel(
    input_chunk_length=30,  # Length of the input sequence
    output_chunk_length=1,  # Number of time steps to predict ahead
    kernel_size=3,          # Size of the convolutional kernel
    num_filters=64,         # Number of filters in the convolutional layers
    dropout=0.1,            # Dropout rate
    dilation_base=2,         # Base for the dilation factor
    n_epochs=2
)

# Fit the model
model.fit(normal_train_series)

# Optionally, you can save the model for later use
# model.save('tcn_model.pth')

normal_test_series = [TimeSeries.from_values(data) for data in normal_test_data]


# Assuming `len(series)` is the length of future time steps you want to predict
num_future_steps = 140  # Adjust this according to your needs

# Generate predictions for each test series
predictions = [model.predict(n=num_future_steps, series=series) for series in normal_test_series]

def normalize(series):
    return (series - min_val) / (max_val - min_val)

normal_test_series = [TimeSeries.from_values(normalize(series.values()).astype(np.float32)) for series in test_series]

# Define how many future time steps you want to predict
num_future_steps = 140

#len(normal_test_series)

# Generate predictions
predictions = [model.predict(n=num_future_steps, series=series) for series in normal_test_series[:50]]

index = 0
actual_series = test_series[index].values()
predicted_series = predictions[index].values()

# Ensure that actual_series and predicted_series are 1-dimensional
actual_series = actual_series.flatten()
predicted_series = predicted_series.flatten()


plt.figure(figsize=(12, 6))
plt.plot(actual_series, label='Actual Data', color='b')
plt.plot(predicted_series, label='Predictions', color='r')
plt.fill_between(np.arange(len(actual_series)), actual_series, predicted_series, color='lightcoral', alpha=0.5, label='Error')
plt.legend()
plt.xlabel('Time Steps')
plt.ylabel('Normalized Value')
plt.title('Actual vs Predicted Data')
plt.show()