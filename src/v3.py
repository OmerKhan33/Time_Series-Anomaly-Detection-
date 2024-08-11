import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from darts import TimeSeries
from darts.models import TCNModel
from darts.metrics import mape

# Load the dataset
train_data = pd.read_csv(
    "C:/Users/omerk/PycharmProjects/Time_Series-Anomaly-Detection-/ECG5000_Dataset/ECG5000_TRAIN.txt", sep='\s+',
    header=None)
test_data = pd.read_csv(
    "C:/Users/omerk/PycharmProjects/Time_Series-Anomaly-Detection-/ECG5000_Dataset/ECG5000_TEST.txt", sep='\s+',
    header=None)

# Merge the training and test datasets
merged_data = pd.concat([train_data, test_data], axis=0).reset_index(drop=True)

# Define normal and anomalous classes
normal_class_label = 1

# Split the data into normal and anomalous
normal_data = merged_data[merged_data[0] == normal_class_label]
anomalous_data = merged_data[merged_data[0] != normal_class_label]

# Combine normal and anomalous data
combined_data = pd.concat([normal_data, anomalous_data], axis=0).reset_index(drop=True)

# Split combined data into features and labels
X_combined = combined_data.iloc[:, 1:].values
y_combined = combined_data.iloc[:, 0].values

# Split the combined data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X_combined, y_combined, test_size=0.4, random_state=42,
                                                    stratify=y_combined)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)


# Convert the data to TimeSeries format
def convert_to_timeseries(X, y):
    timeseries_list = []
    for i in range(X.shape[0]):
        series = TimeSeries.from_times_and_values(
            pd.date_range(start="2020-01-01", periods=X.shape[1], freq='ms'),
            X[i]
        )
        timeseries_list.append(series)
    return timeseries_list


train_series = convert_to_timeseries(X_train, y_train)
val_series = convert_to_timeseries(X_val, y_val)
test_series = convert_to_timeseries(X_test, y_test)

# Create a TCN model
model = TCNModel(input_chunk_length=20, output_chunk_length=1, n_epochs=10, random_state=42)

# Implement early stopping manually
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(model.n_epochs/2):
    print(f"Epoch {epoch + 1}/{model.n_epochs}")
    model.fit(series=train_series, val_series=val_series)

    # Evaluate the model on the validation set
    val_predictions = [model.predict(n=len(ts), series=ts) for ts in val_series]

    val_target_values = np.concatenate([ts.values() for ts in val_series])
    val_predicted_values = np.concatenate([ts.values() for ts in val_predictions])

    current_val_loss = mape(TimeSeries.from_values(val_target_values), TimeSeries.from_values(val_predicted_values))

    print(f"Validation MAPE: {current_val_loss}")

    if current_val_loss < best_val_loss:
        best_val_loss = current_val_loss
        patience_counter = 0
        model.save("tcn_ecg5000_model.pth")
    else:
        patience_counter += 1

    if patience_counter >= patience:
        print("Early stopping triggered")
        break

# Load the best model
model.load("tcn_ecg5000_model.pth")

# Prepare test data for prediction
test_predictions = [model.predict(n=len(ts), series=ts) for ts in test_series]


# Compute anomaly scores based on prediction errors
def compute_anomaly_scores(target_series, predicted_series):
    anomaly_scores = []
    for ts, pred in zip(target_series, predicted_series):
        residuals = ts.values() - pred.values()
        scores = np.abs(residuals)
        anomaly_scores.append(TimeSeries.from_times_and_values(ts.time_index, scores))
    return anomaly_scores


anomaly_scores = compute_anomaly_scores(test_series, test_predictions)


# Threshold to create binary anomalies
def threshold_anomalies(anomaly_scores, threshold=0.5):
    binary_anomalies = []
    for score_ts in anomaly_scores:
        binary = score_ts.values() > threshold
        binary_anomalies.append(TimeSeries.from_times_and_values(score_ts.time_index, binary.astype(int)))
    return binary_anomalies


binary_anomalies = threshold_anomalies(anomaly_scores)

# Print results
for i, ts in enumerate(binary_anomalies):
    print(f"Anomaly Binary Time Series {i}:")
    print(ts)


# Optional: Aggregate binary anomalies if needed (example for a simple average)
def aggregate_anomalies(binary_anomalies):
    aggregated = np.mean([ts.values() for ts in binary_anomalies], axis=0)
    time_index = binary_anomalies[0].time_index
    return TimeSeries.from_times_and_values(time_index, aggregated)



aggregated_anomalies = aggregate_anomalies(binary_anomalies)
print("Aggregated Anomalies:")
print(aggregated_anomalies)
