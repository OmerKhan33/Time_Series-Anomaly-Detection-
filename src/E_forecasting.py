#Libraries
import warnings
from src.C_Pre_Processing import scaler, test_data, train_series, val_series, test_series
from src.D_model_architecture import ecg_model
from darts.models import TCNModel
warnings.filterwarnings('ignore')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.B_graphs import abnormal_data, df
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.ad import ForecastingAnomalyModel, KMeansScorer, NormScorer

# Checking that model loads successfully

from darts.models import TCNModel
from darts.timeseries import TimeSeries

# Load the TCN model
try:
    model = TCNModel.load('../ECG5000_Dataset/Model.pth.tar')  # Replace with your actual model file path
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None



#Forecasting Anomaly Model

abnormal_features_scaled = scaler.transform(abnormal_data)
abnormal_data_scaled = pd.DataFrame(abnormal_features_scaled, columns=test_data.columns)

abnormal_series = TimeSeries.from_dataframe(abnormal_data_scaled)
abnormal_series = abnormal_series.astype(np.float32)

# instantiate the anomaly model with: one fitted model, and 3 scorers
anomaly_model = ForecastingAnomalyModel(
    model=ecg_model,
    scorer=[
        NormScorer(ord=1),
    ],
)

START = 0.1
anomaly_model.fit(train_series, start=START, allow_model_training=False, verbose=True, scorer=NormScorer)

# Calculate anomaly scores for the validation or test series
anomaly_scores, model_forecasting = anomaly_model.score(
    test_series, start=START, return_model_prediction=True, verbose=True
)
pred_start = model_forecasting.start_time()

# Extract anomaly scores from the result
anomaly_scores

# Extract the time index and values from the TimeSeries object
time_index = anomaly_scores.time_index
scores = anomaly_scores.values()

# Plot the anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores')
plt.legend()
plt.show()

mae(model_forecasting, test_series)

rmse(model_forecasting, test_series)

# Step 1: Calculate anomaly scores on the validation data
val_anomaly_scores, val_model_forecasting = anomaly_model.score(
    val_series, start=START, return_model_prediction=True, verbose=True
)
pred_start = model_forecasting.start_time()

val_anomaly_scores

# Convert to numpy arrays if they are not already
time_index = np.array(time_index)
scores = np.array(scores)

# Calculate z-scores
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score

# Define the threshold for anomaly detection
threshold = 3
anomalies = z_scores > threshold
anomalies

# %%
# Print the types and shapes of the variables
print(f"Type of time_index: {type(time_index)}")
print(f"Shape of time_index: {np.shape(time_index)}")
print(f"Type of scores: {type(scores)}")
print(f"Shape of scores: {np.shape(scores)}")
print(f"Type of anomalies: {type(anomalies)}")
print(f"Shape of anomalies: {np.shape(anomalies)}")

# Convert to numpy arrays if they are not already
if not isinstance(time_index, np.ndarray):
    time_index = np.array(time_index)
if not isinstance(scores, np.ndarray):
    scores = np.array(scores)
if not isinstance(anomalies, np.ndarray):
    anomalies = np.array(anomalies)

# Verify the shapes after conversion
print(f"Converted type of time_index: {type(time_index)}")
print(f"Shape of time_index after conversion: {np.shape(time_index)}")
print(f"Converted type of scores: {type(scores)}")
print(f"Shape of scores after conversion: {np.shape(scores)}")
print(f"Converted type of anomalies: {type(anomalies)}")
print(f"Shape of anomalies after conversion: {np.shape(anomalies)}")

# Flatten the scores and anomalies arrays to 1D
scores = scores.flatten()
anomalies = anomalies.flatten()

# Plot the anomaly scores
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores with Detected Anomalies')
plt.legend()
plt.show()

# Print the indices of detected anomalies
print("Anomaly indices:", np.where(anomalies)[0])

# Define the chunk size
chunk_size = 50  # Example chunk size


# Function to calculate anomaly scores for each chunk
def calculate_anomaly_scores_by_chunk(time_index, scores, chunk_size, threshold=3):
    num_chunks = len(time_index) // chunk_size
    if len(time_index) % chunk_size != 0:
        num_chunks += 1

    all_anomalies = []
    all_scores = []

    for i in range(num_chunks):
        start_idx = i * chunk_size
        end_idx = min((i + 1) * chunk_size, len(time_index))

        # Extract chunk
        chunk_time_index = time_index[start_idx:end_idx]
        chunk_scores = scores[start_idx:end_idx]

        # Compute mean and std deviation for the chunk
        mean_score = np.mean(chunk_scores)
        std_dev_score = np.std(chunk_scores)

        # Compute z-scores and detect anomalies
        z_scores = (chunk_scores - mean_score) / std_dev_score
        anomalies = z_scores > threshold

        # Append results
        all_anomalies.append(anomalies)
        all_scores.append(chunk_scores)

    # Combine results
    combined_anomalies = np.concatenate(all_anomalies)
    combined_scores = np.concatenate(all_scores)

    return combined_anomalies, combined_scores


# Calculate anomaly scores by chunk
chunk_anomalies, chunk_scores = calculate_anomaly_scores_by_chunk(time_index, scores, chunk_size)

# Plot the anomaly scores

plt.figure(figsize=(12, 6))
plt.plot(time_index, chunk_scores, label="Anomaly Scores")
plt.scatter(time_index[chunk_anomalies], chunk_scores[chunk_anomalies], color='red', label="Detected Anomalies")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores with Detected Anomalies (by Chunks)')
plt.legend()
plt.show()


# Print the indices of detected anomalies
print("Anomaly indices:", np.where(chunk_anomalies)[0])


# Example data: Replace these with your actual data
time_index = np.arange(len(scores))  # Assuming time index is sequential
normal_data = np.random.normal(0, 1, len(scores))  # Replace with actual normal ECG data
anomalous_data = np.random.normal(0, 1, len(scores))  # Replace with actual anomalous ECG data


# Function to plot ECG data with anomalies
def plot_ecg_with_anomalies(time_index, normal_data, anomalous_data, scores, anomalies, threshold=3):
    plt.figure(figsize=(14, 7))

    # Plot normal ECG data
    plt.plot(time_index, normal_data, label="Normal ECG Data", color='blue', alpha=0.5)

    # Plot anomalous ECG data
    plt.plot(time_index, anomalous_data, label="Anomalous ECG Data", color='orange', alpha=0.5)

    # Plot anomaly scores
    plt.plot(time_index, scores, label="Anomaly Scores", color='green', linestyle='--', alpha=0.7)

    # Highlight detected anomalies
    plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies", marker='x')

    plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('ECG Data with Detected Anomalies')
    plt.legend()
    plt.show()


# Example usage
plot_ecg_with_anomalies(time_index, normal_data, anomalous_data, scores, chunk_anomalies)

# Example data: Replace these with your actual data
time_index = np.arange(len(scores))  # Assuming time index is sequential
ecg_data = np.random.normal(0, 1, len(scores))  # Replace with your actual ECG data
chunk_start = 100  # Define start index for the chunk
chunk_end = 200  # Define end index for the chunk

# Define the threshold for anomalies (e.g., z-score > 3)
threshold = 3

# Compute z-scores (assuming you already have scores and anomalies)
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score
anomalies = z_scores > threshold

# Extract the chunk of data
chunk_time_index = time_index[chunk_start:chunk_end]
chunk_ecg_data = ecg_data[chunk_start:chunk_end]
chunk_scores = scores[chunk_start:chunk_end]
chunk_anomalies = anomalies[chunk_start:chunk_end]


# Function to plot a chunk of ECG data with normal and anomalous segments
def plot_ecg_chunk_with_normal_and_anomalous_segments(time_index, ecg_data, scores, anomalies, threshold):
    plt.figure(figsize=(14, 7))

    # Plot segments
    current_color = 'green'
    for i in range(len(time_index) - 1):
        # Switch color when an anomaly is detected
        if anomalies[i]:
            plt.plot(time_index[i:i + 2], ecg_data[i:i + 2], color='red', alpha=0.8)
        else:
            plt.plot(time_index[i:i + 2], ecg_data[i:i + 2], color=current_color, alpha=0.8)

    # Plot anomaly scores
    plt.plot(time_index, scores, label="Anomaly Scores", color='blue', linestyle='--', alpha=0.7)

    # Highlight detected anomalies
    plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies", marker='x')

    plt.axhline(y=threshold, color='red', linestyle='--', label="Threshold")
    plt.xlabel('Time')
    plt.ylabel('ECG Value')
    plt.title('ECG Chunk with Normal and Anomalous Segments')
    plt.legend()
    plt.show()


# Plot the chunk
plot_ecg_chunk_with_normal_and_anomalous_segments(chunk_time_index, chunk_ecg_data, chunk_scores, chunk_anomalies,
                                                  threshold)

print("Time Index Chunk:", chunk_time_index)
print("ECG Data Chunk:", chunk_ecg_data)
print("Anomalies Chunk:", chunk_anomalies)

if np.any(chunk_anomalies):
    print("Anomalies detected in chunk.")
else:
    print("No anomalies detected in chunk.")

plt.figure(figsize=(14, 7))
for i in range(len(chunk_time_index) - 1):
    plt.plot(chunk_time_index[i:i + 2], chunk_ecg_data[i:i + 2], color='red' if chunk_anomalies[i] else 'green',
             alpha=0.8)
plt.xlabel('Time')
plt.ylabel('ECG Value')
plt.title('ECG Chunk with Normal and Anomalous Segments')
plt.show()

