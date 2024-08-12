"""
This script is designed to perform anomaly detection on ECG time series data using a Temporal Convolutional Network (TCN) model and the Darts library. The steps include:

1. **Model Loading**: Loading a pre-trained TCN model and verifying its successful load.
2. **Anomaly Detection Setup**: Using a ForecastingAnomalyModel with specific scoring methods to detect anomalies in the ECG data.
3. **Anomaly Scoring**: Calculating and plotting anomaly scores for both validation and test datasets.
4. **Anomaly Detection Visualization**: Visualizing detected anomalies within the ECG data using different plotting methods.
5. **Chunk-Based Analysis**: Implementing chunk-based anomaly detection to handle large datasets and plot ECG segments with normal and anomalous segments.
6. **Anomaly Highlighting**: Highlighting and analyzing specific segments of ECG data where anomalies are detected.

This script is structured for use in anomaly detection tasks, particularly in the context of medical time series data like ECG.
"""

# Importing necessary libraries
import warnings
from src.C_Pre_Processing import scaler, test_data, train_series, val_series, test_series
from src.D_model_architecture import ecg_model
from darts.models import TCNModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.B_graphs import abnormal_data, df
from darts import TimeSeries
from darts.metrics import mae, rmse
from darts.ad import ForecastingAnomalyModel, NormScorer

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Checking that the TCN model loads successfully
try:
    model = TCNModel.load('../ECG5000_Dataset/Model.pth.tar')  # Replace with your actual model file path
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

# Prepare and scale the abnormal data for anomaly detection
abnormal_features_scaled = scaler.transform(abnormal_data)
abnormal_data_scaled = pd.DataFrame(abnormal_features_scaled, columns=test_data.columns)

# Convert the scaled abnormal data to a TimeSeries object
abnormal_series = TimeSeries.from_dataframe(abnormal_data_scaled)
abnormal_series = abnormal_series.astype(np.float32)

# Instantiate the anomaly detection model with the TCN model and selected scorers
anomaly_model = ForecastingAnomalyModel(
    model=ecg_model,
    scorer=[NormScorer(ord=1)],  # L1 norm for scoring
)

# Fit the anomaly model with training data
START = 0.1
anomaly_model.fit(train_series, start=START, allow_model_training=False, verbose=True)

# Calculate and plot anomaly scores on the test series
anomaly_scores, model_forecasting = anomaly_model.score(
    test_series, start=START, return_model_prediction=True, verbose=True
)
pred_start = model_forecasting.start_time()

# Extract anomaly scores and time index for plotting
time_index = anomaly_scores.time_index
scores = anomaly_scores.values()

# Plot the anomaly scores over time
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores')
plt.legend()
plt.show()

# Calculate and print MAE and RMSE for the model's predictions
print("MAE:", mae(model_forecasting, test_series))
print("RMSE:", rmse(model_forecasting, test_series))

# Repeat anomaly scoring on the validation dataset
val_anomaly_scores, val_model_forecasting = anomaly_model.score(
    val_series, start=START, return_model_prediction=True, verbose=True
)

# Calculate z-scores for the anomaly scores to detect outliers
mean_score = np.mean(scores)
std_dev_score = np.std(scores)
z_scores = (scores - mean_score) / std_dev_score

# Define a threshold for z-scores to detect anomalies
threshold = 3
anomalies = z_scores > threshold

# Verify the shapes and types of variables used in plotting
print(f"Type of time_index: {type(time_index)}")
print(f"Shape of time_index: {np.shape(time_index)}")
print(f"Type of scores: {type(scores)}")
print(f"Shape of scores: {np.shape(scores)}")
print(f"Type of anomalies: {type(anomalies)}")
print(f"Shape of anomalies: {np.shape(anomalies)}")

# Ensure all relevant data is in numpy array format for consistency
time_index = np.array(time_index)
scores = np.array(scores)
anomalies = np.array(anomalies)

# Flatten the scores and anomalies arrays to 1D for easier handling
scores = scores.flatten()
anomalies = anomalies.flatten()

# Plot the anomaly scores with detected anomalies highlighted
plt.figure(figsize=(12, 6))
plt.plot(time_index, scores, label="Anomaly Scores")
plt.scatter(time_index[anomalies], scores[anomalies], color='red', label="Detected Anomalies")
plt.xlabel('Time')
plt.ylabel('Score')
plt.title('Anomaly Scores with Detected Anomalies')
plt.legend()
plt.show()

# Print the indices where anomalies are detected
print("Anomaly indices:", np.where(anomalies)[0])

# Define a chunk size for chunk-based analysis
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

