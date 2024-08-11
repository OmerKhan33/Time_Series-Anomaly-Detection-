import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from darts import TimeSeries
from darts.ad.utils import (
    eval_metric_from_binary_prediction,
    eval_metric_from_scores,
    show_anomalies_from_scores,
)
from darts.ad import (
    ForecastingAnomalyModel,
    KMeansScorer,
    NormScorer,
    WassersteinScorer,
)
from darts.dataprocessing.transformers import Scaler
from darts.datasets import TaxiNewYorkDataset
from darts.metrics import mae, rmse
from darts.models import TCNModel
from darts.ad.detectors import QuantileDetector

# Load and visualize the data
series_taxi = TaxiNewYorkDataset().load()

# Define start and end dates for some known anomalies
anomalies_day = {
    "NYC Marathon": ("2014-11-02 00:00", "2014-11-02 23:30"),
    "Thanksgiving": ("2014-11-27 00:00", "2014-11-27 23:30"),
    "Christmas": ("2014-12-24 00:00", "2014-12-25 23:30"),
    "New Years": ("2014-12-31 00:00", "2015-01-01 23:30"),
    "Snow Blizzard": ("2015-01-26 00:00", "2015-01-27 23:30"),
}
anomalies_day = {
    k: (pd.Timestamp(v[0]), pd.Timestamp(v[1])) for k, v in anomalies_day.items()
}

# Create a series with the binary anomaly flags
anomalies = pd.Series([0] * len(series_taxi), index=series_taxi.time_index)
for start, end in anomalies_day.values():
    anomalies.loc[(start <= anomalies.index) & (anomalies.index <= end)] = 1.0

series_taxi_anomalies = TimeSeries.from_series(anomalies)

# Plot the data and the anomalies
fig, ax = plt.subplots(figsize=(15, 5))
series_taxi.plot(label="Number of taxi passengers", linewidth=1, color="#6464ff")
(series_taxi_anomalies * 10000).plot(label="5 known anomalies", color="r", linewidth=1)
plt.show()

def plot_anom(selected_anomaly, delta_plotted_days):
    one_day = series_taxi.freq * 24 * 2
    anomaly_date = anomalies_day[selected_anomaly][0]
    start_timestamp = anomaly_date - delta_plotted_days * one_day
    end_timestamp = anomaly_date + (delta_plotted_days + 1) * one_day

    series_taxi[start_timestamp:end_timestamp].plot(
        label="Number of taxi passengers", color="#6464ff", linewidth=0.8
    )
    (series_taxi_anomalies[start_timestamp:end_timestamp] * 10000).plot(
        label="Known anomaly", color="r", linewidth=0.8
    )
    plt.title(selected_anomaly)
    plt.show()

for anom_name in anomalies_day:
    plot_anom(anom_name, 3)
    break  # remove this to see all anomalies

# Split the data into training and testing sets
s_taxi_train = series_taxi[:4500]
s_taxi_test = series_taxi[4500:]

# Add covariates (hour and day of the week)
add_encoders = {
    "cyclic": {"future": ["hour", "dayofweek"]},
}

# One week corresponds to (7 days * 24 hours * 2) of 30 minutes
one_week = 7 * 24 * 2

# Train a TCN forecasting model
forecasting_model = TCNModel(
    input_chunk_length=one_week,
    output_chunk_length=1,
    kernel_size=2,
    num_filters=3,
    dropout=0.1,
    weight_norm=True,
    add_encoders=add_encoders,
    n_epochs=100,
)
forecasting_model.fit(s_taxi_train)

# Instantiate the anomaly model with: one fitted model, and 3 scorers
half_a_day = 2 * 12
full_day = 2 * 24

anomaly_model = ForecastingAnomalyModel(
    model=forecasting_model,
    scorer=[
        NormScorer(ord=1),
        WassersteinScorer(window=half_a_day, window_agg=False),
        WassersteinScorer(window=full_day, window_agg=True),
    ],
)

# Fit the anomaly model
START = 0.1
anomaly_model.fit(s_taxi_train, start=START, allow_model_training=False, verbose=True)

# Compute anomaly scores on the test set
anomaly_scores, model_forecasting = anomaly_model.score(
    s_taxi_test, start=START, return_model_prediction=True, verbose=True
)

pred_start = model_forecasting.start_time()
print(
    "On testing set -> MAE: {}, RMSE: {}".format(
        mae(model_forecasting, s_taxi_test), rmse(model_forecasting, s_taxi_test)
    )
)

# Plot the data and the anomalies
fig, ax = plt.subplots(figsize=(15, 5))
s_taxi_test.plot(label="Number of taxi passengers")
model_forecasting.plot(label="Prediction of the model", linewidth=0.9)
plt.show()

# Evaluate the anomaly model
metric_names = ["AUC_ROC", "AUC_PR"]
metric_data = []
for metric_name in metric_names:
    metric_data.append(
        anomaly_model.eval_metric(
            anomalies=series_taxi_anomalies,
            series=s_taxi_test,
            start=START,
            metric=metric_name,
        )
    )
pd.DataFrame(data=metric_data, index=metric_names).T

# Visualize the results
anomaly_model.show_anomalies(
    series=s_taxi_test,
    anomalies=series_taxi_anomalies[pred_start:],
    start=START,
    metric="AUC_ROC",
)

# Use a QuantileDetector to convert anomaly scores to binary predictions
contamination = 0.95
detector = QuantileDetector(high_quantile=contamination)

# Use the anomaly score that gave the best AUC ROC score: Wasserstein anomaly score with a window of 'full_day'
best_anomaly_score = anomaly_scores[-1]

# Fit and detect on the anomaly scores, it will return a binary prediction
anomaly_pred = detector.fit_detect(series=best_anomaly_score)

# Plot the binary prediction
fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True)
anomaly_pred.plot(label="Prediction", ax=ax1)
series_taxi_anomalies[anomaly_pred.start_time() :].plot(
    label="Known anomalies", ax=ax2, color="red"
)
fig.tight_layout()

for metric_name in ["accuracy", "precision", "recall", "f1"]:
    metric_val = detector.eval_metric(
        pred_scores=best_anomaly_score,
        anomalies=series_taxi_anomalies,
        window=full_day,
        metric=metric_name,
    )
    print(metric_name + f": {metric_val:.2f}/1")

# Internal methods for evaluation and visualization
windows = [1, half_a_day, full_day]
scorer_names = [f"{scorer}_{w}" for scorer, w in zip(anomaly_model.scorers, windows)]

metric_data = {"AUC_ROC": [], "AUC_PR": []}
for metric_name in metric_data:
    metric_data[metric_name] = eval_metric_from_scores(
        anomalies=series_taxi_anomalies,
        pred_scores=anomaly_scores,
        window=windows,
        metric=metric_name,
    )

pd.DataFrame(index=scorer_names, data=metric_data)

# Visualize the anomalies with pre-computed scores
show_anomalies_from_scores(
    series=s_taxi_test,
    anomalies=series_taxi_anomalies[pred_start:],
    pred_scores=anomaly_scores,
    pred_series=model_forecasting,
    window=windows,
    title="Anomaly results using a forecasting method",
    names_of_scorers=scorer_names,
    metric="AUC_ROC",
)

# Zoom in on each anomaly
def plot_anom_eval(selected_anomaly, delta_plotted_days):
    one_day = series_taxi.freq * 24 * 2
    anomaly_date = anomalies_day[selected_anomaly][0]
    start = anomaly_date - one_day * delta_plotted_days
    end = anomaly_date + one_day * (delta_plotted_days + 1)

    # Input series and forecasts
    series_taxi[start:end].plot(
        label="Number of taxi passengers", color="#6464ff", linewidth=0.8
    )
    model_forecasting[start:end].plot(
        label="Model prediction", color="green", linewidth=0.8
    )

    # Actual anomalies and predicted scores
    (series_taxi_anomalies[start:end] * 10000).plot(
        label="Known anomaly", color="r", linewidth=0.8
    )
    # Scaler transforms scores in [0, 1] for better visualization
    scaler = Scaler()
    best_anomaly_score_scaled = scaler.fit_transform(best_anomaly_score)
    (best_anomaly_score_scaled[start:end] * 5000).plot(
        label="Anomaly score", color="purple", linewidth=0.8
    )

    plt.title(selected_anomaly)
    plt.show()

for anom_name in anomalies_day:
    plot_anom_eval(anom_name, 3)