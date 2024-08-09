import os
from v3 import validation_series
from v3 import train_series
import pandas as pd
import json
import matplotlib.pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback
from darts.models import TCNModel
from pytorch_lightning.callbacks import EarlyStopping
from torch.optim import Adam, SGD
import warnings

warnings.filterwarnings('ignore')

train_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TRAIN.txt", delimiter='\s+', header=None)
test_df = pd.read_csv("../ECG5000_Dataset/ECG5000_TEST.txt", delimiter='\s+', header=None)


class LossLoggingCallback(Callback):
    def __init__(self, log_file='../ECG5000_Dataset/training_metrics.json'):
        self.log_file = log_file
        self.metrics = {"epochs": [], "train_loss": [], "val_loss": []}
        self.train_losses = []
        self.val_losses = []
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)

    def on_train_epoch_end(self, trainer, pl_module):
        # Debugging statement
        print(f"Train epoch end: {trainer.callback_metrics}")
        train_loss = trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            print(f"Train epoch end: recorded train loss {train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Debugging statement
        print(f"Validation epoch end: {trainer.callback_metrics}")
        val_loss = trainer.callback_metrics.get("val_loss", None)
        if val_loss is not None:
            val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            print(f"Validation epoch end: recorded validation loss {val_loss}")

            # Append new metrics
            epoch = trainer.current_epoch
            self.metrics["epochs"].append(epoch)
            self.metrics["train_loss"].append(self.train_losses[-1] if self.train_losses else None)
            self.metrics["val_loss"].append(val_loss)

            # Save updated metrics
            try:
                with open(self.log_file, 'w') as f:
                    json.dump(self.metrics, f)
            except IOError as e:
                print(f"Error writing to file {self.log_file}: {e}")

# Ensure your model has proper training/validation loss metrics being logged
# Example of training the model (make sure to pass the correct arguments):

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=5,          # Number of epochs to wait for improvement
    mode="min"           # Mode should be 'min' for loss metrics
)

ecg_model = TCNModel(
    input_chunk_length=50,
    output_chunk_length=30,
    kernel_size=3,
    num_filters=32,
    num_layers=3,
    dropout=0.2,
    optimizer_cls=Adam,
    optimizer_kwargs={"lr": 0.001},
    random_state=42
)

loss_callback = LossLoggingCallback(log_file='../ECG5000_Dataset/training_metrics.json')



trainer = Trainer(
    callbacks=[early_stopping_callback, loss_callback],
    max_epochs=50,
    logger=True,
    enable_progress_bar=True
)
ecg_model.fit(train_series, val_series=validation_series, trainer=trainer)

# Make sure to check the log file for content after training

def plot_val_loss(log_file='../ECG5000_Dataset/training_metrics.json'):
    # Load metrics from the JSON file
    try:
        with open(log_file, 'r') as f:
            metrics = json.load(f)
    except IOError as e:
        print(f"Error reading the file {log_file}: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON from the file {log_file}: {e}")
        return

    # Extract epochs and validation loss
    epochs = metrics.get("epochs", [])
    val_loss = metrics.get("val_loss", [])
    train_loss = metrics.get("train_loss",[])

    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, val_loss,train_loss, marker='o', linestyle='-', color='b')
    plt.xlabel('Epochs')
    plt.ylabel('Validation Loss')
    plt.ylabel('train loss')
    plt.title('Validation Loss vs. Epochs')
    plt.grid(True)
    plt.xticks(range(min(epochs), max(epochs) + 1, 1))  # Adjust x-ticks if needed
    plt.show()

# Call the function to plot
plot_val_loss()























