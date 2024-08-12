"""
This script trains a Temporal Convolutional Network (TCN) model on ECG time series data using the Darts library and PyTorch Lightning.

1. **Model Architecture and Training Setup:**
   - Defines a `LossLoggingCallback` to log and display training and validation losses at the end of each epoch.
   - Initializes a TCN model with specified hyperparameters such as input and output chunk lengths, kernel size, number of filters, layers, dropout rate, and optimizer settings.
   - Sets up an EarlyStopping callback to halt training if validation loss does not improve for a specified number of epochs.
   - Configures a PyTorch Lightning Trainer to manage the training process with the defined callbacks.

2. **Model Training:**
   - Fits the TCN model on the training dataset (`train_series`) with validation data (`val_series`) using the defined trainer.

3. **Loss Visualization:**
   - After training, plots the training and validation losses over epochs to visualize the model's learning process.

4. **Model Saving:**
   - Saves the trained TCN model to a specified file path.

The code includes necessary configurations and callbacks for efficient model training and evaluation, along with plotting to monitor performance.
"""

from darts.models import TCNModel
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, Callback
from src.C_Pre_Processing import train_series, val_series
from torch.optim import Adam

# Define a callback for logging losses
class LossLoggingCallback(Callback):
    def __init__(self):
        self.metrics = {"epochs": [], "train_loss": [], "val_loss": []}
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Log training loss at the end of each epoch
        train_loss = trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            print(f"Train epoch end: recorded train loss {train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
        # Log validation loss at the end of each epoch
        val_loss = trainer.callback_metrics.get("val_loss", None)
        print("Validation Epoch End Callback Triggered")  # Debugging Line
        if val_loss is not None:
            val_loss = val_loss.item()
            self.val_losses.append(val_loss)
            print(f"Validation epoch end: recorded validation loss {val_loss}")

            # Append new metrics
            epoch = trainer.current_epoch
            self.metrics["epochs"].append(epoch)
            self.metrics["train_loss"].append(self.train_losses[-1] if self.train_losses else None)
            self.metrics["val_loss"].append(val_loss)


# Define the TCN model with specified parameters
ecg_model = TCNModel(
    input_chunk_length=300,  # Length of the input sequence
    output_chunk_length=30,  # Length of the output sequence
    kernel_size=3,  # Size of the convolutional kernel
    num_filters=64,  # Number of filters in the convolutional layers
    num_layers=3,  # Number of convolutional layers
    dropout=0.1,  # Dropout rate to prevent overfitting
    optimizer_cls=Adam,  # Optimizer to use
    optimizer_kwargs={"lr": 0.001},  # Learning rate for the optimizer
    random_state=42  # Seed for random number generation
)

# Define the EarlyStopping callback to halt training if validation loss doesn't improve
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor for early stopping
    patience=5,  # Number of epochs to wait for improvement
    mode="min"  # Mode should be 'min' since we are minimizing the loss
)

# Create the loss logging callback instance
loss_callback = LossLoggingCallback()

# Initialize the PyTorch Lightning Trainer
trainer = Trainer(
    callbacks=[early_stopping_callback, loss_callback],  # List of callbacks to use
    max_epochs=100,  # Maximum number of epochs for training
    logger=True,  # Enable logging
    enable_progress_bar=True  # Show progress bar
)

# Train the model on the training data with validation
ecg_model.fit(train_series, val_series=val_series, trainer=trainer)

# Ensure matching lengths for epochs, train_losses, and val_losses
epochs = range(len(loss_callback.train_losses))

num_train_epochs = len(loss_callback.train_losses)
num_val_epochs = len(loss_callback.val_losses)

# Adjust val_losses if it has more entries than train_losses
if num_val_epochs > num_train_epochs:
    loss_callback.val_losses = loss_callback.val_losses[:num_train_epochs]

# Plot training and validation losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_callback.train_losses, label='Train Loss', marker='o')
plt.plot(epochs, loss_callback.val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()

# Save the trained model to a file
ecg_model.save("../ECG5000_Dataset/Model.pth.tar")
