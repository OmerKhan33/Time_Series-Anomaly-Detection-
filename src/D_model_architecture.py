from darts.models import TCNModel
from matplotlib import pyplot as plt
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.callbacks import Callback
from src.C_Pre_Processing import train_series, val_series
from torch.optim import Adam

# Model Architecture
class LossLoggingCallback(Callback):
    def _init_(self):
        self.metrics = {"epochs": [], "train_loss": [], "val_loss": []}
        self.train_losses = []
        self.val_losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        train_loss = trainer.callback_metrics.get("train_loss", None)
        if train_loss is not None:
            train_loss = train_loss.item()
            self.train_losses.append(train_loss)
            print(f"Train epoch end: recorded train loss {train_loss}")

    def on_validation_epoch_end(self, trainer, pl_module):
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


# Define the TCN model
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

# Define the EarlyStopping callback
early_stopping_callback = EarlyStopping(
    monitor="val_loss",  # Metric to monitor
    patience=10,  # Number of epochs to wait for improvement
    mode="max"  # Mode should be 'min' for loss metrics
)

# Create the loss logging callback instance
loss_callback = LossLoggingCallback()

# Initialize the trainer with callbacks
trainer = Trainer(
    callbacks=[early_stopping_callback, loss_callback],
    max_epochs=100,
    logger=True,
    enable_progress_bar=True
)

# %%
ecg_model.fit(train_series, val_series=val_series, trainer=trainer)

# %%
# Make sure to have matching lengths for epochs, train_losses, and val_losses
epochs = range(len(loss_callback.train_losses))

# Ensure that the lengths match
num_train_epochs = len(loss_callback.train_losses)
num_val_epochs = len(loss_callback.val_losses)

# Adjust val_losses if it has more entries than train_losses
if num_val_epochs > num_train_epochs:
    loss_callback.val_losses = loss_callback.val_losses[:num_train_epochs]

# Plot losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, loss_callback.train_losses, label='Train Loss', marker='o')
plt.plot(epochs, loss_callback.val_losses, label='Validation Loss', marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss Over Epochs')
plt.legend()
plt.grid(True)
plt.show()