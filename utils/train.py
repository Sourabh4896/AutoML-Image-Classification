# utils/train.py

import matplotlib.pyplot as plt
import streamlit as st
from tensorflow.keras.callbacks import Callback


class StreamlitTrainingProgress(Callback):
    def __init__(self, total_epochs):
        super().__init__()
        self.total_epochs = total_epochs
        self.progress_bar = st.progress(0)
        self.status_text = st.empty()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        progress = int((epoch + 1) / self.total_epochs * 100)
        self.progress_bar.progress(progress)
        self.status_text.text(
            f"Epoch {epoch+1}/{self.total_epochs} - "
            f"loss: {logs.get('loss'):.4f}, "
            f"val_loss: {logs.get('val_loss'):.4f}, "
            f"acc: {logs.get('accuracy'):.4f}, "
            f"val_acc: {logs.get('val_accuracy'):.4f}"
        )


def train_model(model, X_train, y_train, X_val, y_val, batch_size=32, epochs=10):
    """
    Trains the model and returns the history.
    Also updates Streamlit progress bar live.
    """
    callback = StreamlitTrainingProgress(total_epochs=epochs)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        verbose=0,  # Turn off terminal output
        callbacks=[callback]
    )
    return history


def plot_training_history(history):
    """
    Returns accuracy and loss plots.
    """
    fig_acc, ax1 = plt.subplots()
    ax1.plot(history.history['accuracy'], label='Train Accuracy')
    ax1.plot(history.history['val_accuracy'], label='Val Accuracy')
    ax1.set_title('Model Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True)

    fig_loss, ax2 = plt.subplots()
    ax2.plot(history.history['loss'], label='Train Loss')
    ax2.plot(history.history['val_loss'], label='Val Loss')
    ax2.set_title('Model Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True)

    return fig_acc, fig_loss
