# utils/train.py

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

def compile_model(model, loss_name, optimizer_name):
    """
    Compiles the given model with specified loss and optimizer.
    """
    # Select optimizer
    optimizers = {
        "Adam": tf.keras.optimizers.Adam(),
        "SGD": tf.keras.optimizers.SGD(),
        "RMSprop": tf.keras.optimizers.RMSprop()
    }

    # Select loss
    losses = {
        "categorical_crossentropy": "categorical_crossentropy",
        "sparse_categorical_crossentropy": "sparse_categorical_crossentropy",
        "binary_crossentropy": "binary_crossentropy"
    }

    model.compile(
        optimizer=optimizers[optimizer_name],
        loss=losses[loss_name],
        metrics=['accuracy']
    )
    return model


def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, use_sparse_labels=True):
    """
    Trains the model and returns training history.
    """
    # If using categorical_crossentropy, convert labels to one-hot
    if not use_sparse_labels:
        y_train = tf.keras.utils.to_categorical(y_train)
        y_val = tf.keras.utils.to_categorical(y_val)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=1
    )
    return history


def plot_training_history(history):
    """
    Plots loss and accuracy graphs using matplotlib.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    # Accuracy
    axs[0].plot(history.history['accuracy'], label='Train Acc')
    axs[0].plot(history.history['val_accuracy'], label='Val Acc')
    axs[0].set_title("Accuracy over Epochs")
    axs[0].set_xlabel("Epoch")
    axs[0].set_ylabel("Accuracy")
    axs[0].legend()
    axs[0].grid()

    # Loss
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Val Loss')
    axs[1].set_title("Loss over Epochs")
    axs[1].set_xlabel("Epoch")
    axs[1].set_ylabel("Loss")
    axs[1].legend()
    axs[1].grid()

    return fig
