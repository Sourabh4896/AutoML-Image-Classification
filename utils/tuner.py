# utils/tuner.py

import tensorflow as tf
import optuna
from utils.models import get_model
from utils.train import train_model, compile_model
from sklearn.metrics import accuracy_score


def manual_hyperparameter_config():
    """
    Returns default tunable hyperparameters with UI defaults.
    """
    return {
        "learning_rate": 0.001,
        "batch_size": 32,
        "epochs": 10,
        "dropout": 0.5
    }


def update_model_with_hyperparams(model, learning_rate, optimizer_name="Adam", loss="sparse_categorical_crossentropy"):
    """
    Compile model with user-defined learning rate and optimizer.
    """
    if optimizer_name == "Adam":
        optimizer = tf.keras.optimizers.Adam(learning_rate)
    elif optimizer_name == "SGD":
        optimizer = tf.keras.optimizers.SGD(learning_rate)
    elif optimizer_name == "RMSprop":
        optimizer = tf.keras.optimizers.RMSprop(learning_rate)
    else:
        optimizer = tf.keras.optimizers.Adam(learning_rate)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=["accuracy"]
    )
    return model


# ✅ Optional: Auto-tuning using Optuna
def optuna_objective(trial, X_train, y_train, X_val, y_val, input_shape, num_classes, model_name):
    """
    Objective function for Optuna hyperparameter tuning.
    """
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-2)
    dropout = trial.suggest_uniform("dropout", 0.2, 0.7)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    epochs = trial.suggest_int("epochs", 5, 20)

    # Build model
    model = get_model(model_name, input_shape, num_classes)
    model = update_model_with_hyperparams(model, lr)

    # Modify dropout layer (only for Custom CNN)
    if model_name == "Custom CNN":
        model.layers[-2] = tf.keras.layers.Dropout(dropout)

    history = train_model(model, X_train, y_train, X_val, y_val,
                          epochs=epochs, batch_size=batch_size, use_sparse_labels=True)

    # Evaluate accuracy on val set
    val_preds = model.predict(X_val)
    y_pred = tf.argmax(val_preds, axis=1)
    acc = accuracy_score(y_val, y_pred)

    return acc


def run_optuna_tuning(X_train, y_train, X_val, y_val, input_shape, num_classes, model_name, n_trials=10):
    """
    Run Optuna tuning and return best trial hyperparameters.
    """
    study = optuna.create_study(direction="maximize")
    study.optimize(lambda trial: optuna_objective(
        trial, X_train, y_train, X_val, y_val, input_shape, num_classes, model_name), n_trials=n_trials)

    return study.best_params, study.best_value
