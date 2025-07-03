# utils/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def evaluate_model(model, X_test, y_test, class_names, use_sparse_labels=True):
    """
    Evaluates the model on test data and returns predictions and metrics.
    """
    y_probs = model.predict(X_test)

    # Get predictions
    y_pred = np.argmax(y_probs, axis=1)

    # Convert one-hot to int if needed
    if not use_sparse_labels:
        y_test = np.argmax(y_test, axis=1)

    report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    return y_test, y_pred, report, cm


def plot_confusion_matrix(cm, class_names):
    """
    Plots confusion matrix using seaborn.
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    return plt.gcf()


def plot_roc_auc(model, X_test, y_test):
    """
    Plots ROC curve for binary classification only.
    """
    if len(np.unique(y_test)) != 2:
        return None  # ROC only applies to binary classification

    y_score = model.predict(X_test)[:, 1]
    fpr, tpr, thresholds = roc_curve(y_test, y_score)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label="AUC = %0.2f" % roc_auc)
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    return plt.gcf()
