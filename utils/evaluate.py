# utils/evaluate.py

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score,
    recall_score, f1_score, roc_curve, auc
)
import seaborn as sns
import streamlit as st


def evaluate_model(model, X_test, y_test, label_map):
    """
    Evaluates the model and returns performance metrics and confusion matrix.
    """
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    # ✅ Convert label_map values to string target names
    target_names = [str(label_map[i]) for i in sorted(label_map.keys())]

    report = classification_report(y_test, y_pred, target_names=target_names, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)

    return accuracy, precision, recall, f1, cm, report



def plot_confusion_matrix(cm, label_map):
    """
    Plots confusion matrix using seaborn.
    """
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_map.values(), yticklabels=label_map.values())
    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')
    ax.set_title('Confusion Matrix')
    plt.tight_layout()
    return fig


def plot_roc_curve(y_test, y_pred_probs):
    """
    Plots ROC curve (binary classification only).
    """
    if y_pred_probs.shape[1] != 2:
        return None  # ROC curve is for binary only

    fpr, tpr, _ = roc_curve(y_test, y_pred_probs[:, 1])
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic')
    ax.legend(loc="lower right")
    return fig
