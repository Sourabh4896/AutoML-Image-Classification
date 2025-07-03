# utils/models.py

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2, ResNet50

def build_custom_cnn(input_shape=(128, 128, 3), num_classes=2):
    """
    Custom CNN architecture for image classification.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D(2, 2),
        
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_mobilenetv2(input_shape=(128, 128, 3), num_classes=2):
    """
    Transfer Learning using MobileNetV2
    """
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False  # freeze base model

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def build_resnet50(input_shape=(128, 128, 3), num_classes=2):
    """
    Transfer Learning using ResNet50
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    return model


def get_model(model_name, input_shape, num_classes):
    """
    Factory function to return model based on user selection.
    """
    if model_name == "Custom CNN":
        return build_custom_cnn(input_shape, num_classes)
    elif model_name == "MobileNetV2":
        return build_mobilenetv2(input_shape, num_classes)
    elif model_name == "ResNet50":
        return build_resnet50(input_shape, num_classes)
    else:
        raise ValueError(f"Model '{model_name}' not supported.")
