# utils/models.py

from tensorflow.keras import models, layers, optimizers
from tensorflow.keras.applications import MobileNetV2, ResNet50
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mobilenet_preprocess
from tensorflow.keras.applications.resnet50 import preprocess_input as resnet_preprocess

def get_custom_cnn(input_shape, num_classes, loss_fn, optimizer_name):
    """
    Returns a custom CNN model compiled with chosen loss and optimizer.
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
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=get_optimizer(optimizer_name), loss=loss_fn, metrics=["accuracy"])
    return model

def get_mobilenet_model(input_shape, num_classes, loss_fn, optimizer_name):
    """
    Returns a transfer learning model using MobileNetV2.
    """
    base_model = MobileNetV2(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(optimizer=get_optimizer(optimizer_name), loss=loss_fn, metrics=["accuracy"])
    return model

def get_resnet_model(input_shape, num_classes, loss_fn, optimizer_name):
    """
    Returns a transfer learning model using ResNet50.
    """
    base_model = ResNet50(include_top=False, weights='imagenet', input_shape=input_shape)
    base_model.trainable = False

    model = models.Sequential([
        base_model,
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer=get_optimizer(optimizer_name), loss=loss_fn, metrics=["accuracy"])
    return model

def get_optimizer(name):
    """
    Returns an optimizer object based on name.
    """
    name = name.lower()
    if name == "adam":
        return optimizers.Adam()
    elif name == "sgd":
        return optimizers.SGD()
    elif name == "rmsprop":
        return optimizers.RMSprop()
    else:
        raise ValueError(f"Unsupported optimizer: {name}")
