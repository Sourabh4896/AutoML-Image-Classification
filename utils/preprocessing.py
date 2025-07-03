# utils/preprocessing.py

import os
import numpy as np
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_and_preprocess_images(data_dir, image_size=(128, 128), normalize=True):
    """
    Load all images from subfolders, resize, normalize (optional).
    Returns: numpy arrays X (images), y (labels), label_map
    """
    X = []
    y = []
    label_map = {}
    class_folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])

    for idx, folder in enumerate(class_folders):
        label_map[folder] = idx
        folder_path = os.path.join(data_dir, folder)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                img_path = os.path.join(folder_path, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue  # Skip unreadable files
                img = cv2.resize(img, image_size)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                if normalize:
                    img = img / 255.0
                X.append(img)
                y.append(idx)

    return np.array(X), np.array(y), label_map

def get_augmentor():
    """
    Returns a Keras ImageDataGenerator for augmentation
    """
    datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.15,
        horizontal_flip=True,
        fill_mode="nearest"
    )
    return datagen

