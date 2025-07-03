# utils/preprocessing.py

import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_and_preprocess_images(data_dir, img_size=(128, 128), normalize=True):
    """
    Loads and preprocesses images from subfolders in data_dir.
    Assumes each subfolder represents a class.
    
    Returns:
        X: numpy array of images
        y: list of labels
        class_names: sorted list of class names
    """
    X = []
    y = []
    class_names = sorted(os.listdir(data_dir))

    for class_name in class_names:
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        for file in tqdm(os.listdir(class_path), desc=f"Processing {class_name}"):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(class_path, file)
                img = cv2.imread(img_path)
                img = cv2.resize(img, img_size)
                if normalize:
                    img = img / 255.0  # Normalize pixel values to 0-1
                X.append(img)
                y.append(class_name)

    # Encode labels to integers
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    return np.array(X), np.array(y_encoded), label_encoder.classes_


def get_data_augmentor():
    """
    Returns an ImageDataGenerator instance with common augmentations.
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
