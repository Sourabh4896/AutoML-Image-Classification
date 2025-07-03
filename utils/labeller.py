# utils/labeller.py

import os

def get_class_labels(data_dir: str) -> dict:
    """
    Generate label mapping from folder names in dataset.
    Returns a dictionary like: {'cat': 0, 'dog': 1}
    """
    folders = sorted([f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))])
    label_map = {folder: idx for idx, folder in enumerate(folders)}
    return label_map

def count_images_per_class(data_dir: str) -> dict:
    """
    Count number of images in each class folder.
    """
    counts = {}
    for folder in os.listdir(data_dir):
        folder_path = os.path.join(data_dir, folder)
        if os.path.isdir(folder_path):
            counts[folder] = len([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    return counts
