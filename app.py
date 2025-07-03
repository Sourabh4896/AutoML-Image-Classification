# app.py

import streamlit as st
import os
from utils.labeller import get_class_labels, count_images_per_class
from PIL import Image
import random

# ---------------------- CONFIG -------------------------
st.set_page_config(page_title="AutoML Image Classifier", layout="wide")

# ---------------------- TITLE -------------------------
st.title("🧠 AutoML Image Classifier – Step 1: Input Dataset Path & Label")

# ---------------------- INPUT PATH -------------------------
data_dir = st.text_input("📁 Enter the full path to your dataset folder (folder should contain subfolders for each class):")

if data_dir and os.path.exists(data_dir):
    st.success("✅ Folder path is valid!")

    # Display label mapping
    st.subheader("📛 Auto-Generated Labels")
    labels = get_class_labels(data_dir)
    st.write(labels)

    # Show image counts per class
    st.subheader("🖼️ Image Count Per Class")
    counts = count_images_per_class(data_dir)
    st.write(counts)

    # Preview some images
    st.subheader("🔍 Sample Images from Each Class")
    for label in list(labels.keys())[:3]:  # Show first 3 classes
        folder = os.path.join(data_dir, label)
        files = [f for f in os.listdir(folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if files:
            sample_file = random.choice(files)
            image_path = os.path.join(folder, sample_file)
            st.image(Image.open(image_path), caption=f"{label}/{sample_file}", width=200)

else:
    if data_dir:
        st.error("❌ The provided folder path does not exist. Please check and try again.")
