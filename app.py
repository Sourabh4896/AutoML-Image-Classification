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



from utils.preprocessing import load_and_preprocess_images

if st.button("Preprocess Now"):
    with st.spinner("Loading and preprocessing images..."):
        X, y, label_map = load_and_preprocess_images(data_dir)
        st.session_state.X = X
        st.session_state.y = y
        st.session_state.label_map = label_map
        st.success(f"✅ Loaded {len(X)} images with {len(label_map)} classes.")



from utils.splitter import split_dataset

st.subheader("📊 Step 3: Dataset Splitting")
test_size = st.slider("Test set size (%)", 10, 40, 20) / 100.0
val_size = st.slider("Validation set size (%)", 5, 30, 10) / 100.0

if st.button("Split Dataset"):
    if 'X' in st.session_state and 'y' in st.session_state:
        X = st.session_state.X
        y = st.session_state.y

        with st.spinner("Splitting dataset..."):
            X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
                X, y, test_size=test_size, val_size=val_size)

            st.session_state.X_train = X_train
            st.session_state.X_val = X_val
            st.session_state.X_test = X_test
            st.session_state.y_train = y_train
            st.session_state.y_val = y_val
            st.session_state.y_test = y_test

            st.success("✅ Dataset split successfully!")
            st.write(f"Training Samples: {len(X_train)}")
            st.write(f"Validation Samples: {len(X_val)}")
            st.write(f"Test Samples: {len(X_test)}")
    else:
        st.warning("⚠️ Please preprocess the dataset before splitting.")


from utils.models import get_custom_cnn, get_mobilenet_model, get_resnet_model

st.subheader("🧠 Step 4: Model Selection")

model_type = st.selectbox("Choose Model", ["Custom CNN", "MobileNetV2", "ResNet50"])
loss_fn = st.selectbox("Choose Loss Function", ["sparse_categorical_crossentropy", "categorical_crossentropy"])
optimizer_name = st.selectbox("Choose Optimizer", ["Adam", "SGD", "RMSprop"])

input_shape = (128, 128, 3)
num_classes = len(st.session_state.label_map)

if st.button("Build Model"):
    if model_type == "Custom CNN":
        model = get_custom_cnn(input_shape, num_classes, loss_fn, optimizer_name)
    elif model_type == "MobileNetV2":
        model = get_mobilenet_model(input_shape, num_classes, loss_fn, optimizer_name)
    elif model_type == "ResNet50":
        model = get_resnet_model(input_shape, num_classes, loss_fn, optimizer_name)

    st.session_state.model = model
    st.success(f"✅ {model_type} built and compiled successfully.")
    st.text(model.summary())


from utils.train import train_model, plot_training_history

st.subheader("🛠️ Step 6: Train Your Model")

epochs = st.slider("Select Epochs", 1, 50, 10)
batch_size = st.slider("Batch Size", 8, 128, 32)

if st.button("Start Training"):
    required_keys = ['model', 'X_train', 'y_train', 'X_val', 'y_val']
    if all(k in st.session_state for k in required_keys):
        model = st.session_state.model
        X_train = st.session_state.X_train
        y_train = st.session_state.y_train
        X_val = st.session_state.X_val
        y_val = st.session_state.y_val

        with st.spinner("Training in progress..."):
            history = train_model(model, X_train, y_train, X_val, y_val, batch_size=batch_size, epochs=epochs)
            st.session_state.history = history
            st.success("✅ Training completed!")

        st.subheader("📈 Training Metrics")
        fig_acc, fig_loss = plot_training_history(history)
        st.pyplot(fig_acc)
        st.pyplot(fig_loss)

    else:
        st.error("❌ Please build the model and split dataset before training.")


from utils.evaluate import evaluate_model, plot_confusion_matrix, plot_roc_curve

st.subheader("📊 Step 7: Model Evaluation")

if st.button("Evaluate Model"):
    required_keys = ['model', 'X_test', 'y_test', 'label_map']
    if all(k in st.session_state for k in required_keys):
        model = st.session_state.model
        X_test = st.session_state.X_test
        y_test = st.session_state.y_test
        label_map = st.session_state.label_map

        with st.spinner("Evaluating model..."):
            acc, prec, rec, f1, cm, report = evaluate_model(model, X_test, y_test, label_map)

            st.success("✅ Evaluation completed.")
            st.markdown(f"**Accuracy:** `{acc:.4f}`")
            st.markdown(f"**Precision:** `{prec:.4f}`")
            st.markdown(f"**Recall:** `{rec:.4f}`")
            st.markdown(f"**F1 Score:** `{f1:.4f}`")

            st.text("Classification Report:")
            st.code(report)

            st.subheader("📌 Confusion Matrix")
            fig_cm = plot_confusion_matrix(cm, label_map)
            st.pyplot(fig_cm)

            st.subheader("📈 ROC Curve (if binary classification)")
            probs = model.predict(X_test)
            fig_roc = plot_roc_curve(y_test, probs)
            if fig_roc:
                st.pyplot(fig_roc)
            else:
                st.info("ROC curve only available for binary classification.")

    else:
        st.warning("⚠️ Please make sure model and test data are available.")
