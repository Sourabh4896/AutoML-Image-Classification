# 🔮 AutoVision – No-Code Image Classification Made Magical

**AutoVision** is a powerful, Streamlit-based tool that empowers anyone — especially non-developers — to train, evaluate, and optimize deep learning image classifiers. No coding required. Just load your images, choose a model, and go!

---

## ✨ Key Features

- 📁 **Folder-Based Dataset Input**
- 🔖 **Automatic Labeling** from folder names
- 🧼 **Preprocessing** (Resize, Normalize, Augment)
- ✂️ **Train/Validation/Test Split**
- 🧠 **Model Selection**: Custom CNN, MobileNetV2, ResNet50
- ⚙️ **Choose Loss Functions and Optimizers**
- 🔁 **Live Model Training with Progress Bar**
- 📊 **Evaluation**: Accuracy, Precision, Recall, F1, Confusion Matrix, ROC
- 🖥️ **Streamlit UI** for smooth interaction

---

## 🗂️ Project Structure

```

AutoVision/
├── app.py               # Main Streamlit app
├── run.py               # Alternate CLI launcher
├── requirements.txt     # Python dependencies
├── .gitignore
├── README.md            # You're here!
└── utils/               # Modular backend
├── labeller.py
├── preprocessing.py
├── splitter.py
├── models.py
├── train.py
├── evaluate.py
└── tuner.py

````

---

## 🚀 Getting Started

### 1️⃣ Clone the Repo

```bash
git clone https://github.com/Sourabh4896/AutoVision.git
cd AutoVision
````

### 2️⃣ Install Requirements

```bash
pip install -r requirements.txt
```

### 3️⃣ Launch the App

```bash
streamlit run app.py
```

or

```bash
python run.py
```

---

## 🧪 Dataset Format

Your dataset must be organized like this:

```
your_dataset/
├── class_1/
│   ├── image1.jpg
│   └── ...
└── class_2/
    ├── image1.jpg
    └── ...
```

Each folder becomes a class label automatically.

---

## 🧠 Models Supported

* 🧬 Custom CNN
* ⚡ MobileNetV2
* 🛰️ ResNet50 (Transfer Learning)

---

## 📈 Evaluation Metrics

* ✅ Accuracy
* 🔍 Precision, Recall, F1-Score
* 📊 Confusion Matrix
* 📈 ROC Curve (for binary classification)
* 📉 Loss/Accuracy Training Graphs

---

## 🌟 Future Roadmap

* Export models to `.h5`, `.onnx`, `.tflite`
* Edge-device optimization (quantization, pruning)
* In-browser inference & drag-n-drop predictions
* Docker + HuggingFace integration

---

## 👨‍💻 Author

**Sourabh Pawar**
AI Engineer | Python Developer | Streamlit Enthusiast
📧 [pawarsourabh045@gmail.com](mailto:pawarsourabh045@gmail.com)

---

## 📜 License

This project is licensed under the **MIT License** — free for personal and commercial use.

---

> ✨ AutoVision: See beyond the pixels, without writing a single line of code.


