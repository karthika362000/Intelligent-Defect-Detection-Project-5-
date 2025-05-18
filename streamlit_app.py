# Updated SolarGuard App: Using CNN for Classification

import os
import tempfile
import cv2
import numpy as np
import streamlit as st
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model
from PIL import Image
import zipfile
import random
import matplotlib.pyplot as plt
import joblib

st.set_page_config(page_title="SolarGuard - CNN Defect Detection", layout="centered")
st.title("🔆 SolarGuard - CNN-Based Solar Panel Defect Detection")

st.markdown("""
This app allows you to:
1. Choose between Classification and Object Detection tasks.
2. Upload six folders (Bird-Drop, Clean, Dusty, Electrical-Damage, Physical-Damage, Snow-Covered) as ZIP files.
3. Train a CNN classification model on a sample of the dataset.
4. Upload a test image to detect the panel's condition.
""")

# ---- MENU ----
task = st.sidebar.selectbox("Select Task", ["Classification", "Object Detection (Optional)"])

# ---- COMMON CONFIG ----
category_names = ["Bird-Drop", "Clean", "Dusty", "Electrical-Damage", "Physical-Damage", "Snow-Covered"]
MAX_IMAGES_PER_CLASS = 30
IMG_SIZE = (128, 128)
MODEL_FILE = "cnn_solar_model.h5"
ENCODER_FILE = "cnn_label_encoder.pkl"

# ---- FUNCTIONS ----
def load_images_from_multiple_zips(zip_dict):
    X, y = [], []
    label_encoder = LabelEncoder()
    all_labels = list(zip_dict.keys())

    with tempfile.TemporaryDirectory() as base_tmp:
        for label in all_labels:
            zip_file = zip_dict[label]
            extract_path = os.path.join(base_tmp, label.replace(" ", "_"))
            os.makedirs(extract_path, exist_ok=True)
            with zipfile.ZipFile(zip_file, 'r') as zip_ref:
                zip_ref.extractall(extract_path)

            img_paths = []
            for root, _, files in os.walk(extract_path):
                for img_file in files:
                    img_path = os.path.join(root, img_file)
                    img_paths.append(img_path)

            random.shuffle(img_paths)
            img_paths = img_paths[:MAX_IMAGES_PER_CLASS]

            for img_path in img_paths:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    img = cv2.resize(img, IMG_SIZE)
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    X.append(img)
                    y.append(label)
                except:
                    continue

    X = np.array(X)
    if X.size == 0:
        raise ValueError("No valid images found in the uploaded ZIP files.")
    y = label_encoder.fit_transform(y)
    y = to_categorical(y)
    return X, y, label_encoder

def build_cnn_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def highlight_defect(image):
    fig, ax = plt.subplots()
    ax.imshow(image)
    h, w, _ = image.shape
    x, y, box_w, box_h = w//4, h//4, w//3, h//3
    rect = plt.Rectangle((x, y), box_w, box_h, edgecolor='red', facecolor='none', lw=2)
    ax.add_patch(rect)
    ax.text(x, y-10, 'Defect Detected', color='red', fontsize=12)
    st.pyplot(fig)

# ---- CLASSIFICATION ----
if task == "Classification":
    st.header("🧠 Upload & Train the CNN Model")
    category_files = {}
    for category in category_names:
        uploaded = st.file_uploader(f"Upload ZIP for '{category}'", type="zip", key=category)
        if uploaded:
            category_files[category] = uploaded

    if st.button("Train CNN Model"):
        if len(category_files) < len(category_names):
            st.error("Please upload all six category ZIP files.")
        else:
            with st.spinner("Training CNN model on sampled data..."):
                try:
                    X, y, label_encoder = load_images_from_multiple_zips(category_files)
                    X = X / 255.0
                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                    model = build_cnn_model(input_shape=(128, 128, 3), num_classes=len(category_names))
                    model.fit(X_train, y_train, epochs=10, batch_size=16, validation_split=0.1)
                    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
                    model.save(MODEL_FILE)
                    joblib.dump(label_encoder, ENCODER_FILE)
                    st.success("✅ CNN Model trained and saved successfully!")
                except Exception as e:
                    st.error(f"Training failed: {e}")

    st.header("📸 Predict Solar Panel Condition")
    uploaded_file = st.file_uploader("Upload a test image", type=["jpg", "jpeg", "png"], key="predict")
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)

        if os.path.exists(MODEL_FILE) and os.path.exists(ENCODER_FILE):
            model = load_model(MODEL_FILE)
            label_encoder = joblib.load(ENCODER_FILE)

            img = np.array(image.resize(IMG_SIZE)) / 255.0
            img = img.reshape(1, 128, 128, 3)
            pred = model.predict(img)
            pred_label = label_encoder.inverse_transform([np.argmax(pred)])[0]

            st.subheader(f"🔍 Detected Condition: {pred_label}")

            recommendation = {
                "Clean": "✅ No action needed.",
                "Dusty": "🧹 Schedule cleaning to improve efficiency.",
                "Bird-Drop": "🧽 Clean panel to prevent long-term stains.",
                "Electrical-Damage": "⚡ Immediate inspection by technician required.",
                "Physical-Damage": "🔧 Consider panel replacement or repair.",
                "Snow-Covered": "❄ Remove snow to restore performance."
            }

            st.info(recommendation.get(pred_label, "No recommendation available."))

# ---- OBJECT DETECTION MOCK ----
elif task == "Object Detection (Optional)":
    st.header("📦 Object Detection - Locate Issues (Simulated)")
    uploaded_file = st.file_uploader("Upload a test image for object detection", type=["jpg", "jpeg", "png"], key="detect")

    if uploaded_file:
        image = Image.open(uploaded_file)
        np_img = np.array(image.resize(IMG_SIZE))
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.subheader("🕵 Defect Highlighted")
        highlight_defect(np_img)
        st.info("⚠ Simulated bounding box shown. Real detection model can be integrated for live predictions.")