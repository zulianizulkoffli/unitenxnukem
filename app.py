
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import tempfile

st.title("CNN-Based Image & Video Classification")

# Load a dummy CNN model (replace with your actual model)
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("cnn_model.h5")  # Replace with your model path
    return model

model = load_model()

def preprocess_image(image):
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    return np.expand_dims(img_array, axis=0)

def predict_image(image):
    preprocessed = preprocess_image(image)
    prediction = model.predict(preprocessed)
    return prediction

def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (224, 224))
        frame = frame / 255.0
        frame = np.expand_dims(frame, axis=0)
        pred = model.predict(frame)
        predictions.append(pred)
    cap.release()
    return predictions

uploaded_file = st.file_uploader("Upload Image or Video", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file is not None:
    if uploaded_file.type.startswith("image"):
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        prediction = predict_image(image)
        st.write("Prediction:", prediction)
    elif uploaded_file.type == "video/mp4":
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        predictions = predict_video(tfile.name)
        st.video(tfile.name)
        st.write(f"Processed {len(predictions)} frames.")
        st.write("Predictions (sample):", predictions[:3])
