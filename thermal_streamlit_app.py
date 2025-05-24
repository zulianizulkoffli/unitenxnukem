import streamlit as st
import numpy as np
import cv2
import tempfile
from PIL import Image
from tensorflow.keras.models import load_model

# Load the model
@st.cache_resource
def load_cnn_model():
    return load_model("cnn-mnist-model.h5")  # change this if needed

model = load_cnn_model()

# Prediction function
def predict_image(img_array):
    img_array = img_array.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # (1, H, W, C)
    pred = model.predict(img_array)
    return np.argmax(pred), np.max(pred)

# Streamlit UI
st.title("🧠 CNN-Based Image & Video Classification")
st.markdown("Upload an image or a video to classify digits using a CNN model.")

option = st.radio("Select input type:", ["Image", "Video"])

if option == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("L").resize((28, 28))
        st.image(image, caption="Uploaded Image", use_column_width=True)
        image_np = np.array(image).reshape((28, 28, 1))
        label, confidence = predict_image(image_np)
        st.success(f"🧾 Prediction: {label} with confidence {confidence:.2f}")

elif option == "Video":
    uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
    if uploaded_video is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_video.read())

        cap = cv2.VideoCapture(tfile.name)
        stframe = st.empty()
        frame_count = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or frame_count > 30:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame_resized = cv2.resize(frame, (28, 28)).reshape((28, 28, 1))
            label, confidence = predict_image(frame_resized)
            display_frame = cv2.putText(cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR),
                                        f"Pred: {label}", (5, 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            stframe.image(display_frame, channels="BGR")
            frame_count += 1
        cap.release()
