import streamlit as st
import torch
import cv2
import tempfile
import os

st.set_page_config(page_title="YOLOv5 Video Detection", layout="wide")
st.title("📦 YOLOv5 Object Detection on Video (Offline + Streamlit)")

@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video:
    temp_video = tempfile.NamedTemporaryFile(delete=False)
    temp_video.write(uploaded_video.read())

    cap = cv2.VideoCapture(temp_video.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        result_frame = results.render()[0]
        result_rgb = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)

        stframe.image(result_rgb, channels="RGB", use_column_width=True)

    cap.release()
    os.unlink(temp_video.name)
