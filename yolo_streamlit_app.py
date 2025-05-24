import streamlit as st
import torch
import cv2
import tempfile
import os
from PIL import Image

st.set_page_config(page_title="YOLOv5 Object Detection", layout="centered")
st.title("📹 Offline YOLOv5 Object Detection on Video")

@st.cache_resource
def load_model():
    try:
        model = torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt", source="local")
        st.success("Loaded local model yolov5s.pt")
    except Exception as e:
        st.warning(f"Local model not found. Downloading online model... ({e})")
        model = torch.hub.load("ultralytics/yolov5", "yolov5s")
    return model

model = load_model()

uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = results.render()[0]
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()
    os.remove(video_path)
