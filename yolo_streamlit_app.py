import streamlit as st
import torch
import cv2
import tempfile
import os

st.title("📹 YOLOv5 Object Detection - Offline Mode")

@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "yolov5s", trust_repo=True)

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
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
        annotated = results.render()[0]
        stframe.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), channels="RGB")

    cap.release()
    os.unlink(video_path)
