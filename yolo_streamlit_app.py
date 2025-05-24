import streamlit as st
import torch
import tempfile
import os
import cv2
import numpy as np

st.title("Offline YOLOv5 Object Detection on Video")

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='local')

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
        annotated_frame = np.squeeze(results.render())

        stframe.image(annotated_frame, channels="BGR")

    cap.release()
    os.unlink(video_path)
