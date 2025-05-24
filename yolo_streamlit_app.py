import streamlit as st
import torch
import cv2
import tempfile
import os
import numpy as np
from PIL import Image

st.title("🎥 Offline YOLOv5 Object Detection on Video")

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'custom', path='yolov5s.pt', source='local')

model = load_model()

uploaded_video = st.file_uploader("📤 Upload a video", type=["mp4", "avi", "mov"])
if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        annotated_frame = np.squeeze(results.render())

        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", use_column_width=True)

    cap.release()
    os.remove(tfile.name)
