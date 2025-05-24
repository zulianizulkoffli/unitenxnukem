import streamlit as st
import torch
import cv2
import tempfile
import os
from PIL import Image
import numpy as np

st.set_page_config(layout="wide")
st.title("Offline YOLOv5 Object Detection on Video")

# Load YOLOv5 model (offline - local .pt file)
@st.cache_resource
def load_model():
    return torch.hub.load("ultralytics/yolov5", "custom", path="yolov5s.pt", source="local")

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])

if uploaded_video is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO detection
        results = model(frame)
        rendered_frame = results.render()[0]
        frame_rgb = cv2.cvtColor(rendered_frame, cv2.COLOR_BGR2RGB)

        # Show video in Streamlit
        stframe.image(frame_rgb, channels="RGB", use_column_width=True)

    cap.release()
    os.unlink(tfile.name)
    st.success("Video processing completed.")
