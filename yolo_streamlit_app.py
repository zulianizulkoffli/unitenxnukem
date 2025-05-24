import streamlit as st
import torch
import cv2
import tempfile
import os

st.set_page_config(page_title="Offline YOLOv5 Object Detection on Video")
st.title("🎥 Offline YOLOv5 Object Detection on Video")

@st.cache_resource
def load_model():
    return torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)

model = load_model()

uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())
    cap = cv2.VideoCapture(tfile.name)

    stframe = st.empty()

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        results = model(frame)
        annotated = results.render()[0]
        annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
        stframe.image(annotated, channels="RGB", use_column_width=True)
    
    cap.release()
    os.unlink(tfile.name)
