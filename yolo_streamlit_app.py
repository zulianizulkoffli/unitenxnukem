import streamlit as st
import cv2
import numpy as np
import tempfile
import os

# Load YOLOv2 model
@st.cache_resource
def load_yolov2_model():
    net = cv2.dnn.readNetFromDarknet("yolov2.cfg", "yolov2.weights")
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    return net

net = load_yolov2_model()
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

# UI
st.title("🚀 YOLOv2 Object Detection (OpenCV + Streamlit)")
uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi"])

if uploaded_video:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_video.read())

    cap = cv2.VideoCapture(tfile.name)
    stframe = st.empty()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        height, width = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        outs = net.forward(output_layers)

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype("int")
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    label = f"Class {class_id}: {confidence:.2f}"
                    cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

        stframe.image(frame, channels="BGR")
    cap.release()
