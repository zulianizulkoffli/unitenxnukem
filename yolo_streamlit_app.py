# -*- coding: utf-8 -*-
"""
Created on Sat May 24 21:14:28 2025

@author: zzulk
"""

import streamlit as st
import torch
import tempfile
import pandas as pd
from PIL import Image
import numpy as np
import io

st.set_page_config(page_title="YOLOv5 Video Detection", layout="centered")
st.title("YOLOv5 Video Object Detection with CSV Export")

video_file = st.file_uploader("Upload a .mp4 video", type=["mp4"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Load YOLOv5 model
    model = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)

    # Defer opencv import to inside function
    import cv2
    cap = cv2.VideoCapture(video_path)

    stframe = st.empty()
    frame_idx = 0
    detection_log = []

    while cap.isOpened() and frame_idx < 150:  # Short preview
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)
        rendered = results.render()[0]
        rgb_frame = cv2.cvtColor(rendered, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        stframe.image(pil_image, caption=f"Frame {frame_idx}", use_column_width=True)

        labels = results.xyxyn[0][:, -1].numpy()
        cords = results.xyxyn[0][:, :-1].numpy()
        for i, row in enumerate(cords):
            x1, y1, x2, y2, conf = row
            cls = int(labels[i])
            detection_log.append({
                "frame": frame_idx,
                "class": model.names[cls],
                "confidence": round(conf, 3),
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "x2": round(x2, 4),
                "y2": round(y2, 4)
            })
        frame_idx += 1

    cap.release()

    df = pd.DataFrame(detection_log)
    csv = df.to_csv(index=False).encode("utf-8")
    st.success("Detection completed! Download your results below.")
    st.download_button("Download CSV Log", data=csv, file_name="yolo_detections.csv", mime="text/csv")