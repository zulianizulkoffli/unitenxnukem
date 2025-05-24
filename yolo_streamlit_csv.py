# -*- coding: utf-8 -*-
"""
Created on Sat May 24 19:28:02 2025

@author: zzulk
"""

import streamlit as st
import torch
import cv2
import pandas as pd
import tempfile

st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("YOLOv5 Object Detection on Thermal Video with CSV Log")

video_file = st.file_uploader("Upload a thermal video (.mp4)", type=["mp4"])

if video_file:
    # Save uploaded video to a temporary file
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    video_path = tfile.name

    # Load YOLOv5 pre-trained model (uses COCO dataset)
    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

    # Read video
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()

    detection_data = []
    frame_idx = 0

    while cap.isOpened() and frame_idx < 150:  # Limit for preview
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame)

        # Extract predictions
        labels = results.xyxyn[0][:, -1].numpy()
        cords = results.xyxyn[0][:, :-1].numpy()

        for i, row in enumerate(cords):
            x1, y1, x2, y2, conf = row
            cls = int(labels[i])
            detection_data.append({
                "frame": frame_idx,
                "class": model.names[cls],
                "confidence": round(conf, 3),
                "x1": round(x1, 4),
                "y1": round(y1, 4),
                "x2": round(x2, 4),
                "y2": round(y2, 4)
            })

        # Show rendered frame
        rendered_frame = results.render()[0]
        stframe.image(rendered_frame, channels="BGR", use_column_width=True)
        frame_idx += 1

    cap.release()

    # Export detection results to CSV
    df = pd.DataFrame(detection_data)
    csv = df.to_csv(index=False).encode("utf-8")
    st.success("Detection complete. Download your CSV log below.")
    st.download_button("Download Detection Log CSV", data=csv, file_name="yolo_detection_log.csv", mime="text/csv")
