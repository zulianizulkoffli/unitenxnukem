# -*- coding: utf-8 -*-
"""
Created on Sat May 24 18:42:44 2025

@author: zzulk
"""

import cv2
import numpy as np
import streamlit as st
from PIL import Image

st.set_page_config(page_title="Thermal Heat Zone Detector", layout="centered")
st.title("Thermal Video Heat Zone Detection with Labels")

video_file = st.file_uploader("Upload a thermal video (.mp4)", type=["mp4"])

if video_file:
    # Save uploaded file temporarily
    with open("uploaded_video.mp4", 'wb') as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture("uploaded_video.mp4")
    stframe = st.empty()
    frame_idx = 0

    while cap.isOpened() and frame_idx < 150:  # Limit for fast preview
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        color_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # Draw and label each zone
        count = 0
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) > 50:
                x, y, w, h = cv2.boundingRect(contour)
                cv2.rectangle(color_frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(color_frame, f"Zone {i+1}", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                count += 1

        stframe.image(color_frame, channels="BGR", caption=f"Frame {frame_idx+1} | Detected Zones: {count}", use_column_width=True)
        frame_idx += 1

    cap.release()
