import streamlit as st
from PIL import Image
import torch
import pandas as pd
import tempfile
import os

st.title("YOLO Object Detection with Logging")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp:
        temp.write(uploaded_file.read())
        img_path = temp.name

    image = Image.open(img_path)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    model = torch.hub.load('ultralytics/yolov5', 'yolov5s', trust_repo=True)
    results = model(img_path)

    # Convert to pandas DataFrame
    df = results.pandas().xyxy[0]
    st.write(df)

    # Save to CSV
    csv_path = os.path.join("detection_log.csv")
    df.to_csv(csv_path, index=False)
    st.success("Detections logged to CSV!")

    with open(csv_path, "rb") as f:
        st.download_button("Download CSV Log", f, file_name="detections.csv")
