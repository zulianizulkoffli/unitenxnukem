# -*- coding: utf-8 -*-
"""
Created on Sat May 24 23:12:54 2025

@author: zzulk
"""

# app.py
import streamlit as st
from PIL import Image
import torch
from torchvision import models, transforms
import cv2
import numpy as np

# Load MobileNetV2
model = models.mobilenet_v2(pretrained=True)
model.eval()

# Image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load ImageNet class labels
LABELS_URL = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
imagenet_classes = list(map(str.strip, requests.get(LABELS_URL).text.split("\n")))

# Streamlit UI
st.title("🧠 Smart CNN Image Classifier (MobileNetV2)")
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    input_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)
        confidence, predicted_class = torch.max(prob, 0)

    st.success(f"Prediction: {imagenet_classes[predicted_class]} ({confidence.item():.2f})")
