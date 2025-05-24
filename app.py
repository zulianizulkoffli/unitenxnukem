
import streamlit as st
import numpy as np
from PIL import Image

st.title("CNN-Based Image & Video Classification")

st.markdown("This is a placeholder Streamlit app for image and video classification using a CNN model.")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((64, 64))
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Prediction: [Placeholder]")
