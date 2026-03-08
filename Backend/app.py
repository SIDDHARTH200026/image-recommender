import streamlit as st
import numpy as np
import torch
from PIL import Image
st.title("🖼️ Image Recommendation Engine")
st.write("Upload image for similar item recommendations")

uploaded_file = st.file_uploader("Choose image...", type=['png','jpg','jpeg'])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.success("✅ Model ready - recommendations would appear here!")
