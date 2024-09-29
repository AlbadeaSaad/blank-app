import streamlit as st
import pandas as pd
import math
from PIL import Image
import numpy as np
from ultralytics import YOLO
import cv2

# Set the title and favicon that appear in the Browser's tab bar.
st.set_page_config(
    page_title='YOLO DETECTOR',
    page_icon=':earth_americas:', # This is an emoji shortcode. Could be a URL too.
)

# -----------------------------------------------------------------------------

# Load the model
model = YOLO("best.pt")

# Streamlit app
st.title("Image Classification with PyTorch")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open the uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    # Convert to numpy array
    img_np = np.array(image)

    # Run inference
    results = model(img_np)

    # Display the results
    #st.image(image, caption='Uploaded Image', use_column_width=True)

    # Visualize results
    annotated_frame = results[0].plot()  # Get the annotated image
    st.image(annotated_frame, caption='Detection Results', use_column_width=True)
