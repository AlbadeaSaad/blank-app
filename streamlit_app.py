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
uploaded_file = st.file_uploader('choose a video...', type=['mp4'])

if uploaded_file is not None:
    # Get the file name
    file_name = uploaded_file.name

cap = cv2.VideoCapture(file_name+".mp4")
w, h, fps = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FPS)

width = 640
height = 480

# Create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Replace with your desired codec
out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (width, height))

while cap.isOpened():
    # Read a frame from the video
    success, frame = cap.read()

    if success:
        # Run YOLOv8 tracking on the frame, persisting tracks between frames
        results = model.track(frame, persist=False)
        out.write(results)
out.release()
cap.release()
#uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

#if uploaded_file is not None:
    # Open the uploaded image
    #image = Image.open(uploaded_file).convert("RGB")
    
    # Convert to numpy array
    #img_np = np.array(image)

    # Run inference
    #results = model(img_np)

    # Display the results
    #st.image(image, caption='Uploaded Image', use_column_width=True)

    # Visualize results
    #annotated_frame = results[0].plot()  # Get the annotated image
#video_bytes = out.read()
st.video(out)
