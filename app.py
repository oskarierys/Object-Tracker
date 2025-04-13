import streamlit as st
import cv2
import numpy as np
import torch 
import tempfile
from PIL import Image
import os
import io
import sys
import time

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importing necessary modules from YOLOv5 repository
from modules.detector import load_model, process_image
from modules.utils import process_video, save_uploaded_file
from modules.visualization import plot_boxes, create_object_chart

# Setting page configuration
st.set_page_config(
    page_title="YOLOv5 Object Detection",
    page_icon=":guardsman:",
    layout="wide",
    initial_sidebar_state="expanded"
)
            
def main():
    st.title("Objeckt Tracker")
    st.subheader("Real-time Object Detection and Tracking")

    with st.spinnner("Loading model..."):
        model = load_model()
        st.success("Model loaded successfully!")
    
    tab1, tab2, tab3 = st.tabs(["Image Upload", "Video Upload", "About"])

    with tab1:
        st.header("Image Object Detection")
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

        sample_images_dir = os.path.join('assets', 'sample_images')

        if os.path.exists(sample_images_dir) and os.listdir(sample_images_dir):
            sample_options = ['None'] + os.listdir(sample_images_dir)
            sample_selection = st.selectbox("Or select a sample image", sample_options)

            if sample_selection != 'None':
                sample_path = os.path.join(sample_images_dir, sample_selection)
                if os.path.exists(sample_path):
                    uploaded_image = open(sample_path, 'rb')

        conf_threshold = st.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

        col1, col2 = st.columns(2)

        if uploaded_image is not None:
            # Reading image
            image = Image.open(uploaded_image)

            # Display image 
            with col1:
                st.subheader("Original image")
                st.image(image, use_column_width=True)

            # Processing image
            with st.spinner("Detecting objects..."):
                results = process_image(image, model)

                # Filtering by confidence
                results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] >= conf_threshold]

                # Bounding boxes
                img_with_boxes, detection_df = plot_boxes(results, image)

            # Displaying processsed image
            with col2:
                st.subheader("Processed image")
                st.image(img_with_boxes, use_column_width=True)

            st.subheader("Detection Results")
            if not detection_df.empty:
                st.dataframe(detection_df, use_container_width=True)

                st.success(f"Detected {len(detection_df)} objects in the image!")

                chart = create_object_chart(detection_df)
                st.plotly_chart(chart, use_container_width=True)
            else:
                st.warning("No objects detected with the selected confidence threshold.")
    
    with tab2:
        st.header("Video Object Detection")
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"])

        video_conf_threshold = st.slider("Video confidence threshold", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

        if uploaded_video is not None:
            # Saving uploaded video
            temp_video_path = save_uploaded_file(uploaded_video)

            if st.button("Process video"):
                with st.spinner("Processing video..."):
                    start_time = time.time()
                    processed_video_path, video_stats = process_video(temp_video_path, model, video_conf_threshold)
                    processing_time = time.time() - start_time

                # Displaying processed video
                with open(processed_video_path, 'rb') as video_file:
                    video_bytes = video_file.read()
                    st.video(video_bytes)

                # Status
                st.subheader("Detection statistics")
                st.write(f"Total object detected: {video_stats['total_objects']}")
                st.write(f"Processing time: {processing_time:.2f} seconds")

                # Object distribution
                st.subheader("Object Distribution")
                import pandas as pd
                class_names = list(video_stats['class_counts'].keys())
                class_counts = list(video_stats['class_counts'].values())

                # Creating a bar chart
                chart_data = {
                    'Class': class_names,
                    'Count': class_counts
                }

                chart_df = pd.DataFrame(chart_data)
                chart = create_object_chart(chart_df, x_column='Class', y_column='Count', title="Object Distribution in Video")
                st.plotly_chart(chart, use_container_width=True)

    with tab3:
        st.header("About Object-Tracker")
        st.markdown("""
        ### Project Overview
        Object-Tracker is a computer vision application that demonstrates real-time object detection using the YOLOv5 model. 
        It can identify objects in both images and videos with high accuracy.
        
        ### Technical Features
        - **AI Model**: Uses YOLOv5s pre-trained on COCO dataset (80 common object classes)
        - **Backend**: Built with PyTorch and OpenCV
        - **Interface**: Interactive Streamlit web application
        - **Processing**: Supports both image and video processing
        
        ### How It Works
        1. The system loads a pre-trained YOLOv5 model
        2. For uploaded media, it performs object detection on each image/frame
        3. Detected objects are highlighted with bounding boxes and labels
        4. For videos, it processes each frame and generates a new video with annotations
        5. Results are displayed with confidence scores and statistics
        
        ### Applications
        - Security surveillance
        - Traffic monitoring
        - Retail analytics
        - Automated inspection
        - Wildlife monitoring
        """)
        
        st.info("This project demonstrates skills in computer vision, deep learning, and interactive application development.")