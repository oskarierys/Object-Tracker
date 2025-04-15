import cv2
import numpy as np
import tempfile
import streamlit as st
import os
from PIL import Image
import time 
import modules.visualization as plot_boxes

def save_uploaded_file(uploaded_file):
    """
    Save an uploaded file to a temporary location
    
    Args:
        uploaded_file: Streamlit uploaded file object
        
    Returns:
        str: Path to the saved file
    """

    # Create a temporary file with the same extension
    file_extension = os.path.splitext(uploaded_file.name)[1]
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
    temp_file.write(uploaded_file.read())
    temp_file_path = temp_file.name
    temp_file.close()
    
    return temp_file_path

def process_video(video_path, model, conf_threshold=0.5):
    """
    Process a video file and detect objects in each frame
    
    Args:
        video_file (str): Path to the video file
        model: YOLOv5 model
        confidence_threshold (float): Minimum confidence threshold for detections
        
    Returns:
        tuple: (Path to processed video, Statistics dictionary)
    """

    # Create a temporary file to store the processed video
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    temp_filename = temp_file.name
    temp_file.close()
    
    # Read the input video
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Set up video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_filename, fourcc, fps, (width, height))
    
    # Process each frame
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create a progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Container for stats
    stats = {'total_objects': 0, 'classes': {}}
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(progress)
        status_text.text(f"Processing frame {frame_count+1} / {total_frames}")
        
        # Detect objects
        results = model(frame)
        
        # Apply confidence threshold
        results.xyxy[0] = results.xyxy[0][results.xyxy[0][:, 4] >= conf_threshold]
        
        # Draw boxes on the frame
        annotated_frame, df = plot_boxes(results, frame)
        
        # Update stats
        stats['total_objects'] += len(df)
        for _, row in df.iterrows():
            class_name = row['name']
            if class_name in stats['classes']:
                stats['classes'][class_name] += 1
            else:
                stats['classes'][class_name] = 1
        
        # Write the frame to the output video
        out.write(annotated_frame)
        
        frame_count += 1
    
    # Release resources
    cap.release()
    out.release()
    
    # Return the path to the processed video and stats
    return temp_filename, stats

def ensure_directories_exist():
    """
    Create necessary directories if they don't exist
    """
    directories = [
        'assets',
        'assets/sample_images',
        'modules'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)