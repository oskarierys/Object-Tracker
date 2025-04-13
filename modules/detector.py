import torch 
import numpy as np
from PIL import Image
import streamlit as st

@st.cache_resource
def load_model():
    """
    Load YOLOv5 model from PyTorch Hub
    
    Returns:
        torch.nn.Module: Loaded YOLOv5 model
    """
    try: 
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        model.eval()  # Set the model to evaluation mode
        return model
    except Exception as e:
        raise Exception(f"Error loading model: {e}")
    
def process_image(image, model):
    """
    Detect objects in an image using YOLOv5 model

    Args:
        image (PIL.Image or np.ndarray): Input image
        model (torch.nn.Module): YOLOv5 model

    Returns:
        Object: YOLOv5 results object
    """

    # Check if the image is a numpy array and convert it to PIL Image if necessary
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Run inference
    results = model(image)
    return results

def get_classes():
    """ 
    Get the list of classes that the YOLOv5 model can detect
    
    Returns:
        list: List of class names
    """
    # COCO dataset classes
    return [
        'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat',
        'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack',
        'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
        'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
        'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse',
        'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]