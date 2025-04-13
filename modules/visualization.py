import cv2
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def plot_boxes(results, image):
    """
    Draw bounding boxes and labels on the image
    
    Args:
        results: YOLOv5 results object
        image (PIL.Image or numpy.ndarray): Input image
        
    Returns:
        tuple: (Image with boxes, DataFrame with detection results)
    """

    if isinstance(image, Image.Image):
        image = np.array(image)

    # Get detection dataframe
    df = results.pandas().xyxy[0]

    # Creating a copy of the image to draw 
    img_with_boxes = image.copy()

    # Draw bounding boxes and labels on the image
    for index, row in df.iterrows():
        x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
        conf = row['confidence']
        label = f"{row['name']} {conf:.2f}"

        # Generate color based on class name
        colour_seed = sum(ord(c) for c in row['name'])
        colour = (colour_seed * 123) % 255, (colour_seed * 45) % 255, (colour_seed * 67) % 255

        # Draw rectangle and label
        cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(img_with_boxes, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colour, 2)

    # Format the DataFrame for display
    if not df.empty:
        display_df = df.copy()
        display_df = display_df.rename(columns={'xmin': 'Xmin', 'ymin': 'Ymin', 'xmax': 'Xmax', 'ymax': 'Ymax', 'confidence': 'Confidence', 'class': 'Class ID', 'name': 'Object'})
        display_df['Confidence'] = display_df['Confidence'].round(3)
        display_df = display_df[['Object', 'Confidence', 'X Min', 'Y Min', 'X Max', 'Y Max']]
    else:
        display_df = df

    return img_with_boxes, display_df

def create_object_chart(detection_df, x_column='Object', y_column=None):
    """
    Create a bar chart of detected objects
    
    Args:
        detection_df (pandas.DataFrame): DataFrame with detection results
        x_column (str): Column name for x-axis
        y_column (str, optional): Column name for y-axis values
        
    Returns:
        plotly.graph_objects.Figure: Plotly figure object
    """
    if y_column is None:
        # Count occurrences of each object
        counts = detection_df[x_column].value_counts().reset_index()
        counts.columns = [x_column, 'Count']
        
        fig = px.bar(
            counts, 
            x=x_column, 
            y='Count',
            color=x_column,
            title="Detected Objects",
            labels={x_column: 'Object Class', 'Count': 'Number Detected'}
        )
    else:
        fig = px.bar(
            detection_df,
            x=x_column,
            y=y_column,
            color=x_column,
            title="Detected Objects",
            labels={x_column: 'Object Class', y_column: 'Number Detected'}
        )
    
    fig.update_layout(
        xaxis_title='Object Class',
        yaxis_title='Count',
        showlegend=False,
        template='plotly_white'
    )
    
    return fig