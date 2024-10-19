# Save this code as app.py and run it using `streamlit run app.py`

import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2
from PIL import Image

# Load the pre-trained SSD MobileNet V2 model from TensorFlow Hub
model_url = 'https://tfhub.dev/tensorflow/ssd_mobilenet_v2/fpnlite_320x320/1'
model = hub.load(model_url)

# Load labels for the COCO dataset used by the SSD MobileNet V2 model
LABELS = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 
    'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
    'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
    'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'TV', 'laptop', 'mouse', 
    'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
    'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

# Streamlit UI setup
st.title("Object Detection with SSD MobileNet V2")
st.write("Upload an image and the model will identify objects and draw bounding boxes.")

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def detect_objects(image):
    # Convert the image to a tensor and add batch dimension
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    # Perform detection
    detections = model(input_tensor)

    # Extract detection data
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy().astype(int)
    detection_scores = detections['detection_scores'][0].numpy()

    return detection_boxes, detection_classes, detection_scores

def draw_boxes(image, boxes, classes, scores, threshold=0.5):
    img_with_boxes = image.copy()
    height, width, _ = image.shape

    for i in range(len(boxes)):
        if scores[i] >= threshold:
            # Get box coordinates
            box = boxes[i]
            ymin, xmin, ymax, xmax = box
            xmin, xmax, ymin, ymax = int(xmin * width), int(xmax * width), int(ymin * height), int(ymax * height)

            # Draw the bounding box
            cv2.rectangle(img_with_boxes, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)

            # Label with class name
            label = LABELS[int(classes[i])]  # Adjust class index
            score = scores[i]
            cv2.putText(img_with_boxes, f'{label}: {score:.2f}', (xmin, ymin - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return img_with_boxes

if uploaded_file is not None:
    # Load and display the uploaded image
    image = load_image(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Detect objects
    boxes, classes, scores = detect_objects(image)

    # Draw bounding boxes on the image
    result_image = draw_boxes(image, boxes, classes, scores)

    # Display the result image
    st.image(result_image, caption='Detected Objects', use_column_width=True)
