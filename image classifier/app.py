# Save this code as app.py and run it using `streamlit run app.py`

import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from PIL import Image, ImageOps
import numpy as np

# Load the pre-trained MobileNetV2 model
model = MobileNetV2(weights='imagenet')

# Set up the Streamlit app title and description
st.title("Image Detector with MobileNetV2")
st.write("Upload an image, and the model will identify and detect the objects in the image.")

# Function to preprocess the uploaded image
def preprocess_image(image):
    image = ImageOps.fit(image, (224, 224), Image.ANTIALIAS)  # Resize to 224x224
    img_array = np.array(image)  # Convert to numpy array
    img_array = preprocess_input(img_array)  # Preprocess the image as required by MobileNetV2
    img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension
    return img_array

# File uploader for the user to upload an image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open and display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")

    # Preprocess the image for prediction
    img_array = preprocess_image(image)

    # Make prediction
    predictions = model.predict(img_array)
    decoded_predictions = decode_predictions(predictions, top=3)[0]

    # Display the top 3 predictions
    st.write("Top Predictions:")
    for i, (imagenet_id, label, score) in enumerate(decoded_predictions):
        st.write(f"{i+1}: {label} ({score*100:.2f}%)")
