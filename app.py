import os

# Set the CUDA_VISIBLE_DEVICES environment variable
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load the trained model
model = load_model("imageclassifier.h5")

# Set up the Streamlit app
st.title("Image Classifier")

# Upload image function
def upload_image():
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        # Read and preprocess the uploaded image
        img = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)

        # Make predictions
        prediction = model.predict(img)
        if prediction[0][0] > 0.5:
            result = "Person is Sad"
        else:
            result = "Person is Happy"

        # Display the uploaded image and prediction result
        st.image(img[0], use_column_width=True)
        st.write("Prediction: ", result)

# Run the app
if __name__ == "__main__":
    upload_image()
