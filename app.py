import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_health_cnn.keras")

# Class names
class_names = ["Healthy", "Powdery", "Rust"]

# Page configuration
st.set_page_config(
    page_title="Plant Health Prediction",
    page_icon="🌱"
)

st.title("🌱 Plant Health Prediction")
st.write("Upload a plant leaf image to predict its health condition.")

# Upload image
uploaded_file = st.file_uploader(
    "Choose a plant leaf image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", width=400)

    # Preprocess image
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized)
    image_array = np.expand_dims(image_array, axis=0)

    # Prediction
    prediction = model.predict(image_array)

    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    # Display result
    st.subheader("Prediction Result")

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")
    