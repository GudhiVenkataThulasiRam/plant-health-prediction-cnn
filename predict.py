import tensorflow as tf
import numpy as np
from PIL import Image

# Load trained model
model = tf.keras.models.load_model("plant_health_cnn.keras")

# Class names
class_names = ["Healthy", "Powdery", "Rust"]

# Give the path of the image here
image_path = "leaf.png"

# Load and preprocess image
image = Image.open(image_path).convert("RGB")
image = image.resize((128, 128))

# Convert image to array
image_array = np.array(image)
image_array = np.expand_dims(image_array, axis=0)

# Make prediction
prediction = model.predict(image_array)

# Get predicted class
predicted_class = class_names[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print("Prediction:", predicted_class)
print("Confidence: {:.2f}%".format(confidence))