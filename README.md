# 🌱 Plant Health Prediction Using CNN

A deep learning-based plant health classification system that uses a Convolutional Neural Network (CNN) to classify plant leaf images into three categories:

- Healthy
- Powdery
- Rust

## 📌 Project Overview

This project uses image preprocessing and a CNN model built with TensorFlow/Keras to automatically identify the health condition of plant leaves.

The trained model is integrated with a Streamlit web application, allowing users to upload a plant leaf image and receive a predicted class with confidence.

## 🚀 Features

- Image-based plant health classification
- CNN-based deep learning model
- Three-class classification
- Training and validation performance analysis
- Confusion matrix and classification report
- Streamlit web interface
- Image upload and real-time prediction

## 🧠 Model Architecture

The CNN consists of:

- Rescaling layer
- 3 Convolutional layers
- Max Pooling layers
- Flatten layer
- Dense layer
- Dropout layer
- Softmax output layer

## 📊 Results

The model achieved approximately **87% test accuracy** on the test dataset.

### Classes

| Class | Description |
|---|---|
| Healthy | Healthy plant leaf |
| Powdery | Powdery disease |
| Rust | Rust disease |

## 🛠️ Technologies Used

- Python
- TensorFlow / Keras
- CNN
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit

## 📂 Project Structure

```text
plant_health_cnn/
│
├── Train/
├── Test/
├── Validation/
│
├── train_model.py
├── predict.py
├── app.py
│
├── plant_health_cnn.keras
├── accuracy_graph.png
├── loss_graph.png
├── confusion_matrix.png
├── leaf.png
├── README.md
└── .gitignore