import tensorflow as tf
import numpy as np
import cv2  # OpenCV for image loading/resizing
import os

# Load the trained model
model = tf.keras.models.load_model("/Users/sharyudilipmagre/Internship Project/Plant_model_CNN.keras")

# Classes (must match training)
classes = ['Black_rot', 'ESCA', 'Healthy', 'Leaf_Blight']

# Image parameters
img_height = 128
img_width = 128

def preprocess_image(image_path):
    """
    Load an image, resize, normalize, and expand dims for model input.
    """
    img = cv2.imread(image_path)               # Read image
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Convert BGR to RGB
    img = cv2.resize(img, (img_width, img_height))
    img = img / 255.0                          # Normalize
    img = np.expand_dims(img, axis=0)          # Add batch dimension
    return img

def predict(image_path):
    img = preprocess_image(image_path)
    predictions = model.predict(img)
    class_idx = np.argmax(predictions)
    confidence = predictions[0][class_idx]
    print(f"Predicted class: {classes[class_idx]} with confidence {confidence:.2f}")

# Example usage: predicting a single image
test_image_path = "/Users/sharyudilipmagre/Desktop/Example_data/0a06c482-c94a-44d8-a895-be6fe17b8c06___FAM_B.Rot 5019_flipLR.JPG"  # Replace with your image path
predict(test_image_path)
