import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import cv2  # OpenCV for image processing

# Load the trained model instead of retraining
model = load_model("digit_classifier_model.h5")
print("Loaded saved model.")

# Function to preprocess an image for prediction
def preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    if img is None:
        raise FileNotFoundError(f"Could not find or open {image_path}. Check file path and try again.")
    
    # Invert colors if needed (MNIST is white digits on black background)
    img = cv2.bitwise_not(img)

    # Apply thresholding to enhance contrast
    _, img = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)

    # Resize to match MNIST size
    img = cv2.resize(img, (28, 28))

    # Normalize pixel values (0-255 → 0.0-1.0)
    img = img / 255.0

    # Reshape to fit model input
    img = img.reshape(1, 28, 28)

    return img

# Test with your own image
custom_image_path = "my_digit.png"  # Make sure this file exists
img = preprocess_image(custom_image_path)

# Debugging: Check preprocessed image
print(f"Original Image Shape: {cv2.imread(custom_image_path, cv2.IMREAD_GRAYSCALE).shape}")
print(f"Processed Image Shape: {img.shape}")  # Should be (1, 28, 28)
print(f"Pixel Values Range: Min={img.min()}, Max={img.max()}")

plt.imshow(img.reshape(28, 28), cmap='gray')  # Show processed image
plt.title("Processed Image for Model")
plt.show()

# Model makes a prediction
predicted_label = np.argmax(model.predict(img), axis=-1)
print(f"Predicted Label for Custom Image: {predicted_label}")