# utils.py
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image, ImageOps

def load_and_prepare_image(img_path):
    img = Image.open(img_path).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)               # MNIST digits are white on black
    img = img.resize((28, 28))               # Resize to match MNIST

    img_array = np.array(img) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)

    return img_array
