# predictor.py
import tensorflow as tf
from utils import load_and_prepare_image

def predict_digit(img_path):
    model = tf.keras.models.load_model("digit_model.h5")
    img_array = load_and_prepare_image(img_path)

    predictions = model.predict(img_array)
    predicted_digit = predictions.argmax()

    print(f"🔢 Predicted digit: {predicted_digit}")

# Example usage:
# predict_digit("data/your_digit.png")
