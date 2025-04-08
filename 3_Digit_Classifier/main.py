# main.py
import tensorflow as tf
import numpy as np
from model import create_model


def load_local_mnist(path="mnist.npz"):
    with np.load(path) as data:
        x_train = data["x_train"]
        y_train = data["y_train"]
        x_test = data["x_test"]
        y_test = data["y_test"]
    return (x_train, y_train), (x_test, y_test)

def main():
    # Load MNIST data
    (x_train, y_train), (x_test, y_test) = load_local_mnist("mnist.npz")

    # Normalize and reshape
    x_train = x_train / 255.0
    x_test = x_test / 255.0
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    model = create_model()
    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model.save("digit_model.h5")

    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"✅ Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()
