import pandas as pd
import tensorflow as tf
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# Load MNIST data
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize data
x_train = tf.keras.utils.normalize(x_train, axis=1)
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Build the Fully Connected (Dense) Neural Network model
model = tf.keras.models.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Increased the number of neurons
    tf.keras.layers.Dense(512, activation=tf.nn.relu),  # Another dense layer with more neurons
    tf.keras.layers.Dense(256, activation=tf.nn.relu),  # Added more layers for better learning
    tf.keras.layers.Dense(128, activation=tf.nn.relu),  # Further deepening the network
    tf.keras.layers.Dense(10, activation='softmax')     # Output layer for 10 digits
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, epochs=5)  # Increased epochs to allow better training

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy}")
print(f"Test Loss: {loss}")

# Function to process image and predict the digit
def preprocess_and_predict(image_path):
    img = cv.imread(image_path, cv.IMREAD_GRAYSCALE)  # Read as grayscale
    img = cv.resize(img, (28, 28))  # Ensure the image is resized to 28x28

    # Threshold the image to ensure it is in a consistent black-and-white format
    _, img = cv.threshold(img, 128, 255, cv.THRESH_BINARY_INV)

    # Normalize the image to the 0-1 range and reshape for prediction
    img = np.array(img).reshape(1, 28, 28, 1)  # Add the channel dimension
    img = img / 255.0  # Normalize

    prediction = model.predict(img)
    predicted_value = np.argmax(prediction)
    print("----------------")
    print(f"The predicted value is: {predicted_value}")
    print("----------------")
    plt.imshow(img[0], cmap=plt.cm.binary)
    plt.show()

# Test with your images (replace with your actual image paths)
for x in range(1, 5):
    preprocess_and_predict(f'{x}.png')