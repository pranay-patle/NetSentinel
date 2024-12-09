import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
import pandas as pd
import numpy as np

# Load preprocessed datasets using pandas
X_train = pd.read_csv("X_train.csv")
y_train = pd.read_csv("y_train.csv")
X_test = pd.read_csv("X_test.csv")
y_test = pd.read_csv("y_test.csv")

# Convert to numpy arrays (optional, but needed for TensorFlow)
X_train = X_train.to_numpy()
y_train = y_train.to_numpy().flatten()  # Flatten the y_train to be a 1D array
X_test = X_test.to_numpy()
y_test = y_test.to_numpy().flatten()  # Flatten the y_test to be a 1D array

# Define the model
model = Sequential([
    Input(shape=(X_train.shape[1],)),  # Specify input shape with Input layer
    Dense(16, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="sigmoid")  # Binary classification
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Save the trained model
model.save("iot_ids_model.h5")

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model
with open("iot_ids_model.tflite", "wb") as f:
    f.write(tflite_model)

print("Model training and conversion to TensorFlow Lite complete!")
