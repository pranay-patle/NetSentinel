import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

# Load preprocessed datasets
X_train = np.loadtxt("X_train.csv", delimiter=",")
y_train = np.loadtxt("y_train.csv", delimiter=",")
X_test = np.loadtxt("X_test.csv", delimiter=",")
y_test = np.loadtxt("y_test.csv", delimiter=",")

# Define the model
model = Sequential([
    Dense(16, activation="relu", input_shape=(X_train.shape[1],)),
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
