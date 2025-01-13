import os
import tensorflow as tf
import numpy as np
from scapy.all import sniff, wrpcap
from scapy.layers.inet import IP, TCP, UDP
from datetime import datetime
import joblib
import time

# Suppress TensorFlow and CUDA warnings
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Load the TensorFlow H5 model
try:
    model = tf.keras.models.load_model("iot_ids_model.h5", compile=False)
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

# Load the scaler
try:
    scalar = joblib.load("standard_scaler.pkl")
except Exception as e:
    print(f"Error loading scaler: {e}")
    exit()

# Define real-time packet feature extraction
def extract_features(packet):
    """
    Extract features from a packet in real time.
    """
    try:
        # Extract features
        features = {
            "Protocol Type": packet[IP].proto if IP in packet else 0,
            "Rate": len(packet) / (time.time() - extract_features.start_time) if hasattr(extract_features, "start_time") else 0,
            "Srate": len(packet) / (time.time() - extract_features.start_time) if hasattr(extract_features, "start_time") else 0,
            "Drate": len(packet) / (time.time() - extract_features.last_packet_time) if hasattr(extract_features, "last_packet_time") else 0,
            "TCP": 1 if TCP in packet else 0,
            "UDP": 1 if UDP in packet else 0,
            "Tot size": len(packet),
            "IAT": time.time() - extract_features.last_packet_time if hasattr(extract_features, "last_packet_time") else 0,
        }

        # Update timing variables
        extract_features.start_time = time.time()
        extract_features.last_packet_time = time.time()

        # Return feature values
        return [features[key] for key in [
            "Protocol Type", "Rate", "Srate", "Drate", "TCP", "UDP", "Tot size", "IAT"
        ]]
    except Exception as e:
        print(f"Error extracting features: {e}")
        return [0] * 8  # Return zeros if feature extraction fails

# Initialize timing variables
extract_features.start_time = time.time()
extract_features.last_packet_time = time.time()

# Real-time detection
def detect_packet(packet):
    """
    Detect suspicious activity in real-time traffic.
    """
    try:
        # Extract features
        features = np.array([extract_features(packet)], dtype=np.float32)

        # Scale features
        features_scaled = scalar.transform(features)
        #print("Scaled Features: ", features_scaled)

        # Predict
        prediction = model.predict(features_scaled, verbose=0)
        print(f"Prediction: {prediction}")

        # Check threshold
        if prediction[0] > 0.99:  # Adjust threshold based on testing
            print("Suspicious activity detected!")
            capture_traffic()
        else:
            print("No suspicious activity detected.")

    except Exception as e:
        print(f"Error in detection: {e}")

# Capture packets for 30 seconds
def capture_traffic():
    """
    Capture network packets for 30 seconds and save to a PCAP file.
    """
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"device_{timestamp}.pcap"
        print(f"Capturing suspicious packets to {filename}...")
        packets = sniff(timeout=30)
        wrpcap(filename, packets)
        log_activity(filename)
    except Exception as e:
        print(f"Error capturing traffic: {e}")

# Log activity
def log_activity(filename):
    """
    Log the captured PCAP file for suspicious activity.
    """
    try:
        with open("ids_log.txt", "a") as log_file:
            log_file.write(f"{datetime.now()}: Captured {filename}\n")
    except Exception as e:
        print(f"Error logging activity: {e}")

# Start monitoring
sniff(prn=detect_packet, store=0)
