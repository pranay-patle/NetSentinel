import tensorflow as tf
import numpy as np
import time
import os
from scapy.all import sniff, wrpcap
from datetime import datetime
import pandas as pd

# Load the TensorFlow Lite model
interpreter = tf.lite.Interpreter(model_path="iot_ids_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define real-time packet feature extraction
def extract_features(packet):
    features = {
        "Protocol Type": packet[IP].proto if IP in packet else 0,
        "Rate": len(packet) / (time.time() - extract_features.start_time) if hasattr(extract_features, "start_time") else 0,
        "Srate": 0,  # Placeholder
        "Drate": 0,  # Placeholder
        "TCP": 1 if TCP in packet else 0,
        "UDP": 1 if UDP in packet else 0,
        "Tot size": len(packet),
        "IAT": time.time() - extract_features.last_packet_time if hasattr(extract_features, "last_packet_time") else 0
    }
    extract_features.start_time = time.time()
    extract_features.last_packet_time = time.time()
    return list(features.values())

# Real-time detection
def detect_packet(packet):
    features = np.array([extract_features(packet)], dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], features)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])
    if prediction > 0.5:
        print("Suspicious activity detected!")
        capture_traffic()

# Capture packets for 30 seconds if suspicious activity is detected
def capture_traffic():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"device_{timestamp}.pcap"
    print(f"Capturing suspicious packets to {filename}...")
    packets = sniff(timeout=30)
    wrpcap(filename, packets)
    log_activity(filename)

# Log suspicious activity
def log_activity(filename):
    with open("ids_log.txt", "a") as log_file:
        log_file.write(f"{datetime.now()}: Captured {filename}\n")

# Start monitoring
sniff(prn=detect_packet, store=0)
