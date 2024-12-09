#!/bin/bash

# Update packages and install dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip tcpdump

# Install Python libraries
pip3 install tensorflow scapy numpy pandas

# Set permissions
chmod +x ids_script.py

echo "Raspberry Pi setup complete!"
