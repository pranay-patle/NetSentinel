#!/bin/bash

# Update packages
sudo apt-get update
sudo apt-get install -y tcpdump python3 python3-pip

# Install Python libraries
pip3 install numpy pandas

echo "Server setup complete!"
