import os
import time
import requests

SERVER_URL = "http://<server-ip>:<port>/upload"  # Replace with your server's IP and port
UPLOAD_DIR = "/path/to/pcap/files"  # Directory containing .pcap files
SENT_LOG = "/path/to/sent_files.log"  # Log file to track sent files

def send_pcap_files():
    """
    Send .pcap files from the IoT device to the central server.
    """
    try:
        # Load already sent files
        if os.path.exists(SENT_LOG):
            with open(SENT_LOG, "r") as log:
                sent_files = set(log.read().splitlines())
        else:
            sent_files = set()

        # List all .pcap files in the directory
        files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pcap")]

        for file in files:
            if file not in sent_files:
                file_path = os.path.join(UPLOAD_DIR, file)
                print(f"Sending {file} to server...")
                with open(file_path, "rb") as f:
                    response = requests.post(SERVER_URL, files={"file": f})
                    if response.status_code == 200:
                        print(f"Successfully sent {file}")
                        with open(SENT_LOG, "a") as log:
                            log.write(file + "\n")
                        os.remove(file_path)  # Remove the file after successful upload
                    else:
                        print(f"Failed to send {file}: {response.status_code}")

    except Exception as e:
        print(f"Error while sending PCAP files: {e}")

# Schedule the script to run periodically
if __name__ == "__main__":
    while True:
        send_pcap_files()
        time.sleep(60)  # Run every 60 seconds
