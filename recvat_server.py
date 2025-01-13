from flask import Flask, request
import os

UPLOAD_FOLDER = "/path/to/server/uploads"  # Directory to save uploaded .pcap files

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

@app.route("/upload", methods=["POST"])
def upload_file():
    """
    Endpoint to receive PCAP files from IoT devices.
    """
    if "file" not in request.files:
        return "No file part", 400
    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400
    if file:
        file_path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(file_path)
        print(f"Saved {file.filename} to {UPLOAD_FOLDER}")
        return "File uploaded successfully", 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)  # Replace port if needed
