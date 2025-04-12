import json
from pathlib import Path
import gc
import os
import base64
import sys

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO

HERE = Path(__file__).parent

TEMPLATES = HERE / "templates"
STATIC = HERE / "static"

# Define supported image extensions and their MIME types
IMAGE_MIME_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".gif": "image/gif",
    ".webp": "image/webp",
    ".bmp": "image/bmp",
}


def load_images_as_data_uris(directory_path: Path) -> list[str]:
    """
    Loads images from a directory, sorts them by filename, and returns
    a list of Base64 encoded Data URIs.

    Args:
        directory_path: Path object pointing to the directory containing images.

    Returns:
        A list of strings, where each string is a Data URI for an image.
        Returns an empty list if the directory doesn't exist or contains no images.
    """
    data_uris = []
    if not directory_path.is_dir():
        print(f"Error: Directory not found - {directory_path}", file=sys.stderr)
        return data_uris

    image_files = []
    for filename in os.listdir(directory_path):
        file_path = directory_path / filename
        if file_path.is_file():
            ext = file_path.suffix.lower()
            if ext in IMAGE_MIME_TYPES:
                image_files.append(filename)

    # Sort files naturally (e.g., frame_1.png, frame_2.png, ..., frame_10.png)
    image_files.sort(key=lambda f: int("".join(filter(str.isdigit, f))) if any(c.isdigit() for c in f) else float("inf"))
    # Fallback simple sort if no digits found for natural sort
    if not any(any(c.isdigit() for c in f) for f in image_files):
        image_files.sort()

    for filename in image_files:
        file_path = directory_path / filename
        ext = file_path.suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext)

        if mime_type:
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    data_uri = f"data:{mime_type};base64,{encoded_string}"
                    data_uris.append(data_uri)
            except Exception as e:
                print(f"Error processing image {filename}: {e}", file=sys.stderr)

    return data_uris


class Visualizer:
    def __init__(self):
        self.app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATIC))
        self.socketio = SocketIO(self.app, json=json, cors_allowed_origins="*", async_mode="eventlet")
        self.setup_routes()
        self.setup_socket_handlers()

    def setup_routes(self):
        @self.app.route("/")
        def index():
            return render_template("index.html")

    def setup_socket_handlers(self):
        @self.socketio.on("connect")
        def handle_connect():
            print("Client connected")

        @self.socketio.on("disconnect")
        def handle_disconnect():
            print("Client disconnected")

        @self.socketio.on("get_available_files")
        def handle_get_available_files():
            try:
                self.socketio.emit(
                    "available_files",
                    {
                        "references": ["model1", "model2", "model3"],
                        "measurements": ["measurement1", "measurement2", "measurement3"],
                    },
                )
            except Exception as e:
                print(f"Error loading model: {e}")
                self.socketio.emit("error", {"message": "Error loading model"})

        @self.socketio.on("get_images")
        def handle_get_images(data):
            try:
                images_1_path = HERE.parent / "chunk1"  # Adjust
                images_2_path = HERE.parent / "chunk2"  # Adjust as needed
                # Assumes a function 'load_images_from_folder' exists at the top level
                # This function should take a Path object and return a dictionary
                # mapping image filenames to base64 encoded data URIs.
                # e.g., {'img1.png': 'data:image/png;base64,...'}
                images_1 = load_images_as_data_uris(images_1_path)
                images_2 = load_images_as_data_uris(images_2_path)

                self.socketio.emit(
                    "images_data",
                    {
                        "references": images_1,
                        "measurements": images_2,
                    },
                )
                gc.collect()  # Optional: Suggest garbage collection
            except Exception as e:
                print(f"Error loading states: {e}")
                self.socketio.emit("error", {"message": "Error loading states"})

    def run(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5420):
        self.socketio.run(self.app, debug=debug, host=host, port=port, log_output=True)

    @staticmethod
    def start(host: str = "0.0.0.0", port: int = 5420):
        server = Visualizer()
        server.run(host=host, port=port)


if __name__ == "__main__":
    Visualizer.start()
