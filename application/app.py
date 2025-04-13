import json
from pathlib import Path
import gc
import os
import base64
import sys
import random
import math
import time

from flask import Flask
from flask import render_template, jsonify
from flask_socketio import SocketIO, emit

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


def generate_dummy_contours(index, is_measurement=False, img_width=512, img_height=512):
    """Generate dummy contour data for testing

    Args:
        index: Frame index
        is_measurement: Whether these are contours for measurement images (adds offset)
        img_width: Image width
        img_height: Image height
    """
    import math

    # Create different contour patterns based on frame index
    contours = {}

    # Apply offset for measurement contours to make projection difference visible
    offset_x = 30 if is_measurement else 0
    offset_y = 20 if is_measurement else 0

    # Shift center for measurement images
    center_x = img_width // 2 + offset_x
    center_y = img_height // 2 + offset_y

    # GTV - circular contour that changes size with frame
    gtv_radius = 50 + 10 * math.sin(index * 0.5)
    gtv_points = []
    for angle in range(0, 360, 10):
        x = center_x + gtv_radius * math.cos(math.radians(angle))
        y = center_y + gtv_radius * math.sin(math.radians(angle))
        gtv_points.append([int(x), int(y)])
    contours["gtv"] = gtv_points

    # CTV - rectangular contour that changes position
    rect_offset_x = 30 * math.sin(index * 0.3) + (offset_x * 0.7)  # Different offset for measurement
    rect_offset_y = 20 * math.cos(index * 0.3) + (offset_y * 0.7)
    rect_size = 80
    ctv_points = [
        [center_x - rect_size + rect_offset_x, center_y - rect_size + rect_offset_y],
        [center_x + rect_size + rect_offset_x, center_y - rect_size + rect_offset_y],
        [center_x + rect_size + rect_offset_x, center_y + rect_offset_y],
        [center_x - rect_size + rect_offset_x, center_y + rect_offset_y],
        [center_x - rect_size + rect_offset_x, center_y - rect_size + rect_offset_y],  # Close the loop
    ]
    contours["ctv"] = ctv_points

    # PTV - star-shaped contour
    ptv_radius = 120 + 15 * math.sin(index * 0.4)
    ptv_points = []
    for i in range(10):
        # Alternating between outer and inner points to create a star
        radius = ptv_radius if i % 2 == 0 else ptv_radius * 0.6
        angle = math.radians(i * 36)
        x = center_x + radius * math.cos(angle)
        y = center_y + radius * math.sin(angle)
        ptv_points.append([int(x), int(y)])
    ptv_points.append(ptv_points[0])  # Close the loop
    contours["ptv"] = ptv_points

    # Spinal cord - zigzag line
    spinal_points = []
    start_y = center_y - 150
    for i in range(10):
        y = start_y + i * 30
        x = center_x - 100 + ((-1) ** i) * (20 + 10 * math.sin(index * 0.2))
        spinal_points.append([int(x), int(y)])
    contours["spinal_cord"] = spinal_points

    # Parotid - oval shape
    parotid_points = []
    a, b = 40 + 5 * math.sin(index * 0.7), 70 + 7 * math.cos(index * 0.7)
    for angle in range(0, 360, 15):
        x = center_x + 150 + a * math.cos(math.radians(angle))
        y = center_y - 50 + b * math.sin(math.radians(angle))
        parotid_points.append([int(x), int(y)])
    contours["parotid"] = parotid_points

    return contours


def load_images_as_data_uris(directory_path: Path, is_measurement=False) -> list[dict]:
    """
    Loads images from a directory, sorts them by filename, and returns
    a list of objects containing Data URIs and contour data.

    Args:
        directory_path: Path to the directory containing images
        is_measurement: Whether these are measurement images (affects contour generation)
    """
    data_objects = []
    if not directory_path.is_dir():
        print(f"Error: Directory not found - {directory_path}", file=sys.stderr)
        return data_objects

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

    for idx, filename in enumerate(image_files):
        file_path = directory_path / filename
        ext = file_path.suffix.lower()
        mime_type = IMAGE_MIME_TYPES.get(ext)

        if mime_type:
            try:
                with open(file_path, "rb") as image_file:
                    encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                    data_uri = f"data:{mime_type};base64,{encoded_string}"

                    # Create image object with dataUri and contours
                    image_object = {
                        "dataUri": data_uri,
                        **generate_dummy_contours(idx, is_measurement),  # Pass is_measurement flag
                    }
                    data_objects.append(image_object)
            except Exception as e:
                print(f"Error processing image {filename}: {e}", file=sys.stderr)

    return data_objects


# --- Simplified Mock Data Store ---
# Maps user-friendly names to actual directory paths
# These are now only used to populate the dropdowns, not for loading images in handle_get_images
AVAILABLE_REFERENCES = {
    "Ref Set 1 (Jan 15)": HERE.parent / "ref_2024_01_15",
    "Ref Set 2 (Feb 10)": HERE.parent / "ref_2024_02_10",
    # Add more reference sets as needed
}

AVAILABLE_MEASUREMENTS = {
    "Meas Set A (Jan 16)": HERE.parent / "meas_2024_01_16",
    "Meas Set B (Jan 17)": HERE.parent / "meas_2024_01_17",
    "Meas Set C (Feb 11)": HERE.parent / "meas_2024_02_11",
    # Add more measurement sets as needed
}
# --- End Simplified Mock Data Store ---


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

        @self.app.route("/api/status")
        def get_status():
            """Returns the current system status for the medical imaging application"""
            import random
            import time

            # In a real implementation, these would come from actual system measurements
            # This is sample data for demonstration
            rand_val = random.random()

            if rand_val < 0.7:  # 70% chance of OK status
                status = {
                    "message": "System operational",
                    "severity": 0,
                    "content": {
                        "Image Quality": "Good",
                        "Dose": f"Within limits ({random.uniform(20, 25):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(95, 99):.1f}%",
                        "Processing Time": f"{random.uniform(0.8, 1.5):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }
            elif rand_val < 0.9:  # 20% chance of warning
                status = {
                    "message": "System requires attention",
                    "severity": 1,
                    "content": {
                        "Image Quality": "Fair",
                        "Dose": f"Near threshold ({random.uniform(25, 28):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(90, 95):.1f}%",
                        "Processing Time": f"{random.uniform(1.5, 2.5):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }
            else:  # 10% chance of error
                status = {
                    "message": "System error detected",
                    "severity": 2,
                    "content": {
                        "Image Quality": "Poor",
                        "Dose": f"Exceeds limits ({random.uniform(28, 35):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(80, 90):.1f}%",
                        "Processing Time": f"{random.uniform(2.5, 4.0):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }

            return jsonify(status)

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
                # Send the keys (names) of the available sets
                reference_names = sorted(list(AVAILABLE_REFERENCES.keys()))
                measurement_names = sorted(list(AVAILABLE_MEASUREMENTS.keys()))

                print(f"Sending available sets - Refs: {reference_names}, Meas: {measurement_names}")
                self.socketio.emit(
                    "available_files",
                    {
                        "references": reference_names,
                        "measurements": measurement_names,
                    },
                )
            except Exception as e:
                print(f"Error getting available files: {e}", file=sys.stderr)
                self.socketio.emit("error", {"message": f"Error getting available files: {e}"})

        @self.socketio.on("get_images")
        def handle_get_images(data):
            # Log the request but ignore the values for loading
            reference_name = data.get("reference")
            measurement_name = data.get("measurement")
            print(f"Received request for images (Ref='{reference_name}', Meas='{measurement_name}'). Loading fixed chunk1/chunk2.")

            # --- Load fixed paths ---
            images_ref_path = HERE.parent / "chunk1"
            images_meas_path = HERE.parent / "chunk2"
            # --- End fixed paths ---

            # Removed checks for name existence in AVAILABLE_* dictionaries

            try:
                print(f"Loading reference images from fixed path: {images_ref_path}")
                print(f"Loading measurement images from fixed path: {images_meas_path}")

                images_ref = load_images_as_data_uris(images_ref_path, False)  # Reference images
                images_meas = load_images_as_data_uris(images_meas_path, True)  # Measurement images with offset

                print(f"Loaded {len(images_ref)} reference images and {len(images_meas)} measurement images with contour data.")

                if not images_ref and not images_meas:
                    # Adjusted warning message
                    print(f"Warning: No images found in fixed paths chunk1/chunk2.", file=sys.stderr)

                self.socketio.emit(
                    "images_data",
                    {
                        "references": images_ref,
                        "measurements": images_meas,
                    },
                )
                gc.collect()
            except Exception as e:
                # Adjusted error message
                msg = f"Error loading images from fixed paths chunk1/chunk2: {e}"
                print(f"Error: {msg}", file=sys.stderr)
                self.socketio.emit("error", {"message": msg})

        @self.socketio.on("get_status")
        def handle_status_request():
            """Handle status request from client and emit status data"""
            # In a real implementation, these would come from actual system measurements
            rand_val = random.random()

            if rand_val < 0.7:  # 70% chance of OK status
                status = {
                    "message": "System operational",
                    "severity": 0,
                    "content": {
                        "Image Quality": "Good",
                        "Dose": f"Within limits ({random.uniform(20, 25):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(95, 99):.1f}%",
                        "Processing Time": f"{random.uniform(0.8, 1.5):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }
            elif rand_val < 0.9:  # 20% chance of warning
                status = {
                    "message": "System requires attention",
                    "severity": 1,
                    "content": {
                        "Image Quality": "Fair",
                        "Dose": f"Near threshold ({random.uniform(25, 28):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(90, 95):.1f}%",
                        "Processing Time": f"{random.uniform(1.5, 2.5):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }
            else:  # 10% chance of error
                status = {
                    "message": "System error detected",
                    "severity": 2,
                    "content": {
                        "Image Quality": "Poor",
                        "Dose": f"Exceeds limits ({random.uniform(28, 35):.1f} Gy)",
                        "Contour Accuracy": f"{random.uniform(80, 90):.1f}%",
                        "Processing Time": f"{random.uniform(2.5, 4.0):.2f}s",
                        "Last Check": time.strftime("%H:%M:%S"),
                    },
                }

            emit("status_update", status)

    def run(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5420):
        self.socketio.run(self.app, debug=debug, host=host, port=port, log_output=True)

    @staticmethod
    def start(host: str = "0.0.0.0", port: int = 5420, debug: bool = False):
        server = Visualizer()
        server.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    Visualizer.start(debug=True)
