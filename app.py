import json
from pathlib import Path
import gc
import os
import base64
import sys
from PIL import Image
import io
import base64

from flask import Flask
from flask import render_template
from flask_socketio import SocketIO

from core import get_sample_reference_and_measurement, run_estimation_pipeline, pipeline_results_to_image_outputs
from detection.detect_intersects import detect_intersect, compare_contour_sets
from contours.compare_contours_in_slices import process_contours


HERE = Path(__file__).parent

TEMPLATES = HERE / "application/templates"
STATIC = HERE / "application/static"

# ROI color definitions for use in mapping and visualization
ROI_COLORS = {
    "GTV": "#FF4500",  # Orange-Red
    "CTV": "#FFD700",  # Gold
    "PTV": "#32CD32",  # Lime Green
    "SPINAL_CORD": "#1E90FF",  # Dodger Blue
    "PAROTID": "#9370DB",  # Medium Purple
    "SUBMANDIBULAR": "#FF69B4",  # Hot Pink
    "ESOPHAGUS": "#CD853F",  # Peru (brownish)
}

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


def map_contour_name_to_js(roi_name):
    """
    Maps real contour names from the estimation pipeline to the expected JavaScript contour names.
    This ensures compatibility with the frontend visualization.

    Args:
        roi_name: Original ROI name from the contour dictionary

    Returns:
        Mapped name that's compatible with the JavaScript frontend
    """
    # Convert to lowercase for case-insensitive matching
    name_lower = roi_name.lower()

    # Primary mapping dictionary - maps specific names to JS expected names
    mapping = {
        # Standard mappings
        "gtv": "gtv",
        "ctv": "ctv",
        "ptv": "ptv",
        "spinal_cord": "spinal_cord",
        "parotid": "parotid",
        "submandibular": "submandibular",
        "esophagus": "esophagus",
        # Handle common variants
        "spinal": "spinal_cord",
        "cord": "spinal_cord",
        "spine": "spinal_cord",
        "parotid_left": "parotid",
        "parotid_right": "parotid",
        "left_parotid": "parotid",
        "right_parotid": "parotid",
        "submandibular_gland": "submandibular",
        "left_submandibular": "submandibular",
        "right_submandibular": "submandibular",
        "esophagus_upper": "esophagus",
        "esophagus_lower": "esophagus",
    }

    # Check for exact matches in our mapping dictionary
    if name_lower in mapping:
        return mapping[name_lower]

    # Handle prefix matches (e.g., "ptv_low" -> "ptv")
    for prefix in ["gtv", "ctv", "ptv"]:
        if name_lower.startswith(prefix):
            return prefix

    # Handle more complex matches
    if "spinal" in name_lower or "cord" in name_lower or "spine" in name_lower:
        return "spinal_cord"

    if "parotid" in name_lower:
        return "parotid"

    if "submandibular" in name_lower:
        return "submandibular"

    if "esophagus" in name_lower:
        return "esophagus"

    # If no mapping found, return the original name but ensure it's lowercase and uses
    # underscores instead of spaces or hyphens for JavaScript compatibility
    return name_lower.replace(" ", "_").replace("-", "_")


def process_tensor_images(tensor_list, is_measurement=False, convert_to_data_uri=True, contours_by_slice=None) -> list[dict]:
    """
    Processes a list of PyTorch tensor images and returns a list of objects
    containing data and contour information.

    Args:
        tensor_list: List of PyTorch tensors in range [0, 1], format [D, H, W] or [C, H, W]
        is_measurement: Whether these are measurement images (affects contour generation)
        convert_to_data_uri: If True, converts tensors to data URIs; otherwise keeps tensors
        contours_by_slice: Optional dictionary containing real contours organized by slice index

    Returns:
        list[dict]: List of dictionaries containing image data (as dataUri or tensor) and contours
    """

    data_objects = []

    # Log available slices for debugging
    if contours_by_slice:
        contour_type = "transformed" if is_measurement else "original"
        if contour_type in contours_by_slice:
            print(f"Available {contour_type} slices: {sorted(list(contours_by_slice[contour_type].keys()))}")
            for slice_idx in contours_by_slice[contour_type]:
                print(f"  ROIs in slice {slice_idx}: {list(contours_by_slice[contour_type][slice_idx].keys())}")
        else:
            print(f"No {contour_type} contours found in data")

    for idx, tensor in enumerate(tensor_list):
        # Ensure tensor is on CPU and in range [0, 1]
        tensor = tensor.detach().cpu()
        if tensor.min() < 0 or tensor.max() > 1:
            tensor = tensor.clamp(0, 1)

        image_object = {}

        if convert_to_data_uri:
            # Handle different tensor formats
            if len(tensor.shape) == 2:
                # Single 2D image [H, W]
                pil_image = Image.fromarray((tensor.numpy() * 255).astype("uint8"), "L")
            elif len(tensor.shape) == 3 and tensor.shape[0] == 1:
                # Single channel [1, H, W]
                pil_image = Image.fromarray((tensor[0].numpy() * 255).astype("uint8"), "L")
            elif len(tensor.shape) == 3 and tensor.shape[0] == 3:
                # RGB image [3, H, W]
                pil_image = Image.fromarray((tensor.permute(1, 2, 0).numpy() * 255).astype("uint8"), "RGB")
            else:
                # Assume it's a grayscale slice from a volume [D, H, W], take middle slice if multiple
                if len(tensor.shape) == 3:
                    # If it's [D, H, W], take a single slice
                    slice_idx = tensor.shape[0] // 2 if idx == 0 else min(idx, tensor.shape[0] - 1)
                    tensor_slice = tensor[slice_idx]
                else:
                    tensor_slice = tensor

                pil_image = Image.fromarray((tensor_slice.numpy() * 255).astype("uint8"), "L")

            # Save to bytes buffer
            buffer = io.BytesIO()
            pil_image.save(buffer, format="PNG")
            buffer.seek(0)

            # Encode to base64
            encoded_string = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_uri = f"data:image/png;base64,{encoded_string}"

            image_object["dataUri"] = data_uri
        else:
            # Keep the tensor
            image_object["tensor"] = tensor

        # Add real contours if available
        # We'll no longer fall back to dummy contours
        slice_indices_to_try = [idx]  # Start with exact match

        # If we have a 3D volume, this tensor index might not match the slice index in contours
        # Try to find closest matching slice
        if len(tensor.shape) == 3 and tensor.shape[0] > 1:
            # This is a 3D volume, try finding the closest slice if exact match fails
            slice_indices_to_try = [idx, idx - 1, idx + 1, idx - 2, idx + 2]  # Try nearby slices too

        contour_type = "transformed" if is_measurement else "original"
        found_contours = False

        if contours_by_slice and contour_type in contours_by_slice:
            # Try each potential slice index
            for try_idx in slice_indices_to_try:
                if try_idx in contours_by_slice[contour_type]:
                    # Found matching slice
                    slice_contours = {}
                    roi_mapping = {}  # Track mapped names to handle duplicates

                    # Debug
                    print(f"Found contours for {'measurement' if is_measurement else 'reference'} image {idx} in slice {try_idx}")

                    for roi_name, points in contours_by_slice[contour_type][try_idx].items():
                        # Convert tensor to list of [x, y] coordinates
                        try:
                            points_list = points.cpu().numpy().tolist()
                            # Convert from (h, w) to (x, y) for display
                            points_list = [[float(p[1]), float(p[0])] for p in points_list]

                            # Map the ROI name to JavaScript expected name
                            js_roi_name = map_contour_name_to_js(roi_name)

                            # Debug - check the points and mapping
                            if len(points_list) > 0:
                                print(f"  ROI {roi_name} -> {js_roi_name}: {len(points_list)} points, first point: {points_list[0]}")

                            # Handle duplicates by keeping the one with more points or prioritizing exact matches
                            if js_roi_name in roi_mapping:
                                existing_name = roi_mapping[js_roi_name]
                                # If the current name is an exact match and existing isn't, prefer current
                                if roi_name.lower() == js_roi_name and existing_name.lower() != js_roi_name:
                                    slice_contours[js_roi_name] = points_list
                                    roi_mapping[js_roi_name] = roi_name
                                # Otherwise, prefer the one with more points
                                elif len(points_list) > len(slice_contours[js_roi_name]):
                                    slice_contours[js_roi_name] = points_list
                                    roi_mapping[js_roi_name] = roi_name
                            else:
                                # First time seeing this mapped name
                                slice_contours[js_roi_name] = points_list
                                roi_mapping[js_roi_name] = roi_name

                        except Exception as e:
                            print(f"Error converting contour {roi_name}: {e}")
                            continue

                    # Only add contours if we found any valid ones
                    if slice_contours:
                        image_object.update(slice_contours)
                        found_contours = True
                        break  # Stop trying other indices

        # If no real contours were found, we simply don't add any contours
        # No longer falling back to dummy contours
        if not found_contours:
            print(f"No contours found for {'measurement' if is_measurement else 'reference'} image {idx}")

        data_objects.append(image_object)

    return data_objects


class Visualizer:
    def __init__(self, sample_name: str = "SAMPLE_001"):
        self.app = Flask(__name__, template_folder=str(TEMPLATES), static_folder=str(STATIC))
        self.socketio = SocketIO(self.app, json=json, cors_allowed_origins="*", async_mode="eventlet")
        self.setup_routes()
        self.setup_socket_handlers()
        self.sample_meas_refs = get_sample_reference_and_measurement(sample_name)

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
                # Add a small delay to prevent rapid reconnection cycles
                import time

                time.sleep(0.2)  # 200ms delay

                self.socketio.emit(
                    "available_files",
                    {
                        "references": list(self.sample_meas_refs[0].keys()),
                        "measurements": list(self.sample_meas_refs[1].keys()),
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
            reference_dir = self.sample_meas_refs[0].get(reference_name)
            measurement_dir = self.sample_meas_refs[1].get(measurement_name)

            try:
                print(f"Processing reference and measurement data...")

                # Run the pipeline
                alignment_results, contours_dict = run_estimation_pipeline(reference_dir, measurement_dir)
                results = pipeline_results_to_image_outputs(alignment_results, contours_dict)
                intersections = detect_intersect(contours_dict)
                cover = compare_contour_sets(contours_dict["binned_z_transform"], contours_dict["binned_z_original"])
                # Get tensors from your results
                reference_tensors = [alignment_results["reference"][i] for i in range(alignment_results["reference"].shape[0])]
                measurement_tensors = [alignment_results["measurement"][i] for i in range(alignment_results["measurement"].shape[0])]

                # Get contours by slice from the results
                contours_by_slice = results.get("contours_by_slice", {})

                # Print structure information for debugging
                print("\nContours by slice structure:")
                print(f"Keys: {list(contours_by_slice.keys())}")
                for key in contours_by_slice:
                    print(f"{key} has {len(contours_by_slice[key])} slices")
                    if len(contours_by_slice[key]) > 0:
                        first_slice = next(iter(contours_by_slice[key]))
                        roi_names = list(contours_by_slice[key][first_slice].keys())
                        print(f"First slice {first_slice} has ROIs: {roi_names}")

                        # Debug: Print the mapped names for each ROI
                        mapped_names = [f"{roi} -> {map_contour_name_to_js(roi)}" for roi in roi_names]
                        print(f"Mapped ROI names: {mapped_names}")

                # Process tensors with real contours
                reference_data = process_tensor_images(reference_tensors, is_measurement=False, contours_by_slice=contours_by_slice)
                measurement_data = process_tensor_images(measurement_tensors, is_measurement=True, contours_by_slice=contours_by_slice)

                print(f"Processed {len(reference_data)} reference images and {len(measurement_data)} measurement images")
                print(f"Intersections: {intersections}")
                print(f"Cover: {cover}")

                results, status, ok = process_contours(
                    contours_dict["binned_z_transform"],
                    contours_dict["binned_z_original"],
                    alignment_results["origin"],
                    alignment_results["spacing"],
                )

                # Send the processed data and status to client in a single message
                self.socketio.emit(
                    "images_data",
                    {
                        "references": reference_data,
                        "measurements": measurement_data,
                        "roi_checks": results.get("roi_checks", {}),
                        "status": status,  # Include status in the same message
                    },
                )

                # Clean up to free memory
                gc.collect()
                print("Data and status sent successfully")

            except Exception as e:
                msg = f"Error processing data: {e}"
                print(f"Error: {msg}", file=sys.stderr)
                self.socketio.emit("error", {"message": msg})

    def run(self, debug: bool = False, host: str = "0.0.0.0", port: int = 5420):
        self.socketio.run(self.app, debug=debug, host=host, port=port, log_output=True)

    @staticmethod
    def start(sample, host: str = "0.0.0.0", port: int = 5420, debug: bool = False):
        server = Visualizer(sample)
        server.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Start the Visualizer server.")
    parser.add_argument("--sample", type=str, default="SAMPLE_001", help="Sample name to visualize")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    args = parser.parse_args()
    Visualizer.start(args.sample, debug=args.debug)
