import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt
import sys

# --- Configuration ---
# Input directory: The output of the 'organize_ct_by_date_frame.py' script
ORGANIZED_CT_DIRECTORY = "/Users/davidkorcak/Documents/Rakathon2025/data/radioprotect/Organized_CT_Data"
# Output directory: Where the PNG files will be saved, mirroring the input structure
PNG_OUTPUT_DIRECTORY = "/Users/davidkorcak/Documents/Rakathon2025/data/radioprotect/Organized_CT_PNGs"
# PNG settings
COLORMAP = "gray"  # Colormap for saving PNGs (e.g., 'gray', 'bone')
# Optional: Define window/level for PNG contrast (set to None to use default full range)
# Example: Soft tissue window
# WINDOW_LEVEL = {"window": 400, "level": 50}
# Example: Bone window
# WINDOW_LEVEL = {"window": 1800, "level": 400}
WINDOW_LEVEL = None
# -------------------


def apply_window_level(slice_data, window, level):
    """Applies window and level to a numpy array."""
    min_val = level - window / 2
    max_val = level + window / 2
    # Apply windowing
    slice_data = np.clip(slice_data, min_val, max_val)
    # Normalize to 0-1 range for colormapping
    if max_val > min_val:
        slice_data = (slice_data - min_val) / (max_val - min_val)
    else:
        slice_data = np.zeros_like(slice_data)  # Avoid division by zero if window is 0
    return slice_data


def save_slice_as_png(slice_data, output_filepath, colormap, window_level=None):
    """Saves a 2D numpy array slice as a PNG image."""
    try:
        # Ensure data is float for processing
        processed_data = slice_data.astype(np.float32)

        if window_level:
            processed_data = apply_window_level(processed_data, window_level["window"], window_level["level"])

        # Use matplotlib to save the image with the specified colormap
        # origin='lower' might need adjustment depending on how SimpleITK loads vs. how matplotlib displays
        plt.imsave(output_filepath, processed_data, cmap=colormap, format="png", origin="upper")
        # print(f"    Saved: {os.path.basename(output_filepath)}")
    except Exception as e:
        print(f"Error saving slice to {output_filepath}: {e}", file=sys.stderr)


def process_frame_directory(frame_dir_path, output_png_dir):
    """
    Reads a DICOM series from a directory and saves each slice as a PNG.
    Assumes the directory contains a single, sorted CT series.
    """
    print(f"  Processing series in: {os.path.basename(frame_dir_path)}")
    try:
        # Use SimpleITK's ImageSeriesReader to load the sorted series
        reader = sitk.ImageSeriesReader()
        dicom_names = reader.GetGDCMSeriesFileNames(frame_dir_path)
        if not dicom_names:
            print(f"  Warning: No DICOM files found by SeriesReader in {frame_dir_path}. Skipping.", file=sys.stderr)
            return

        reader.SetFileNames(dicom_names)
        image_3d = reader.Execute()

        # Get the image data as a numpy array (z, y, x)
        image_array = sitk.GetArrayFromImage(image_3d)
        num_slices = image_array.shape[0]
        print(f"    Loaded series with {num_slices} slices.")

        # Create the output directory if it doesn't exist
        try:
            os.makedirs(output_png_dir, exist_ok=True)
        except OSError as e:
            print(f"Error creating output directory {output_png_dir}: {e}", file=sys.stderr)
            return  # Cannot save PNGs if directory creation fails

        # Save each slice
        for i in range(num_slices):
            slice_data = image_array[i, :, :]
            # Format filename like slice_000.png, slice_001.png, etc.
            output_filename = f"slice_{i:03d}.png"
            output_filepath = os.path.join(output_png_dir, output_filename)
            save_slice_as_png(slice_data, output_filepath, COLORMAP, WINDOW_LEVEL)

    except Exception as e:
        print(f"Error processing series in {frame_dir_path}: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()


def process_all_organized_samples(input_base_dir, output_base_dir):
    """
    Walks the organized CT directory structure and generates PNGs.
    """
    if not os.path.isdir(input_base_dir):
        print(f"Error: Input directory '{input_base_dir}' not found.", file=sys.stderr)
        return

    print(f"\nStarting PNG generation process.")
    print(f"Input organized CT directory: {input_base_dir}")
    print(f"Output PNG directory: {output_base_dir}")

    # Walk through the structure: Sample -> Date -> Frame UID
    for sample_name in sorted(os.listdir(input_base_dir)):
        sample_path = os.path.join(input_base_dir, sample_name)
        if not os.path.isdir(sample_path):
            continue
        print(f"\nProcessing Sample: {sample_name}")

        for date_str in sorted(os.listdir(sample_path)):
            date_path = os.path.join(sample_path, date_str)
            if not os.path.isdir(date_path):
                continue
            print(f"  Processing Date: {date_str}")

            for frame_dir_name in sorted(os.listdir(date_path)):
                frame_dir_path = os.path.join(date_path, frame_dir_name)
                if not os.path.isdir(frame_dir_path) or not frame_dir_name.startswith("frame_uid_"):
                    continue

                # Construct the corresponding output directory path for PNGs
                output_png_path = os.path.join(output_base_dir, sample_name, date_str, frame_dir_name)

                # Process the directory containing the DICOM series
                process_frame_directory(frame_dir_path, output_png_path)

    print("\nPNG generation process finished.")


if __name__ == "__main__":
    # Basic check if paths are default/placeholder
    if "/path/to/" in ORGANIZED_CT_DIRECTORY or "/path/to/" in PNG_OUTPUT_DIRECTORY:
        print("Please update ORGANIZED_CT_DIRECTORY and PNG_OUTPUT_DIRECTORY variables in the script.", file=sys.stderr)
    else:
        process_all_organized_samples(ORGANIZED_CT_DIRECTORY, PNG_OUTPUT_DIRECTORY)
