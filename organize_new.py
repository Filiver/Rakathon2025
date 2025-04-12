import os
import pydicom
from pydicom.errors import InvalidDicomError
from collections import defaultdict
from datetime import datetime


def process_sample(sample_dir):
    output = {}
    for item in os.listdir(sample_dir):
        item_path = os.path.join(sample_dir, item)
        if os.path.isdir(item_path):
            process_sample(item_path)
        elif item.lower().endswith(".dcm"):
            try:
                ds = pydicom.dcmread(item_path, stop_before_pixels=True)
                # print(f"Processing DICOM file: {item_path}")
                # Add your processing logic here
                if ds.FrameOfReferenceUID not in output:
                    output[ds.FrameOfReferenceUID] = {}
                if ds.Modality not in output[ds.FrameOfReferenceUID]:
                    output[ds.FrameOfReferenceUID][ds.Modality] = []
                output[ds.FrameOfReferenceUID][ds.Modality].append(item)
            except InvalidDicomError:
                print(f"Invalid DICOM file: {item_path}")
            except Exception as e:
                print(f"Error processing {item_path}: {e}")
    return output


def process_all_samples(main_dir):
    """
    Processes all subdirectories (samples) in the main directory.

    Args:
        main_dir (str): Path to the main directory containing sample subdirectories.
        output_dir (str): Path to the directory where reports will be saved.
    """
    output = {}

    if not os.path.isdir(main_dir):
        print(f"Error: Main directory not found: {main_dir}")
        return

    print(f"Starting processing in main directory: {main_dir}")
    # print(f"Reports will be saved to: {output_dir}")

    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isdir(item_path):
            output[item] = process_sample(item_path)

    print("Processing finished.")
    return output


# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rakathon Data"
# Set the directory where you want to save the reports
REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
# -------------------

if __name__ == "__main__":
    # Basic check if paths are placeholder
    if "/path/to/" in MAIN_SAMPLES_DIRECTORY or "/path/to/" in REPORTS_OUTPUT_DIRECTORY:
        print("Please update MAIN_SAMPLES_DIRECTORY and REPORTS_OUTPUT_DIRECTORY variables in the script.")
    else:
        print(os.listdir(os.getcwd()))
        output = process_all_samples(MAIN_SAMPLES_DIRECTORY)
        counts = {}
        for sample in output:
            counts[sample] = {}
            for identifier in output[sample]:
                counts[sample][identifier] = {}
                for modality in output[sample][identifier]:
                    counts[sample][identifier][modality] = len(
                        output[sample][identifier][modality])

        print(counts)
