import organize_new
import os
from pointcloud_alignment.fourier import align_measurement_to_reference_scan, dicom_filenames_from_dir
from contours_finder import find_contours_in_meas
from itk import process_rt_ct_pairs
from pathlib import Path

HERE = Path(__file__).parent
# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rackaton Data"
CURR_SAMPLE = "SAMPLE_001"
SAMPLE_ROOT = os.path.join(MAIN_SAMPLES_DIRECTORY, CURR_SAMPLE)
# Set the directory where you want to save the reports
# REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
DEFAULT_REFERENCE = (
        HERE / "data/radioprotect/Organized_CT_Data_Axial/SAMPLE_001/2023-06-05/ref_1_2_246_352_221_559666980133719263215614360979762074268/"
)
DEFAULT_MEASUREMENT = (
        HERE / "data/radioprotect/Organized_CT_Data_Axial/SAMPLE_001/2023-06-21/meas_1_2_246_352_221_523526543250385987917834924930119139461/"
)

# -------------------

def load_data(folder):
    return os.listdir(os.path.join(folder, "CT")), os.listdir(os.path.join(folder, "RT"))

# Load all files of sample organized by frame of reference

# Select FoR for processing
scan_to_process_ref = DEFAULT_REFERENCE
scan_to_process_meas = DEFAULT_MEASUREMENT
# Align the two scans
alignment_results = align_measurement_to_reference_scan(
    dicom_filenames_from_dir(scan_to_process_ref / "CT"),
    dicom_filenames_from_dir(scan_to_process_meas / "CT"),
    save_videos=False
)
# Compute measurement contours
# {contour_name: [(x,y,z), ...]}
contours_dict_ref = process_rt_ct_pairs(scan_to_process_ref, *load_data(scan_to_process_ref))
contours_dict, contours_torch = find_contours_in_meas(
    alignment_results["reference"],
    alignment_results["measurement"],
    contours_dict_ref
)
