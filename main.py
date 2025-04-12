import organize_new
import os
from pointcloud_alignment.fourier import align_measurement_to_reference_scan
from contours_finder import find_countours_in_scan

here = os.path.dirname(os.path.abspath(__file__))
# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rackaton Data"
CURR_SAMPLE = "SAMPLE_001"
SAMPLE_ROOT = os.path.join(MAIN_SAMPLES_DIRECTORY, CURR_SAMPLE)
# Set the directory where you want to save the reports
# REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
# -------------------


# Load all files of sample organized by frame of reference
all_files = list(organize_new.process_sample(SAMPLE_ROOT).items())
sample_name = "SAMPLE_001"
all_files = organize_new.process_sample(
    os.path.join(MAIN_SAMPLES_DIRECTORY, sample_name))
# Select FoR for processing
scan_to_process_ref = list(all_files.values())[0]
scan_to_process_meas = list(all_files.values())[1]
# Align the two scans
scan_ref, scan_meas = align_scans(scan_to_process_ref, scan_to_process_meas)
# Compute measurement contours
# {contour_name: [(x,y,z), ...]}
contours_dict, contours_torch = find_countours_in_scan(
    scan_ref, scan_to_process_ref, sample_name)
