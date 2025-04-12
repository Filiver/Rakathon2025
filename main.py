import organize_new
import os
from alignment import align_scans
from contours_finder import find_countours_in_scan

# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rakathon Data"
# Set the directory where you want to save the reports
# REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
# -------------------


# Load all files of sample organized by frame of reference
all_files = organize_new.process_sample(os.path.join(MAIN_SAMPLES_DIRECTORY, "SAMPLE_001"))
# Select FoR for processing
scan_to_process_A = all_files.values()[0]
scan_to_process_B = all_files.values()[1]
# Align the two scans
scan_A, scan_B, origin, spacing = align_scans(scan_to_process_A, scan_to_process_B)
# Compute measurement contours
# {contour_name: [(x,y,z), ...]}
contours = find_countours_in_scan(scan_A, scan_B, origin, spacing)


