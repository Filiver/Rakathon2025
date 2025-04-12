import os
from pointcloud_alignment.fourier import align_measurement_to_reference_scan, dicom_filenames_from_dir
from contours_finder import find_all_contours_in_meas
from itk import process_rt_ct_pairs
from pathlib import Path
from visualize_conturs import visualize_all_contours_from_dict, visualize_two_contour_dicts
import numpy as np
HERE = Path(__file__).parent
# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rackaton Data"
CURR_SAMPLE = "SAMPLE_001"
SAMPLE_ROOT = os.path.join(MAIN_SAMPLES_DIRECTORY, CURR_SAMPLE)
# Set the directory where you want to save the reports
# REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
DEFAULT_REFERENCE = (
    HERE / "data/radioprotect/Organized_CT_Data_Axial/SAMPLE_004/2023-05-02/ref_1_2_246_352_221_50382907113527305278273607881698676893/"
)
DEFAULT_MEASUREMENT = (
    HERE / "data/radioprotect/Organized_CT_Data_Axial/SAMPLE_004/2023-06-13/meas_1_2_246_352_221_54278781642968663956664906787711437486/"
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
    save_videos=False,
)
# Compute measurement contours
# {contour_name: [(x,y,z), ...]}
ct_list, rt_list = load_data(scan_to_process_ref)
contours_dict_ref = process_rt_ct_pairs(
    scan_to_process_ref, ct_list, rt_list)
print(contours_dict_ref.keys())
a = contours_dict_ref[list(contours_dict_ref.keys())[0]]
print(a)
print(a.shape)
contours_meas_torch_dict = find_all_contours_in_meas(
    alignment_results["reference"],
    alignment_results["measurement"],
    contours_dict_ref
)
print("Origin:", alignment_results["origin"])
print("Spacing:", alignment_results["spacing"])
visualize_all_contours_from_dict(contours_meas_torch_dict, np.array(
    alignment_results["spacing"]),
    np.array(alignment_results["origin"]))
visualize_two_contour_dicts(
    contours_dict_ref, contours_meas_torch_dict,
    np.array(alignment_results["spacing"]),
    np.array(alignment_results["origin"])
)
