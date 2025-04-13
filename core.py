from pathlib import Path
import datetime
import torch
import os
from pointcloud_alignment.fourier import align_measurement_to_reference_scan, dicom_filenames_from_dir
from contours_finder import find_all_contours_in_meas
from itk import process_rt_ct_pairs
from pathlib import Path
from visualize_conturs import visualize_all_contours_from_dict, visualize_two_contour_dicts, visualize_all_contours_from_dict2
import numpy as np

HERE = Path(__file__).parent
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Organized_CT_Data_Axial"


def get_sample_reference_and_measurement(sample_name):
    """
    Get the reference and measurement directories for a given sample name.
    """
    sample_path = Path(MAIN_SAMPLES_DIRECTORY) / sample_name
    references = {}
    measurements = {}
    for date in sample_path.iterdir():
        if date.is_dir():
            for folder in date.iterdir():
                if folder.is_dir():
                    if "ref" in folder.name:
                        references[date.name] = folder
                    elif "meas" in folder.name:
                        measurements[date.name] = folder
    # Sort the references and measurements by date
    references = dict(sorted(references.items(), key=lambda x: datetime.datetime.strptime(x[0], "%Y-%m-%d")))
    measurements = dict(sorted(measurements.items(), key=lambda x: datetime.datetime.strptime(x[0], "%Y-%m-%d")))
    return references, measurements


def load_data_details(folder):
    return os.listdir(os.path.join(folder, "CT")), os.listdir(os.path.join(folder, "RT"))


# Load all files of sample organized by frame of reference
def run_estimation_pipeline(ref_scan_dir, meas_scan_dir):
    # Align the two scans
    alignment_results = align_measurement_to_reference_scan(
        dicom_filenames_from_dir(ref_scan_dir / "CT"),
        dicom_filenames_from_dir(meas_scan_dir / "CT"),
        save_videos=False,
    )
    # Compute measurement contours
    # {contour_name: [(x,y,z), ...]}
    ct_list, rt_list = load_data_details(ref_scan_dir)
    contours_dict_ref = process_rt_ct_pairs(ref_scan_dir, ct_list, rt_list)
    print(contours_dict_ref.keys())
    a = contours_dict_ref[list(contours_dict_ref.keys())[0]]
    print(a)
    print(a.shape)
    print("Origin:", alignment_results["origin"])
    print("Spacing:", alignment_results["spacing"])
    contours_meas_torch_dict = find_all_contours_in_meas(
        alignment_results["reference"], alignment_results["measurement"], alignment_results["spacing"], alignment_results["origin"], contours_dict_ref
    )
    return alignment_results, contours_meas_torch_dict


def organize_contours_by_slice(contours_dict):
    """
    Reorganizes contours by D-slice (depth).

    Args:
        contours_dict (dict): Dictionary with keys 'original_image' and 'transformed_image',
                              each containing a sub-dictionary mapping ROI names to
                              tensor (N, 3) in image coordinates (d, h, w).

    Returns:
        dict: A dictionary with the following structure:
              {
                  'original': {
                      d_index: {
                          roi_name: tensor of shape (M, 2) containing (h, w) coordinates
                      }
                  },
                  'transformed': {
                      d_index: {
                          roi_name: tensor of shape (M, 2) containing (h, w) coordinates
                      }
                  }
              }
    """
    result = {"original": {}, "transformed": {}}

    # Process original contours
    for roi_name, coords in contours_dict["original_image"].items():
        # Round D values to nearest integer and convert to int
        d_indices = coords[:, 0].round().long()

        # For each unique D index
        for d_idx in torch.unique(d_indices):
            # Convert to integer for proper indexing - this helps JavaScript access
            d_int = int(d_idx.item())

            # Find points at this D index
            mask = d_indices == d_idx
            hw_points = coords[mask, 1:]  # Extract (h, w) coordinates

            # Initialize slice dict if needed
            if d_int not in result["original"]:
                result["original"][d_int] = {}

            # Store points for this ROI at this slice
            result["original"][d_int][roi_name] = hw_points

    # Process transformed contours (same logic)
    for roi_name, coords in contours_dict["transformed_image"].items():
        d_indices = coords[:, 0].round().long()

        for d_idx in torch.unique(d_indices):
            # Convert to integer for proper indexing
            d_int = int(d_idx.item())

            mask = d_indices == d_idx
            hw_points = coords[mask, 1:]

            if d_int not in result["transformed"]:
                result["transformed"][d_int] = {}

            result["transformed"][d_int][roi_name] = hw_points

    return result


def pipeline_results_to_image_outputs(
    alignment_results,
    contours_meas_torch_dict,
):
    """
    Process alignment results and contours to create slice-organized data structure.

    Args:
        alignment_results (dict): Results from alignment process.
        contours_meas_torch_dict (dict): Contours dictionary from find_all_contours_in_meas.

    Returns:
        dict: A dictionary containing contours organized by slice for both original and transformed coordinates.
    """
    # Organize contours by slice for easy access when visualizing
    contours_by_slice = organize_contours_by_slice(contours_meas_torch_dict)

    # Return the organized data
    return {"contours_by_slice": contours_by_slice, "original_contours": contours_meas_torch_dict, "alignment_results": alignment_results}


if __name__ == "__main__":
    # Example usage
    sample_name = "SAMPLE_001"
    references, measurements = get_sample_reference_and_measurement(sample_name)
    print("References:", references)
    print("Measurements:", measurements)
    # Select FoR for processing
    scan_to_process_ref = references["2023-06-05"]
    scan_to_process_meas = measurements["2023-06-21"]
    # Run the estimation pipeline
    alignment_results, contours_dict = run_estimation_pipeline(scan_to_process_ref, scan_to_process_meas)

    # Process results to organize by slice
    results = pipeline_results_to_image_outputs(alignment_results, contours_dict)

    # Now results['contours_by_slice'] contains contours organized by slice
    print("Contours organized by slice:")
    print(results["contours_by_slice"])

    # Now results['contours_by_slice'] contains:
    # {
    #     'original': {
    #         slice_index: {roi_name: tensor(h,w)}
    #     },
    #     'transformed': {
    #         slice_index: {roi_name: tensor(h,w)}
    #     }
    # }

    # Example of accessing contours for a specific slice:
    # slice_index = 50  # Example slice
    # if slice_index in results['contours_by_slice']['original']:
    #     print(f"ROIs in original slice {slice_index}:", list(results['contours_by_slice']['original'][slice_index].keys()))
