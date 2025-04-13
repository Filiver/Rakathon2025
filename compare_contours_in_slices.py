import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
from scipy.spatial import cKDTree
from contour_slice_test_generator import generate_ellipse_points
from constants import *
from contours_finder import binned_metric_xy_to_image_hw

def load_points_from_pkl(filepath):
    with open(filepath, "rb") as f:
        return pickle.load(f)

def filter_by_z(points1, points2):
    # Extract the z-values (assuming they're whole numbers already)
    z_slices1 = points1[:, 2]
    z_slices2 = points2[:, 2]

    # Get unique z-slice values (the z-values will be used as the slice key)
    unique_z_slices1 = np.unique(z_slices1)
    unique_z_slices2 = np.unique(z_slices2)

    # Group points by z-slice (x and y are preserved as floats)
    slices1 = {z: points1[z_slices1 == z] for z in unique_z_slices1}
    slices2 = {z: points2[z_slices2 == z] for z in unique_z_slices2}

    return slices1, slices2


def round_z_coordinates_tensor(points, method='round'):
    """
    Round or floor the z-coordinates of the points in a tensor to the nearest whole number.
    
    Parameters:
    - points: torch.Tensor of shape (N, 3)
    - method: 'round' for rounding or 'floor' for flooring the z-coordinate
    """
    points_rounded = points.clone()  # Create a clone to avoid modifying the original tensor
    if method == 'round':
        points_rounded[:, 2] = torch.round(points_rounded[:, 2])
    elif method == 'floor':
        points_rounded[:, 2] = torch.floor(points_rounded[:, 2])  # or torch.ceil() for ceiling if needed
    return points_rounded

def to_numpy(arr):
    """Convert torch tensor or leave numpy array unchanged."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return arr

def plot_contour_slices(points1, points2, filepath="contour_comparison.png", label1="Set 1", label2="Set 2"):
    """
    Plot and save a 2D comparison of two sets of 3D points (ignoring z-axis).
    Handles both NumPy arrays and PyTorch tensors.
    """
    # Convert to NumPy if needed
    points1 = to_numpy(points1)
    points2 = to_numpy(points2)

    # Extract x and y coordinates (ignore z)
    x1, y1 = points1[:, 0], points1[:, 1]
    x2, y2 = points2[:, 0], points2[:, 1]

    # Plotting
    plt.figure(figsize=(6, 6))
    plt.scatter(x1, y1, c='blue', label=label1, s=10, alpha=0.6)
    plt.scatter(x2, y2, c='red', label=label2, s=10, alpha=0.6)

    plt.xlabel("X (mm)")
    plt.ylabel("Y (mm)")
    plt.title("Contour Slice Comparison")
    plt.legend()
    plt.axis("equal")
    plt.grid(True)

    # Save and close
    plt.savefig(filepath)
    plt.close()

def compare_contour_slices(points1, points2, threshold=2.0):
    """
    Compare two sets of 3D points in the same z-slice and check if local shift exceeds threshold.

    Args:
        points1: (N, 3) array-like, reference contour.
        points2: (M, 3) array-like, measured contour.
        threshold: float, distance in mm to flag as significant shift.

    Returns:
        exceeded: bool, True if any shift > threshold.
        problem_points: list of tuples [(p1, nearest_p2, dist), ...] where dist > threshold.
    """
    # Convert to NumPy if needed and reduce to x, y
    p1 = np.asarray(points1)[:, :2]
    p2 = np.asarray(points2)[:, :2]

    tree = cKDTree(p2)
    distances, indices = tree.query(p1)

    problem_points = []
    for i, dist in enumerate(distances):
        if dist > threshold:
            problem_points.append((points1[i], points2[indices[i]], dist))

    exceeded = len(problem_points) > 0
    return exceeded, problem_points

def compare_contour_slices_2d(points1, points2, origin_zyx, spacing_zyx, depth, threshold=2.0):
    """
    Compare two sets of 2D points in the same slice and check if local shift exceeds threshold.

    Args:
        points1: (N, 2) array-like, reference contour.
        points2: (M, 2) array-like, measured contour.
        threshold: float, distance in mm to flag as significant shift.

    Returns:
        exceeded: bool, True if any shift > threshold.
        problem_points: list of tuples [(p1, nearest_p2, dist), ...] where dist > threshold.
    """
    # Convert to NumPy arrays
    p1 = np.asarray(points1)
    p2 = np.asarray(points2)

    # Create a KD-tree for the second set of points (measured contour)
    tree = cKDTree(p2)

    # Find the closest points in p2 for each point in p1
    distances, indices = tree.query(p1)

    problem_points = []
    for i, dist in enumerate(distances):
        if dist > threshold:
            # Add to problem points if the distance exceeds threshold
            problem_points.append(
                (binned_metric_xy_to_image_hw({depth: points1[i]}, origin_zyx, spacing_zyx),
                 binned_metric_xy_to_image_hw({depth: points2[indices[i]]}, origin_zyx, spacing_zyx),
                 dist))



    exceeded = len(problem_points) > 0
    return exceeded, problem_points

def process_contours(orig, transf, origin_zyx, spacing_zyx):
    """
    Process both original and transformed contour data for each body part and slice.
    
    Args:
        orig (dict): Original contour data with body parts as keys and slice data as nested dicts.
        transf (dict): Transformed contour data with body parts as keys and slice data as nested dicts.
        origin_zyx: Origin coordinates for image conversion.
        spacing_zyx: Spacing information for image conversion.
    
    Returns:
        dict: A dictionary with the structure:
            {slice_number: {body_part: list_of_problematic_points}}
    """
    results = {}  # Initialize the main results dictionary

    ok = True
    message = ""
    exceeded_in = {}
    average_excess_per_part = {}

    for body_part in orig.keys():
        threshold = None
        match body_part:
            case 'parotid_l' | 'parotid_r':
                threshold = thresh_parotid
            case 'submandibular_l' | 'submandibular_r' | 'glnd_submand_l' | 'glnd_submand_r':
                threshold = thresh_submandibular_gland
            case 'esophagus':
                threshold = thresh_esophagus
            case 'spinal_cord' | 'spinalcord_prv' | 'spinalcord':
                threshold = thresh_spinal_cord
            case 'ctv_low' | 'ctv_high':
                threshold = thresh_CTV
            case 'ptv_low' | 'ptv_mid00':
                threshold = thresh_PTV
        
        if threshold is None:
            # Consider logging a warning or skipping if a default behavior is acceptable
            print(f"Warning: Unknown body part '{body_part}', skipping.")
            continue # Skip this body part if threshold is not defined
            # Or raise ValueError(f"Unknown body part: {body_part}") if it's critical
        
        # Check if the body part exists in the transformed data
        if body_part not in transf:
            print(f"Warning: Body part '{body_part}' not found in transformed data. Skipping.")
            continue

        body_part_total_exceeded = 0.0 # Initialize per-part total excess
        body_part_count_exceeded = 0   # Initialize per-part count

        for slice_num in orig[body_part].keys():
            # Ensure the slice also exists in the transformed data for this body part
            if slice_num not in transf[body_part]:
                print(f"Warning: Slice {slice_num} missing in transformed data for body part {body_part}. Skipping slice.")
                continue

            slice1 = orig[body_part][slice_num].numpy()  # Assuming tensors, convert to numpy arrays
            slice2 = transf[body_part][slice_num].numpy()

            
            # Compare the slices and get the problematic points
            exceeded, problems = compare_contour_slices_2d(slice1, slice2, origin_zyx, spacing_zyx, slice_num, threshold)
            if exceeded:
                ok = False
                exceeded_in.setdefault(body_part, []).append(slice_num)
                for _, _, dist in problems:
                    body_part_total_exceeded += abs(dist) # Accumulate per-part excess
                    body_part_count_exceeded += 1    
                


            
            # Ensure the slice number exists as a key in the results dictionary
            if slice_num not in results:
                results[slice_num] = {}
            
            # Store the problems under the current slice number and body part
            results[slice_num][body_part] = problems

        if body_part_count_exceeded > 0:
            average_excess_per_part[body_part] = body_part_total_exceeded / body_part_count_exceeded
        
    if ok:
        # message = "All contours are within the shift thresholds"
        status = {
                    "message": "All deviations are within limits",
                    "severity": 0,
                    "content": {},
                }
    else:
        message_parts = []
        content_details = {} # Initialize dictionary for status content
        for bp, slices in exceeded_in.items():
            avg_excess = average_excess_per_part.get(bp, 0) # Get average, default to 0 if not found
            sorted_unique_slices = sorted(list(set(slices))) # Get unique, sorted list of slices
            # Format message part for the overall message string
            message_parts.append(f"{bp} on average by {avg_excess:.2f} mm in slices: {sorted_unique_slices}")
            # Populate the content dictionary for the status object
            content_details[bp] = f"average deviation: {avg_excess:.2f} mm, in slices: {sorted_unique_slices}"

        message = "Threshold exceeded in the following body parts: " + ", ".join(message_parts)

        status = {
                    "message": "Review by a doctor required, deviations detected",
                    "severity": 1,
                    "content": content_details, # Assign the populated dictionary here
                }
              
    return results, status, ok


# def process_contours(orig, transf):
#     """
#     Process both original and transformed contour data for each body part and slice.
    
#     Args:
#         orig (dict): Original contour data with body parts as keys and slice data as nested dicts.
#         transf (dict): Transformed contour data with body parts as keys and slice data as nested dicts.
#         threshold (float): The threshold for considering a point as problematic.
    
#     Returns:
#         dict: A dictionary with the structure:
#             {body_part: {slice_number: list_of_problematic_points}}
#     """
#     results = {}
    
#     for body_part in orig.keys():
#         part_results = {}
#         threshold = None
#         match body_part:
#             case 'parotid_l' | 'parotid_r':
#                 threshold = thresh_parotid
#             case 'submandibular_l' | 'submandibular_r' | 'glnd_submand_l' | 'glnd_submand_r':
#                 threshold = thresh_submandibular_gland
#             case 'esophagus':
#                 threshold = thresh_esophagus
#             case 'spinal_cord' | 'spinalcord_prv' | 'spinalcord':
#                 threshold = thresh_spinal_cord
#             case 'ctv_low' | 'ctv_high':
#                 threshold = thresh_CTV
#             case 'ptv_low' | 'ptv_mid00':
#                 threshold = thresh_PTV
        
#         if threshold is None:
#             raise ValueError(f"Unknown body part: {body_part}")
 
        
#         for slice_num in orig[body_part].keys():
#             # Check if the slice exists in the transformed contours as well
#             if slice_num in transf[body_part]:
#                 slice1 = orig[body_part][slice_num].numpy()  # Assuming tensors, convert to numpy arrays
#                 slice2 = transf[body_part][slice_num].numpy()
                
#                 # Compare the slices and get the problematic points
#                 exceeded, problems = compare_contour_slices_2d(slice1, slice2, threshold)
                
#                 # Store the problems for this body part and slice
#                 part_results[slice_num] = problems
#             else:
#                 # If slice is missing in transf, you can decide how to handle this case (e.g., skip, or log an error)
#                 print(f"Warning: Slice {slice_num} missing in transformed data for body part {body_part}")
        
#         # Store the results for this body part
#         results[body_part] = part_results
    
    return results

def visualize_comparison(points1, points2, problem_points, filepath="comparison_visual.png"):
    """
    Visualize matched points and highlight problem areas.

    Args:
        points1: array-like, shape (n, 3)
        points2: array-like, shape (m, 3)
        problem_points: list of (p1, p2, dist) where p1 and p2 are 3D np.arrays
        filepath: where to save the plot
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot all points as background (light gray)
    ax.scatter(points1[:, 0], points1[:, 1], c='gray', s=5, label="Ref", alpha=0.3)
    ax.scatter(points2[:, 0], points2[:, 1], c='blue', s=5, label="Meas", alpha=0.3)

    # Plot problem points with red line and highlight
    for p1, p2, dist in problem_points:
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', linewidth=1.5)
        ax.scatter(p1[0], p1[1], c='red', s=15)
        ax.scatter(p2[0], p2[1], c='red', s=15, marker='x')

    ax.set_aspect("equal")
    ax.set_title("Comparison of Contours with Problem Areas Highlighted")
    ax.legend()
    plt.savefig(filepath)
    plt.close()

def compare_all_contours(contours_orig, contours_transf):
    ...

if __name__ == "__main__":
    pts = load_points_from_pkl("rand3.pkl")
    # print(pts.keys())
    orig = pts['binned_z_original']
    transf = pts['binned_z_transform']

    result = process_contours(orig, transf)
    print(result['spinalcord'].keys())

    # print(transf.keys())
    # Load the points from the pickle files
    # dict1 = load_points_from_pkl("rand1.pkl")
    # dict2 = load_points_from_pkl("rand2.pkl")

    # points1 = dict1['parotid_l']
    # points2 = dict2['parotid_l']
    # points2_rounded = round_z_coordinates_tensor(points2, method='floor')

    # slices1, slices2 = filter_by_z(points1, points2_rounded)

    # print(slices2[z])
    # print(slices1.keys())
    # dict_keys([np.float64(-204.0), np.float64(-201.0), np.float64(-198.0), np.float64(-195.0), np.float64(-192.0), np.float64(-189.0), np.float64(-186.0), np.float64(-183.0), np.float64(-180.0), np.float64(-177.0), np.float64(-174.0), np.float64(-171.0)])

    # z = -189.0

    # plot_contour_slices(slices1[z], slices2[z], filepath="./images/contour_slice_comparison.png", label1="Ref", label2="Meas")

    # exceeded, problems = compare_contour_slices(slices1[z], slices2[z], threshold=2.0)
    # visualize_comparison(slices1[z], slices2[z], problems, filepath="./images/comparison_visual.png")

    # points1 = generate_ellipse_points(a=20, b=20, num_points=200, center=(0, 0), noise=0.3, z=0)
    # points2 = generate_ellipse_points(a=20, b=16, num_points=200, center=(0, 0), noise=0.3, z=0)

    # plot_contour_slices(points1, points2, filepath="./images/contour_slice_comparison_2.png", label1="Ref", label2="Meas")
    # exceeded, problems = compare_contour_slices(points1, points2, threshold=3.0)
    # visualize_comparison(points1, points2, problems, filepath="./images/comparison_visual_2.png")





