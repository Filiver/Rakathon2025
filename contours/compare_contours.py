import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from misc_data.constants import *
from scipy.spatial import Delaunay
from misc_scripts.tumor_generator import generate_sphere, generate_biconcave_shape_from_sphere
from tools.logger import log_treatment_check, log_doctor_review, replanning_needed, log_treatment_proceeded


def load_contours(file1: str, file2: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load two 3D contour point sets from text files.

    Each line in the file should be of the format: x,y,z

    Args:
        file1 (str): Path to the first contour file.
        file2 (str): Path to the second contour file.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays of shape (N, 3) and (M, 3)
    """

    def parse_file(filepath: str) -> np.ndarray:
        path = Path(filepath)
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {filepath}")
        data = []
        with open(path, "r") as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) != 3:
                    continue  # skip malformed lines
                try:
                    data.append([float(x) for x in parts])
                except ValueError:
                    continue  # skip lines with non-numeric values
        return np.array(data, dtype=np.float64)

    contour1 = parse_file(file1)
    contour2 = parse_file(file2)
    return contour1, contour2


def plot_contours_3d(points_a: np.ndarray, points_b: np.ndarray, output_file: str):
    """
    Visualize two 3D point clouds in the same plot with different colors, and save to file.

    Args:
        points_a (np.ndarray): First contour, shape (N, 3)
        points_b (np.ndarray): Second contour, shape (M, 3)
        output_file (str): Path to save the output plot image
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(*points_a.T, c="blue", label="Contour A (Reference)", alpha=0.6, s=5)
    ax.scatter(*points_b.T, c="red", label="Contour B (New)", alpha=0.6, s=5)

    ax.set_xlabel("X [mm]")
    ax.set_ylabel("Y [mm]")
    ax.set_zlabel("Z [mm]")
    ax.legend()
    ax.set_title("3D Contour Comparison")
    ax.grid(True)
    plt.tight_layout()

    # Save the plot as a PNG file
    plt.savefig(output_file, dpi=300)
    print(f"Plot saved to {output_file}")
    plt.close()


def compute_point_to_point_distance_3d(contour_a: np.ndarray, contour_b: np.ndarray, threshold: float):
    """
    Compute point-to-point distances between two 3D point clouds and report statistics.

    Args:
        contour_a (np.ndarray): First contour (N, 3) array of 3D points
        contour_b (np.ndarray): Second contour (M, 3) array of 3D points
        threshold_mm (float): The distance threshold (in mm) to flag large shifts, default is 3.0 mm

    Returns:
        dict: Statistics of distances:
            - mean_distance: The average point-to-point distance
            - max_distance: The maximum point-to-point distance
            - percentage_above_threshold: Percentage of points with distance > threshold_mm
    """
    # Build a KDTree for the second contour
    tree_b = cKDTree(contour_b)

    # Find the nearest points in contour_b for each point in contour_a
    distances, _ = tree_b.query(contour_a)

    # Calculate statistics
    mean_distance = np.mean(distances)
    max_distance = np.max(distances)
    percentage_above_threshold = np.sum(distances > threshold) / len(distances) * 100

    # Return the results as a dictionary
    return {"mean_distance": mean_distance, "max_distance": max_distance, "percentage_above_threshold": percentage_above_threshold}


def find_neighbors(seed_point, contour, radius: float) -> np.ndarray:
    """
    Finds points within a given radius of the seed point.
    """
    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(contour)

    # Query points within the specified radius
    indices = tree.query_ball_point(seed_point, radius)

    return indices


def segment_contour_by_proximity(contour, radius=1.0) -> list:
    """
    Segments the contour into regions based on proximity.
    The function iteratively selects seed points and finds neighboring points within a radius.
    """
    n = len(contour)
    unvisited = set(range(n))  # Set of all unvisited points
    segments = []  # List to hold the resulting segments

    while unvisited:
        # Pick an arbitrary seed point (first unvisited point)
        seed_idx = unvisited.pop()
        seed_point = contour[seed_idx]

        # Initialize the segment with the seed point
        segment = [seed_idx]
        neighbors = find_neighbors(seed_point, contour, radius)

        # Add neighbors to the segment and mark them as visited
        for neighbor_idx in neighbors:
            if neighbor_idx in unvisited:
                segment.append(neighbor_idx)
                unvisited.remove(neighbor_idx)

        segments.append(segment)

    return segments


def plot_segments(contour, segments, output_file: str):
    """
    Plots the contour points in 3D, with each segment in a different random color.
    Saves the plot to a file.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i, segment in enumerate(segments):
        # Use the indices from the segment to get the corresponding points in the contour
        segment_points = contour[segment]

        # Generate a random color for each segment
        color = np.random.rand(
            3,
        )

        ax.scatter(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], color=color.tolist(), label=f"Segment {i + 1}")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Contour Segmentation")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_file)


def visualize_segment_difference(segment, contour1, contour2, output_file: str):
    """
    Visualizes the difference between two segments from different contours.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    # Plot the first segment in blue
    ax.scatter(*contour1[segment].T, color="blue", label="Segment 1 (Contour 1)", alpha=0.5)

    # Plot the second segment in red
    ax.scatter(*contour2[segment].T, color="red", label="Segment 2 (Contour 2)", alpha=0.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Segment Difference Visualization")
    ax.legend()

    # Save the plot to a file
    plt.savefig(output_file)


def hausdorff_distance_segment(segment1, segment2, contour1, contour2) -> float:
    """
    Calculates the Hausdorff distance for a pair of segments.
    """
    points1 = contour1[segment1]
    points2 = contour2[segment2]

    # Compute the distance matrix between the two sets of points
    dist_matrix = cdist(points1, points2, "euclidean")

    # Compute the Hausdorff distance for this pair of segments
    hausdorff = max(np.min(dist_matrix, axis=1).max(), np.min(dist_matrix, axis=0).max())

    return hausdorff


def localized_hausdorff_distance(contour1, contour2, segments1, segments2, threshold: float) -> Tuple[float, float, list]:
    """
    Calculates the localized Hausdorff distance for corresponding segments between two contours.
    Returns the mean, max Hausdorff distance, and segments exceeding the threshold.
    """
    localized_hausdorff_distances = []
    problematic_segments = []  # To store segments that exceed the threshold

    # Iterate over corresponding segments from both contours
    for i, (segment1, segment2) in enumerate(zip(segments1, segments2)):
        # Calculate the Hausdorff distance for the current pair of segments
        hd = hausdorff_distance_segment(segment1, segment2, contour1, contour2)
        localized_hausdorff_distances.append(hd)

        # Check if the Hausdorff distance exceeds the threshold
        if hd > threshold:
            problematic_segments.append(i)  # Append the index of the problematic segment

    # Aggregate the results: You can change this to np.mean() or np.max() based on your needs
    mean_hausdorff = np.mean(localized_hausdorff_distances)
    max_hausdorff = np.max(localized_hausdorff_distances)

    return mean_hausdorff, max_hausdorff, problematic_segments


def tetrahedron_volume(p1, p2, p3, p4) -> float:
    # Using the determinant method to calculate the volume of the tetrahedron
    return abs(np.dot(p1 - p4, np.cross(p2 - p4, p3 - p4))) / 6.0


def calculate_volume(contour) -> float:
    # Perform Delaunay triangulation on the contour
    delaunay = Delaunay(contour)

    total_volume = 0
    # Iterate over the simplices (tetrahedra) in the Delaunay triangulation
    for simplex in delaunay.simplices:
        # Get the 4 points defining the tetrahedron
        p1, p2, p3, p4 = contour[simplex]
        total_volume += tetrahedron_volume(p1, p2, p3, p4)

    return total_volume


def compare_contours(c1, c2, threshold: float) -> dict:
    # print(f"Using threshold: {threshold} mm")

    ptpd = compute_point_to_point_distance_3d(c1, c2, threshold)
    segments = segment_contour_by_proximity(c1, radius=contour_segmentation_radius)
    # Filter segments to include only those with 10 or more points
    filtered_segments = [seg for seg in segments if len(seg) >= 10]
    print(len(filtered_segments), "segments found with >= 10 points (originally", len(segments), ")")

    # Check if there are any valid segments left before proceeding
    if not filtered_segments:
        print("Warning: No segments with >= 10 points found. Skipping localized Hausdorff distance calculation.")
        mean_hd, max_hd, problematic_segments = 0, 0, []
    else:
        # Use the filtered segments for visualization and distance calculation
        # Note: visualize_segment_difference might need adjustment if segment indices change
        # For now, let's visualize the first valid segment if it exists
        mean_hd, max_hd, problematic_segments = localized_hausdorff_distance(c1, c2, filtered_segments, filtered_segments, threshold)
        # visualize_segment_difference(segments[problematic_segments[0]], c1, c2, 'segment_difference.png')
        # print(len(problematic_segments[0]), "points in the problematic segment")

    # plot_segments(c1, filtered_segments, 'segmented_contour.png') # Use filtered_segments if plotting

    volume_diff = calculate_volume(c1) - calculate_volume(c2)

    return {
        "mean_point_to_point_distance": ptpd["mean_distance"],
        "max_point_to_point_distance": ptpd["max_distance"],
        "percentage_above_ptp_threshold": ptpd["percentage_above_threshold"],
        "mean_localized_hausdorff_distance": mean_hd,
        "max_localized_hausdorff_distance": max_hd,
        "problematic_segments": problematic_segments,
        "problematic_segments_count": len(problematic_segments),
        "volume_difference": volume_diff,
    }


def check_contours(c1, c2, type: str, log: bool = False, print_comparison: bool = False) -> Tuple[int, list]:
    threshold = None

    if type.startswith("gtv"):
        threshold = thresh_GTV
    elif type.startswith("ctv"):
        threshold = thresh_CTV
    elif type.startswith("ptv"):
        threshold = thresh_PTV
    elif "spinalcord" in type:
        threshold = thresh_spinal_cord
    elif "parotid" in type:
        threshold = thresh_parotid
    elif "submandibular" in type or "glnd_submand" in type:
        threshold = thresh_submandibular_gland
    elif "esophagus" in type:
        threshold = thresh_esophagus

    if threshold is None:
        raise ValueError(f"Unknown contour type: {type}, possible types: GTV, CTV, PTV, spinalcord, parotid, submandibulargland, esophagus")

    comp = compare_contours(c1, c2, threshold)
    if print_comparison:
        print(comp)

    alert_level = OK
    reasons = []

    if comp["mean_point_to_point_distance"] > threshold + error_margin:
        alert_level = REPLANNING_NEEDED
        reasons.append("Mean point-to-point distance exceeds threshold")
    elif comp["mean_point_to_point_distance"] > threshold - error_margin:
        alert_level = DOCTOR_REVIEW
        reasons.append("Mean point-to-point distance is close to threshold")

    if log:
        if alert_level == OK:
            log_treatment_check("T042", "P001", "OK", "Contour check passed")
            log_treatment_proceeded("D042", "P001", "OK")
        elif alert_level == DOCTOR_REVIEW:
            log_treatment_check("T042", "P001", "DOCTOR_REVIEW", "Contour check requires doctor review, reasons: " + ", ".join(reasons))
            log_doctor_review("D012", "P001", "T042", "ACCEPT", "After review treatment accepted")
            log_treatment_proceeded("D042", "P001", "DOCTOR_REVIEW")
        elif alert_level == REPLANNING_NEEDED:
            log_treatment_check(
                "T042", "P001", "REPLANNING_NEEDED", "Contour check requires replanning, dangerous contours detected, reasons: " + ", ".join(reasons)
            )
            replanning_needed("T042", "P001", "Contour check requires replanning, dangerous contours detected")

    return alert_level, reasons


def check_all_contours(contours_dict_ref, contours_meas_torch_dict):
    output = {}
    for contour_name, contour_points in contours_dict_ref.items():
        contour_meas_points = contours_meas_torch_dict[contour_name]
        alert_level, reasons = check_contours(contour_points, contour_meas_points, contour_name)
        print(f"Contour: {contour_name}, Alert Level: {alert_level}, Reasons: {reasons}")
        output[contour_name] = {"alert_level": alert_level, "reasons": reasons}
    return output


if __name__ == "__main__":
    # import sys

    # if len(sys.argv) != 3:
    #     print("Usage: python load_contours.py <file1.txt> <file2.txt>")
    #     sys.exit(1)

    # file1, file2 = sys.argv[1], sys.argv[2]

    # ---------------------------------------------------
    # file1 = "E:\\radioprotect\original_points.txt"
    # file2 = "E:\\radioprotect\shifted_points.txt"
    # c1, c2 = load_contours(file1, file2)

    # print(f"Loaded {len(c1)} points from {file1}")
    # print(f"Loaded {len(c2)} points from {file2}")

    # alert_level, reasons = check_contours(c1, c2, "GTV")
    # print(f"Alert Level: {alert_level}")
    # print(f"Reasons: {reasons}")
    # ---------------------------------------------------
    contour1 = generate_sphere(num_points=5000)  # Points on the sphere
    contour2 = generate_biconcave_shape_from_sphere(contour1, deformation_scale=0.5)  # Deformed points
    # # plot_contours_3d(contour1, contour2, 'sample_contours_2.png')
    # res = compare_contours(contour1, contour2, 0.2)
    # print(res)

    alert_level, reasons = check_contours(contour1, contour2, "GTV", log=True, print_comparison=True)
    print(f"Alert Level: {alert_level}")
    print(f"Reasons: {reasons}")

    # plot_contours_3d(c1, c2, 'sample_contours.png')

    # metrics_p2pd = compute_point_to_point_distance(c1, c2, 0.5)
    # # hausdorff = hausdorff_distance(c1, c2)

    # print(f"Mean distance: {metrics_p2pd['mean_distance']:.2f} mm")
    # print(f"Max distance: {metrics_p2pd['max_distance']:.2f} mm")
    # print(f"Percentage of points > 3mm: {metrics_p2pd['percentage_above_threshold']:.2f}%")
    # # print(f"Hausdorff Distance: {hausdorff} mm")

    # segments = segment_contour_by_proximity(contour2, radius=0.4)

    # # Plot the segmented contour
    # plot_segments(contour2, segments, 'segmented_contour_2.png')
    # print(len(segments), "segments found")

    # thresh = 3
    # mean_hd, max_hd, problematic_segments = localized_hausdorff_distance(c1, c2, segments, segments, thresh)

    # print(f"Mean Localized Hausdorff Distance: {mean_hd} mm")
    # print(f"Max Localized Hausdorff Distance: {max_hd} mm")
    # print(f"Problematic Segments (Hausdorff > {thresh} mm): {problematic_segments}")

    # volume = calculate_volume(c1)
    # print(f"Estimated Volume: {volume}")
