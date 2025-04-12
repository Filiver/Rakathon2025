import numpy as np
from pathlib import Path
from typing import Tuple
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from constants import *


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
        with open(path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
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
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(*points_a.T, c='blue', label='Contour A (Reference)', alpha=0.6, s=5)
    ax.scatter(*points_b.T, c='red', label='Contour B (New)', alpha=0.6, s=5)

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


def compute_point_to_point_distance(contour_a: np.ndarray, contour_b: np.ndarray, threshold_mm: float = 3.0):
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
    percentage_above_threshold = np.sum(distances > threshold_mm) / len(distances) * 100

    # Return the results as a dictionary
    return {
        "mean_distance": mean_distance,
        "max_distance": max_distance,
        "percentage_above_threshold": percentage_above_threshold
    }

def hausdorff_distance(contour1, contour2):
    """
    Calculates the Hausdorff distance between two contours.
    """
    dist_matrix = cdist(contour1, contour2, 'euclidean')
    hausdorff = max(np.min(dist_matrix, axis=1).max(), np.min(dist_matrix, axis=0).max())
    return hausdorff

def compare_contours(c1, c2, type):
    treshold = None
    match type.lower():
        case "gtv":
            threshold = tresh_GTV
        case "ctv":
            threshold = tresh_CTV
        case "ptv":
            threshold = tresh_PTV
        case "spinal_cord":
            threshold = tresh_spinal_cord
        case "parotid":
            threshold = tresh_parotid
        case "submandibular_gland":
            threshold = tresh_submandibular_gland
        case "esophagus":
            threshold = tresh_esophagus
    if not threshold:
        raise ValueError(f"Unknown contour type: {type}, possible types: GTV, CTV, PTV, spinal_cord, parotid, submandibular_gland, esophagus")
    


if __name__ == "__main__":
    # import sys

    # if len(sys.argv) != 3:
    #     print("Usage: python load_contours.py <file1.txt> <file2.txt>")
    #     sys.exit(1)

    # file1, file2 = sys.argv[1], sys.argv[2]
    file1 = "E:\\radioprotect\original_points.txt"
    file2 = "E:\\radioprotect\shifted_points.txt"
    c1, c2 = load_contours(file1, file2)

    print(f"Loaded {len(c1)} points from {file1}")
    print(f"Loaded {len(c2)} points from {file2}")
    # plot_contours_3d(c1, c2, 'sample_contours.png')

    metrics_p2pd = compute_point_to_point_distance(c1, c2, 0.5)
    hausdorff = hausdorff_distance(c1, c2)

    print(f"Mean distance: {metrics_p2pd['mean_distance']:.2f} mm")
    print(f"Max distance: {metrics_p2pd['max_distance']:.2f} mm")
    print(f"Percentage of points > 3mm: {metrics_p2pd['percentage_above_threshold']:.2f}%")
    print(f"Hausdorff Distance: {hausdorff} mm")


