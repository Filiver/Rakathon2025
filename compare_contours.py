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

def find_neighbors(seed_point, contour, radius):
    """
    Finds points within a given radius of the seed point.
    """
    # Build a KDTree for efficient nearest neighbor search
    tree = cKDTree(contour)
    
    # Query points within the specified radius
    indices = tree.query_ball_point(seed_point, radius)
    
    return indices

def segment_contour_by_proximity(contour, radius=1.0):
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

    
def plot_segments(contour, segments, output_file):
    """
    Plots the contour points in 3D, with each segment in a different random color.
    Saves the plot to a file.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, segment in enumerate(segments):
        # Use the indices from the segment to get the corresponding points in the contour
        segment_points = contour[segment]
        
        # Generate a random color for each segment
        color = np.random.rand(3,)
        
        ax.scatter(segment_points[:, 0], segment_points[:, 1], segment_points[:, 2], 
                   color=color.tolist(), label=f'Segment {i+1}')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Contour Segmentation')
    ax.legend()
    
    # Save the plot to a file
    plt.savefig(output_file)

def hausdorff_distance_segment(segment1, segment2, contour1, contour2):
    """
    Calculates the Hausdorff distance for a pair of segments.
    """
    points1 = contour1[segment1]
    points2 = contour2[segment2]
    
    # Compute the distance matrix between the two sets of points
    dist_matrix = cdist(points1, points2, 'euclidean')
    
    # Compute the Hausdorff distance for this pair of segments
    hausdorff = max(np.min(dist_matrix, axis=1).max(), np.min(dist_matrix, axis=0).max())
    
    return hausdorff

def localized_hausdorff_distance(contour1, contour2, segments1, segments2, threshold):
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
    # hausdorff = hausdorff_distance(c1, c2)

    print(f"Mean distance: {metrics_p2pd['mean_distance']:.2f} mm")
    print(f"Max distance: {metrics_p2pd['max_distance']:.2f} mm")
    print(f"Percentage of points > 3mm: {metrics_p2pd['percentage_above_threshold']:.2f}%")
    # print(f"Hausdorff Distance: {hausdorff} mm")

    segments = segment_contour_by_proximity(c1, radius=0.4)

    # Plot the segmented contour
    plot_segments(c1, segments, 'segmented_contour.png')
    print(len(segments), "segments found")

    tresh = 3
    mean_hd, max_hd, problematic_segments = localized_hausdorff_distance(c1, c2, segments, segments, tresh)

    print(f"Mean Localized Hausdorff Distance: {mean_hd} mm")
    print(f"Max Localized Hausdorff Distance: {max_hd} mm")
    print(f"Problematic Segments (Hausdorff > {tresh} mm): {problematic_segments}")



