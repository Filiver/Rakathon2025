import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch

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


if __name__ == "__main__":
    # Load the points from the pickle files
    dict1 = load_points_from_pkl("rand1.pkl")
    dict2 = load_points_from_pkl("rand2.pkl")

    points1 = dict1['parotid_l']
    points2 = dict2['parotid_l']
    points2_rounded = round_z_coordinates_tensor(points2, method='floor')

    slices1, slices2 = filter_by_z(points1, points2_rounded)

    # print(slices2[z])
    # print(slices1.keys())
    # dict_keys([np.float64(-204.0), np.float64(-201.0), np.float64(-198.0), np.float64(-195.0), np.float64(-192.0), np.float64(-189.0), np.float64(-186.0), np.float64(-183.0), np.float64(-180.0), np.float64(-177.0), np.float64(-174.0), np.float64(-171.0)])

    z = -189.0

    plot_contour_slices(slices1[z], slices2[z], filepath="./images/contour_slice_comparison.png", label1="Ref", label2="Meas")


