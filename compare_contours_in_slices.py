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

if __name__ == "__main__":
    # Load the points from the pickle files
    dict1 = load_points_from_pkl("rand1.pkl")
    dict2 = load_points_from_pkl("rand2.pkl")

    points1 = dict1['parotid_l']
    points2 = dict2['parotid_l']
    points2_rounded = round_z_coordinates_tensor(points2, method='floor')


