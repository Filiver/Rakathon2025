import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
# Import the function from your contours_finder file
# Make sure contours_finder.py is updated to handle W, H, D scans!
from contours_finder import find_contours_in_meas_my, load_contours_from_txt


def create_hollow_sphere_xyz(shape_whd, center_xyz, radius, thickness=1, bg_noise_level=0.1, device='cpu'):
    """
    Creates a hollow sphere (surface) of ones in a 3D array with (W, H, D) dimensions.
    Args:
        shape_whd (tuple): (W, H, D) for the volume dimensions (X, Y, Z axes).
        center_xyz (tuple): (x, y, z) coordinates for the sphere center.
        radius (float): Radius of the sphere.
        thickness (float): Thickness of the sphere shell.
        bg_noise_level (float): Maximum value for background noise.
        device (str): Device ('cpu' or 'cuda').
    Returns:
        torch.Tensor: 3D tensor representing the scan with shape (W, H, D).
    """
    W, H, D = shape_whd
    # Initialize with background noise
    volume = torch.rand(shape_whd, device=device) * bg_noise_level

    # Create coordinates (X, Y, Z order)
    x, y, z = torch.meshgrid(
        torch.arange(W, device=device),
        torch.arange(H, device=device),
        torch.arange(D, device=device),
        indexing='xy'  # x, y, z order
    )

    # Ensure center is a tensor for broadcasting
    center_t = torch.tensor(center_xyz, device=device, dtype=torch.float32)

    # Calculate distance from the center (using x, y, z coordinates)
    dist = ((x - center_t[0])**2 + (y - center_t[1])
            ** 2 + (z - center_t[2])**2).sqrt()

    # Create mask for the shell based on distance and thickness
    outer_radius = radius + thickness / 2.0
    inner_radius = radius - thickness / 2.0
    shell_mask = (dist >= inner_radius) & (dist <= outer_radius)

    # Set the shell voxels to 1.0
    volume[shell_mask] = 1.0

    return volume.float()  # Return volume with noise and shell


def extract_surface_coords_xyz_from_whd(volume_whd, threshold=0.5):
    """
    Extracts (x, y, z) coordinates where volume > threshold.
    Assumes input volume has shape (W, H, D).
    Args:
        volume_whd (torch.Tensor): Input scan (W, H, D).
        threshold (float): Threshold to identify surface points.
    Returns:
        torch.Tensor: Coordinates (N, 3) in (x, y, z) order.
    """
    # Find indices where volume > threshold. nonzero returns (x, y, z) order for (W, H, D) input.
    coords_xyz = (volume_whd > threshold).nonzero(
        as_tuple=False)  # Shape (N, 3) -> (x, y, z)
    if coords_xyz.numel() == 0:
        return torch.empty((0, 3), dtype=torch.float32, device=volume_whd.device)
    return coords_xyz.float()


def visualize_alignment_xyz(ref_coords_xyz, meas_coords_xyz, transformed_coords_xyz, shape_whd, title=""):
    """
    Visualize sphere surfaces using scatter plots. Assumes input coordinates are (x, y, z).
    Args:
        ref_coords_xyz (torch.Tensor): Reference surface points (N, 3) -> (x, y, z).
        meas_coords_xyz (torch.Tensor): Measurement surface points (M, 3) -> (x, y, z).
        transformed_coords_xyz (torch.Tensor): Transformed reference points (N, 3) -> (x, y, z).
        shape_whd (tuple): Original scan dimensions (W, H, D) for aspect ratio.
        title (str): Plot title.
    """
    W, H, D = shape_whd
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    # Plot reference surface points (green)
    ref_np = ref_coords_xyz.cpu().numpy()
    ax.scatter(ref_np[:, 0], ref_np[:, 1], ref_np[:, 2], c='green',
               marker='.', s=10, alpha=0.6, label="Reference")

    # Plot original measurement surface points (red)
    meas_np = meas_coords_xyz.cpu().numpy()
    ax.scatter(meas_np[:, 0], meas_np[:, 1], meas_np[:, 2], c='red',
               marker='.', s=10, alpha=0.6, label="Measurement")

    # Plot transformed contour points (blue) - these are the reference points moved by the transform
    trans_np = transformed_coords_xyz.cpu().numpy()
    ax.scatter(trans_np[:, 0], trans_np[:, 1], trans_np[:, 2], c='blue',
               marker='.', s=10, alpha=0.6, label="Transformed")

    # Plot setup
    ax.set_title(title)
    ax.set_xlabel("X (Width)")
    ax.set_ylabel("Y (Height)")
    ax.set_zlabel("Z (Depth)")
    # Ensure consistent aspect ratio based on shape (W, H, D) -> (X, Y, Z)
    # Set aspect based on data range in each dimension
    ax.set_box_aspect((W, H, D))
    ax.view_init(elev=30, azim=45)  # Adjust view angle if needed
    # Set limits based on shape to keep origin consistent
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_zlim(0, D)
    # ax.invert_zaxis() # Optional: Invert Z if needed for specific view
    ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    shape_whd = (64, 64, 64)  # W, H, D (corresponds to X, Y, Z axes)
    W, H, D = shape_whd

    # --- Sphere Parameters (centers are X, Y, Z) ---
    ref_center_xyz = (W//2, H//2, D//2)         # Centered
    ref_radius = 10
    # Introduce translation and scaling for the measurement sphere
    meas_center_xyz = (W//2+5, H//2+10, D//2-12)  # Shifted in x, y, z
    meas_radius = 15  # Slightly different radius
    thickness = 2    # Thickness of the shell
    noise_level = 0.1  # Max value for background noise
    surface_threshold = 0.5  # Threshold > noise_level to extract surface points

    # --- Device Setup ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Create Scans (W, H, D) ---
    print("Creating synthetic scans (W, H, D)...")
    scan_ref = create_hollow_sphere_xyz(
        shape_whd, ref_center_xyz, ref_radius, thickness, noise_level, device)
    scan_meas = create_hollow_sphere_xyz(
        shape_whd, meas_center_xyz, meas_radius, thickness, noise_level, device)
    print(f"Scan shape: {scan_ref.shape}")  # Should print (W, H, D)

    # --- Extract Contour Points (x, y, z) from Reference Scan ---
    print("Extracting contour points (x, y, z) from reference scan...")
    contours_xyz = extract_surface_coords_xyz_from_whd(
        scan_ref, threshold=surface_threshold)

    if contours_xyz.numel() == 0:
        print(
            f"Error: No contour points found in reference scan using threshold {surface_threshold}.")
        return
    print(f"Number of contour points extracted: {contours_xyz.shape[0]}")

    # --- Run Alignment ---
    # find_contours_in_meas needs to be adapted to handle (W, H, D) scans
    print("Running alignment optimization...")
    transformed_contours_xyz = find_contours_in_meas_my(
        scan_ref,      # Reference scan (W, H, D)
        scan_meas,     # Measurement scan (W, H, D)
        contours_xyz   # Contour points from ref scan (N, 3) -> (x, y, z)
    )
    print("Alignment finished.")

    # --- Visualization ---
    print("Preparing visualization...")
    # Extract measurement surface points just for visualization comparison
    meas_surface_coords_xyz = extract_surface_coords_xyz_from_whd(
        scan_meas, threshold=surface_threshold)

    if meas_surface_coords_xyz.numel() == 0:
        print(
            f"Warning: No surface points found in measurement scan using threshold {surface_threshold} for visualization.")

    visualize_alignment_xyz(
        contours_xyz,             # Original reference points (green)
        meas_surface_coords_xyz,  # Original measurement points (red)
        # Transformed reference points (blue) - should align with red
        transformed_contours_xyz.detach(),
        shape_whd,
        title="Sphere Surface Alignment Test (WHD Scans)"
    )


if __name__ == "__main__":
    main()
