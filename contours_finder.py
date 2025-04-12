import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torchvision
import torch.optim as optim

import os
import numpy as np
import torch


import os
import torch


def load_contours_from_txt(dir_path):
    raw_roi_points = {}     # original: roi_name ➝ list of [x, y, z]
    tensor_roi_points = {}  # new: roi_name ➝ torch.tensor of shape (N, 3)

    for filename in os.listdir(dir_path):
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    roi_name, points_str = line.strip().split(':', 1)
                    points = [list(map(float, p.split(',')))
                              for p in points_str.strip().split()
                              if len(p.split(',')) == 3]

                    if not points:
                        continue

                    # Append to raw list
                    if roi_name not in raw_roi_points:
                        raw_roi_points[roi_name] = points
                    else:
                        raw_roi_points[roi_name].extend(points)

    # Convert raw points to torch tensors
    for roi_name, point_list in raw_roi_points.items():
        tensor_roi_points[roi_name] = torch.tensor(
            point_list, dtype=torch.float32)

    return raw_roi_points, tensor_roi_points


def find_contours_in_meas(scan_ref, scan_meas, contours_xyz):
    # Initialize 3x4 affine transform with batch dim: [1, 3, 4]
    # swap axes to match (x, y, z) -> (z, y, x)
    # scan_ref = scan_ref.permute(2, 1, 0)
    # scan_meas = scan_meas.permute(2, 1, 0)
    # contours_xyz = contours_xyz.permute(1, 0)

    # theta = torch.tensor([[[1, 0, 0, 0],   # x-axis
    #                       [0, 1, 0, 0],   # y-axis
    #                       [0, 0, 1, 0]]],  # z-axis
    #                     dtype=torch.float32, device=scan_ref.device, requires_grad=True)
    theta = torch.rand(1, 3, 4, device=scan_ref.device,
                       dtype=torch.float32, requires_grad=True)
    theta = torch.nn.Parameter(theta)

    optimizer = optim.AdamW([theta], lr=0.01)
    criterion = torch.nn.MSELoss()

    # Make sure input is 5D: [B, C, D, H, W]
    scan_ref = scan_ref.unsqueeze(0).unsqueeze(0)
    scan_meas = scan_meas.unsqueeze(0).unsqueeze(0)

    # --- Start Changes ---
    # Check scan dimensions after unsqueezing
    if not (len(scan_meas.shape) == 5 and all(s > 0 for s in scan_meas.shape[2:])):
        raise ValueError(
            f"scan_meas must have shape [B, C, D, H, W] with D, H, W > 0. Got shape: {scan_meas.shape}")

    # Get scan dimensions (D, H, W)
    D, H, W = scan_meas.shape[2:]

    # Size for normalization: [W-1, H-1, D-1] to map 0..shape-1 to -1..1
    # Construct directly to avoid potential slicing issues
    norm_factors = torch.tensor(
        [W - 1, H - 1, D - 1], device=theta.device, dtype=torch.float32)

    # Original contours must be float for transformations and grid_sample
    print(f"contours_xyz shape: {contours_xyz.shape}")
    print(f"contours_xyz dtype: {contours_xyz.dtype}")

    contours_xyz_float = torch.tensor(contours_xyz).float()
    # contours_xyz_float = contours_xyz.float()
    # --- End Changes ---

    for i in range(10000):  # Or increased iterations
        optimizer.zero_grad()

        # Calculate transformed contours (keep as float)
        transformed_contours_float = torch.bmm(
            # Use contours_xyz_float
            contours_xyz_float.unsqueeze(0), theta[:, :, :3].transpose(1, 2))
        transformed_contours_float = transformed_contours_float.squeeze(0)
        transformed_contours_float = transformed_contours_float + \
            theta[:, :, 3].squeeze(0)

        # --- Differentiable Sampling using grid_sample ---
        # Normalize coordinates to [-1, 1] range
        norm_transformed = (transformed_contours_float /
                            norm_factors) * 2.0 - 1.0
        norm_original = (contours_xyz_float / norm_factors) * \
            2.0 - 1.0  # Use contours_xyz_float

        # Reshape coordinates for grid_sample: [N, 3] -> [1, N, 1, 1, 3] (for 3D)
        # grid_sample expects grid in order (z, y, x) for input tensor (D, H, W)
        # So we reverse the last dimension: [..., (x, y, z)] -> [..., (z, y, x)]
        # [1, N, 1, 1, 3]
        grid_transformed = norm_transformed.flip(
            -1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # [1, N, 1, 1, 3]
        grid_original = norm_original.flip(
            -1).unsqueeze(0).unsqueeze(2).unsqueeze(3)

        # Sample using grid_sample (bilinear interpolation is default)
        # Input tensor shape: [B, C, D, H, W] -> [1, 1, D, H, W]
        # Output shape: [B, C, N, 1, 1] -> [1, 1, N, 1, 1]
        pred = F.grid_sample(scan_meas, grid_transformed,
                             mode='bilinear', padding_mode='border', align_corners=True)
        gt = F.grid_sample(scan_ref, grid_original, mode='bilinear',
                           padding_mode='border', align_corners=True)

        # Reshape pred/gt for loss calculation: [1, 1, N, 1, 1] -> [N]
        pred = pred.squeeze()
        gt = gt.squeeze()
        # Handle case where N=1, squeeze might remove the dimension entirely
        if pred.dim() == 0:
            pred = pred.unsqueeze(0)
        if gt.dim() == 0:
            gt = gt.unsqueeze(0)
        # --- End of grid_sample changes ---

        A = theta[:, :, :3].transpose(1, 2).squeeze(0)
        eigenvalues = torch.linalg.eigvalsh(A)  # guaranteed real-valued

        # print(f"Eigenvalues: {eigenvalues}")
        eigenval_penalty = torch.maximum(
            (1 - eigenvalues), torch.zeros_like(eigenvalues, device=eigenvalues.device))
        eigenval_penalty = eigenval_penalty.sum()
        loss = criterion(pred, gt) + eigenval_penalty

        loss.backward()  # Gradient calculation should work now
        optimizer.step()

        if i % 10 == 0:
            print(f"Step {i}: loss = {loss.item():.6f}")

    # Calculate final transformed contours outside the loop
    with torch.no_grad():  # No need for gradients here
        transformed_contours = torch.bmm(
            contours_xyz_float.unsqueeze(0), theta[:, :, :3].transpose(1, 2))
        transformed_contours = transformed_contours.squeeze(0)
        transformed_contours = transformed_contours + theta[:, :, 3].squeeze(0)

    return transformed_contours  # Return float coordinates

def find_all_contours_in_meas(scan_ref, scan_meas, contours_dict):
    transformed_contours_dict = {}
    for roi_name, contours_xyz in contours_dict.items():
        transformed_contours = find_contours_in_meas(
            scan_ref, scan_meas, contours_xyz)
        transformed_contours_dict[roi_name] = transformed_contours
    return transformed_contours_dict

# --- Example of how to use the function ---
# Note: This requires load_contours_from_txt and actual scan/contour data

# if __name__ == '__main__':
#     # 1. Load contours (assuming load_contours_from_txt exists and works)
#     contour_dir = 'path/to/your/contour/textfiles' # <<< CHANGE THIS
#     roi_name_to_use = 'YOUR_ROI_NAME' # <<< CHANGE THIS
#     try:
#         _, tensor_contours_dict = load_contours_from_txt(contour_dir)
#         if roi_name_to_use not in tensor_contours_dict:
#              raise ValueError(f"ROI '{roi_name_to_use}' not found in loaded contours.")
#         initial_contours = tensor_contours_dict[roi_name_to_use] # Should be (N, 3) -> (x, y, z)
#     except Exception as e:
#         print(f"Error loading contours: {e}. Exiting.")
#         exit()

#     # 2. Load scans (replace with your actual scan loading logic)
#     # Ensure scans are torch.Tensor with shape (D, H, W)
#     try:
#         # scan_ref = load_my_scan_data('path/to/reference_scan')
#         # scan_meas = load_my_scan_data('path/to/measurement_scan')
#         # Dummy data for demonstration:
#         D, H, W = 64, 128, 128
#         print(f"Using dummy scan data of shape ({D}, {H}, {W})")
#         scan_ref = torch.rand(D, H, W) * 255
#         scan_meas = torch.rand(D, H, W) * 255 # Ideally, scan_meas is a transformed version of scan_ref
#     except Exception as e:
#         print(f"Error loading scan data: {e}. Exiting.")
#         exit()

#     # 3. Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     scan_ref = scan_ref.to(device)
#     scan_meas = scan_meas.to(device)
#     initial_contours = initial_contours.to(device)

#     # 4. Run alignment
#     print(f"\nAligning contours for ROI: {roi_name_to_use}")
#     transformed_contours = find_contours_in_meas(scan_ref, scan_meas, initial_contours)

#     print(f"\nInitial contour points (sample):\n{initial_contours[:5]}")
#     print(f"\nTransformed contour points (sample):\n{transformed_contours[:5]}")

#     # You can now use 'transformed_contours' for further processing
