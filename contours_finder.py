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
    # norm_factors = torch.tensor(
    #     [W - 1, H - 1, D - 1], device=theta.device, dtype=torch.float32)

    # Original contours must be float for transformations and grid_sample
    print(f"contours_xyz shape: {contours_xyz.shape}")
    print(f"contours_xyz dtype: {contours_xyz.dtype}")

    contours_xyz_float = torch.tensor(contours_xyz).float()
    # contours_xyz_float = contours_xyz.float()
    # --- End Changes ---

    for i in range(100):  # Or increased iterations
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
        # norm_transformed = (transformed_contours_float /
        #                     norm_factors) * 2.0 - 1.0
        # norm_original = (contours_xyz_float / norm_factors) * \
        #     2.0 - 1.0  # Use contours_xyz_float

        # Reshape coordinates for grid_sample: [N, 3] -> [1, N, 1, 1, 3] (for 3D)
        # grid_sample expects grid in order (z, y, x) for input tensor (D, H, W)
        # So we reverse the last dimension: [..., (x, y, z)] -> [..., (z, y, x)]
        # [1, N, 1, 1, 3]
        grid_transformed = transformed_contours_float.flip(
            -1).unsqueeze(0).unsqueeze(2).unsqueeze(3)
        # # [1, N, 1, 1, 3]
        grid_original = contours_xyz_float.flip(
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



def find_contours_in_meas_my(scan_ref, scan_meas, spacing, origin, contours_xyz_input):
    device = scan_ref.device
    dtype = torch.float32

    # --- Input Handling ---
    # Ensure scans are 5D: [B, C, D, H, W]
    scan_ref = scan_ref.unsqueeze(0).unsqueeze(0) if scan_ref.dim() == 3 else scan_ref
    scan_meas = scan_meas.unsqueeze(0).unsqueeze(0) if scan_meas.dim() == 3 else scan_meas
    if scan_ref.dim() != 5 or scan_meas.dim() != 5:
        raise ValueError("Input scans must be 5D (B, C, D, H, W)")

    # Ensure spacing and origin are tensors on the correct device
    spacing = torch.as_tensor(spacing, dtype=dtype, device=device)
    origin = torch.as_tensor(origin, dtype=dtype, device=device)

    # Ensure contours are tensor (N, 3) on the correct device
    contours_xyz_metric = torch.as_tensor(contours_xyz_input, dtype=dtype, device=device)
    if contours_xyz_metric.dim() != 2 or contours_xyz_metric.shape[1] != 3:
         raise ValueError(f"Input contours must have shape (N, 3), got {contours_xyz_metric.shape}")
    num_points = contours_xyz_metric.shape[0]

    # --- Coordinate System Setup ---
    # Scan shape (D, H, W)
    D, H, W = scan_ref.shape[2:]
    scan_shape_dhw = torch.tensor([D, H, W], device=device, dtype=dtype)

    # Convert spacing/origin to (x, y, z) order to match contour coordinates
    spacing_xyz = spacing.flip(-1) # (z, y, x) -> (x, y, z)
    origin_xyz = origin.flip(-1) # (z, y, x) -> (x, y, z)
    scan_shape_whd = torch.tensor([W, H, D], device=device, dtype=dtype) # Use (W, H, D) order

    # --- Normalization (for align_corners=True) ---
    # Divisor: (shape - 1) * spacing. Clamp to avoid division by zero for dims of size 1.
    norm_divisor = (scan_shape_whd - 1.0).clamp(min=1e-6) * spacing_xyz
    # Add batch/channel dims for broadcasting: [1, 1, 1, 1, 3]
    origin_b = origin_xyz.view(1, 1, 1, 1, 3)
    norm_divisor_b = norm_divisor.view(1, 1, 1, 1, 3)

    # Reshape metric contours for broadcasting and grid_sample: [1, 1, 1, N, 3]
    contours_xyz_metric_b = contours_xyz_metric.view(1, 1, 1, num_points, 3)

    # Normalize metric coordinates to [-1, 1] range
    # Formula: normalized = ((metric - origin) / norm_divisor) * 2.0 - 1.0
    contours_xyz_normalized = ((contours_xyz_metric_b - origin_b) / norm_divisor_b) * 2.0 - 1.0
    # Flip last dim -> (z, y, x) order for grid_sample
    grid_normalized = contours_xyz_normalized.flip(-1) # Shape: [1, 1, 1, N, 3], order (z, y, x)

    # --- Optimization Setup ---
    # Initialize theta closer to identity
    theta_init = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0)
    theta = torch.nn.Parameter(theta_init.clone()) # Use clone for safety
    optimizer = optim.AdamW([theta], lr=0.0001) # Maybe adjust LR
    criterion = torch.nn.MSELoss()

    # --- Precompute Reference Intensities ---
    # Use align_corners=True as normalization matches it
    ref_intensities = F.grid_sample(scan_ref, grid_normalized, mode='bilinear', padding_mode='zeros', align_corners=True)
    # ref_intensities shape: [1, 1, 1, N, 1] -> squeeze -> [1, 1, N] ? Check shape

    print(f"Initial theta:\n{theta.data}")

    # --- Optimization Loop ---
    for i in range(1000): # Increased iterations might be needed
        optimizer.zero_grad()

        # Get current transformation matrix M and translation t
        M = theta[:, :, :3] # Shape: [1, 3, 3]
        t = theta[:, :, 3:] # Shape: [1, 3, 1]

        # Apply transformation to *normalized* coordinates (x, y, z)
        # Input contours_xyz_normalized: [1, 1, 1, N, 3]
        # Reshape for matmul: [1, N, 3]
        contours_norm_reshaped = contours_xyz_normalized.squeeze(1).squeeze(1) # [1, N, 3]
        # Apply rotation/scaling (M)
        # M [1, 3, 3] @ contours_norm_reshaped.transpose(1, 2) [1, 3, N] -> [1, 3, N]
        transformed_norm_reshaped = M @ contours_norm_reshaped.transpose(1, 2)
        # Apply translation (t)
        # transformed_norm_reshaped [1, 3, N] + t [1, 3, 1] -> [1, 3, N] (broadcast t)
        transformed_norm_reshaped = transformed_norm_reshaped + t
        # Transpose back and reshape for grid_sample: [1, 3, N] -> [1, N, 3] -> [1, 1, 1, N, 3]
        transformed_contours_normalized = transformed_norm_reshaped.transpose(1, 2).view(1, 1, 1, num_points, 3)

        # Create grid for sampling: flip last dim -> (z, y, x) order
        grid_transformed_normalized = transformed_contours_normalized.flip(-1) # Shape: [1, 1, 1, N, 3]

        # Sample measurement scan at transformed normalized coordinates
        mes_intensities = F.grid_sample(scan_meas, grid_transformed_normalized, mode='bilinear', padding_mode='zeros', align_corners=True)

        # Calculate loss
        # Ensure shapes match, e.g., squeeze unnecessary dims if needed
        loss = criterion(ref_intensities, mes_intensities)

        # Regularization (optional, e.g., prevent excessive scaling/shear)
        # Example: Penalize deviation of M from identity or orthogonality
        # reg_loss = torch.mean((M @ M.transpose(1, 2) - torch.eye(3, device=device))**2)
        # total_loss = loss + 0.01 * reg_loss # Add weighted regularization
        total_loss = loss

        total_loss.backward()
        optimizer.step()

        if i % 20 == 0: # Print less often
            print(f"Step {i}: loss = {loss.item():.6f}")
            # print(f"Theta:\n{theta.data}") # Optional: print theta updates

    # --- Final Transformation and Un-normalization ---
    with torch.no_grad():
        # Get final transformation
        M = theta[:, :, :3]
        t = theta[:, :, 3:]

        # Apply final transformation to original *normalized* coordinates
        contours_norm_reshaped = contours_xyz_normalized.squeeze(1).squeeze(1)
        transformed_norm_reshaped = (M @ contours_norm_reshaped.transpose(1, 2)) + t
        final_transformed_normalized = transformed_norm_reshaped.transpose(1, 2) # Shape: [1, N, 3]

        # Un-normalize: Convert back from [-1, 1] to metric coordinates
        # Inverse formula: metric = ((normalized + 1.0) / 2.0) * norm_divisor + origin
        # Reshape norm_divisor and origin for broadcasting with [1, N, 3]
        norm_divisor_rs = norm_divisor.view(1, 1, 3)
        origin_rs = origin_xyz.view(1, 1, 3)
        # Apply inverse normalization
        final_transformed_metric = ((final_transformed_normalized + 1.0) / 2.0) * norm_divisor_rs + origin_rs

        # Squeeze batch dimension -> [N, 3]
        final_transformed_metric = final_transformed_metric.squeeze(0)

    print(f"Final loss: {loss.item():.6f}")
    print(f"Final theta:\n{theta.data}")
    return final_transformed_metric # Return metric coordinates


def find_all_contours_in_meas(scan_ref, scan_meas, spacing, origin, contours_dict):
    transformed_contours_dict = {}
    for roi_name, contours_xyz in contours_dict.items():
        transformed_contours = find_contours_in_meas_my(
            scan_ref, scan_meas, spacing, origin, contours_xyz)
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
