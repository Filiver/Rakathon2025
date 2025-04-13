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
    """
    Finds the transformation that aligns contours from a reference scan space
    to a measurement scan space using intensity profile matching.

    Args:
        scan_ref (torch.Tensor): Reference scan volume (D, H, W) or (B, C, D, H, W).
        scan_meas (torch.Tensor): Measurement scan volume (D, H, W) or (B, C, D, H, W).
        spacing (tuple or list or torch.Tensor): Voxel spacing (sz, sy, sx).
        origin (tuple or list or torch.Tensor): Scan origin in metric coordinates (oz, oy, ox).
        contours_xyz_input (torch.Tensor or list): Initial contour points (N, 3) in metric coordinates (x, y, z).

    Returns:
        torch.Tensor: Transformed contour points (N, 3) in metric coordinates (x, y, z).
    """
    device = scan_ref.device
    dtype = torch.float32

    # --- Input Handling ---
    # Ensure scans are 5D: [B, C, D, H, W]
    scan_ref = scan_ref.unsqueeze(0).unsqueeze(0) if scan_ref.dim() == 3 else scan_ref
    scan_meas = scan_meas.unsqueeze(0).unsqueeze(0) if scan_meas.dim() == 3 else scan_meas
    if scan_ref.dim() != 5 or scan_meas.dim() != 5:
        raise ValueError("Input scans must be 5D (B, C, D, H, W)")

    # Ensure spacing and origin are tensors on the correct device (z, y, x order)
    spacing_zyx = torch.as_tensor(spacing, dtype=dtype, device=device)
    origin_zyx = torch.as_tensor(origin, dtype=dtype, device=device)
    if spacing_zyx.shape != (3,) or origin_zyx.shape != (3,):
        raise ValueError("Spacing and origin must have 3 elements (z, y, x)")

    # Ensure contours are tensor (N, 3) on the correct device (x, y, z order)
    contours_xyz_metric = torch.as_tensor(contours_xyz_input, dtype=dtype, device=device)
    if contours_xyz_metric.dim() != 2 or contours_xyz_metric.shape[1] != 3:
         raise ValueError(f"Input contours must have shape (N, 3), got {contours_xyz_metric.shape}")
    num_points = contours_xyz_metric.shape[0]

    # --- Coordinate System Setup ---
    # Scan shape (D, H, W)
    D, H, W = scan_ref.shape[2:]
    # Use (W, H, D) order for normalization calculations related to (x, y, z) metric coords
    scan_shape_whd = torch.tensor([W, H, D], device=device, dtype=dtype)
    # Convert spacing/origin to (x, y, z) order for normalization calculations
    spacing_xyz = spacing_zyx.flip(-1) # (z, y, x) -> (x, y, z)
    origin_xyz = origin_zyx.flip(-1) # (z, y, x) -> (x, y, z)


    # --- Normalization (for align_corners=True) ---
    # Divisor: (shape - 1) * spacing. Clamp to avoid division by zero for dims of size 1.
    # Use WHD shape and XYZ spacing/origin here
    norm_divisor_xyz = (scan_shape_whd - 1.0).clamp(min=1e-6) * spacing_xyz
    # Add batch/channel dims for broadcasting: [1, 1, 1, 1, 3]
    origin_xyz_b = origin_xyz.view(1, 1, 1, 1, 3)
    norm_divisor_xyz_b = norm_divisor_xyz.view(1, 1, 1, 1, 3)

    # Reshape metric contours for broadcasting and grid_sample: [1, 1, 1, N, 3] (x, y, z order)
    contours_xyz_metric_b = contours_xyz_metric.view(1, 1, 1, num_points, 3)

    # Normalize metric coordinates (x, y, z) to [-1, 1] range
    # Formula: normalized = ((metric - origin) / norm_divisor) * 2.0 - 1.0
    contours_xyz_normalized = ((contours_xyz_metric_b - origin_xyz_b) / norm_divisor_xyz_b) * 2.0 - 1.0

    # Flip last dim -> (z, y, x) order for grid_sample input grid
    grid_normalized_zyx = contours_xyz_normalized.flip(-1) # Shape: [1, 1, 1, N, 3], order (z, y, x)

    # --- Optimization Setup ---
    # Initialize theta closer to identity
    theta_init = torch.eye(3, 4, device=device, dtype=dtype).unsqueeze(0)
    theta = torch.nn.Parameter(theta_init.clone()) # Use clone for safety
    optimizer = optim.AdamW([theta], lr=0.0001) # Maybe adjust LR
    criterion = torch.nn.MSELoss()

    # --- Precompute Reference Intensities ---
    # Use align_corners=True as normalization matches it
    # Sample scan_ref using the normalized grid (z, y, x order)
    ref_intensities = F.grid_sample(scan_ref, grid_normalized_zyx, mode='bilinear', padding_mode='zeros', align_corners=True)
    # ref_intensities shape: [1, 1, 1, N, 1]

    print(f"Initial theta:\n{theta.data}")

    # --- Optimization Loop ---
    for i in range(1000): # Increased iterations might be needed
        optimizer.zero_grad()

        # Get current transformation matrix M and translation t
        M = theta[:, :, :3] # Shape: [1, 3, 3] (operates on x, y, z)
        t = theta[:, :, 3:] # Shape: [1, 3, 1] (translation in x, y, z)

        # Apply transformation to *normalized* coordinates (x, y, z)
        # Input contours_xyz_normalized: [1, 1, 1, N, 3]
        # Reshape for matmul: [1, N, 3]
        contours_norm_reshaped_xyz = contours_xyz_normalized.squeeze(1).squeeze(1) # [1, N, 3]
        # Apply rotation/scaling (M)
        # M [1, 3, 3] @ contours_norm_reshaped_xyz.transpose(1, 2) [1, 3, N] -> [1, 3, N]
        transformed_norm_reshaped_xyz = M @ contours_norm_reshaped_xyz.transpose(1, 2)
        # Apply translation (t)
        # transformed_norm_reshaped_xyz [1, 3, N] + t [1, 3, 1] -> [1, 3, N] (broadcast t)
        transformed_norm_reshaped_xyz = transformed_norm_reshaped_xyz + t
        # Transpose back and reshape for grid_sample: [1, 3, N] -> [1, N, 3] -> [1, 1, 1, N, 3]
        transformed_contours_normalized_xyz = transformed_norm_reshaped_xyz.transpose(1, 2).view(1, 1, 1, num_points, 3)

        # Create grid for sampling: flip last dim -> (z, y, x) order
        grid_transformed_normalized_zyx = transformed_contours_normalized_xyz.flip(-1) # Shape: [1, 1, 1, N, 3]

        # Sample measurement scan at transformed normalized coordinates
        mes_intensities = F.grid_sample(scan_meas, grid_transformed_normalized_zyx, mode='bilinear', padding_mode='zeros', align_corners=True)
        # mes_intensities shape: [1, 1, 1, N, 1]

        # Calculate loss
        loss = criterion(ref_intensities, mes_intensities)

        # Regularization (optional)
        # reg_loss = torch.mean((M @ M.transpose(1, 2) - torch.eye(3, device=device))**2)
        # total_loss = loss + 0.01 * reg_loss
        total_loss = loss

        total_loss.backward()
        optimizer.step()

        if i % 100 == 0: # Print less often
            print(f"Step {i}: loss = {loss.item():.6f}")
            # print(f"Theta:\n{theta.data}") # Optional: print theta updates

    # --- Final Transformation and Un-normalization ---
    with torch.no_grad():
        # Get final transformation
        M = theta[:, :, :3]
        t = theta[:, :, 3:]

        # Apply final transformation to original *normalized* coordinates (x, y, z)
        contours_norm_reshaped_xyz = contours_xyz_normalized.squeeze(1).squeeze(1) # [1, N, 3]
        transformed_norm_reshaped_xyz = (M @ contours_norm_reshaped_xyz.transpose(1, 2)) + t # [1, 3, N]
        final_transformed_normalized_xyz = transformed_norm_reshaped_xyz.transpose(1, 2) # Shape: [1, N, 3]

        # Un-normalize: Convert back from [-1, 1] (x, y, z) to metric coordinates (x, y, z)
        # Inverse formula: metric = ((normalized + 1.0) / 2.0) * norm_divisor + origin
        # Reshape norm_divisor_xyz and origin_xyz for broadcasting with [1, N, 3]
        norm_divisor_xyz_rs = norm_divisor_xyz.view(1, 1, 3)
        origin_xyz_rs = origin_xyz.view(1, 1, 3)
        # Apply inverse normalization
        final_transformed_metric_xyz = ((final_transformed_normalized_xyz + 1.0) / 2.0) * norm_divisor_xyz_rs + origin_xyz_rs

        # Squeeze batch dimension -> [N, 3]
        final_transformed_metric_xyz = final_transformed_metric_xyz.squeeze(0)

    print(f"Final loss: {loss.item():.6f}")
    print(f"Final theta:\n{theta.data}")
    return final_transformed_metric_xyz # Return metric coordinates (x, y, z)


def metric_to_image_coords(metric_coords_xyz, origin_zyx, spacing_zyx):
    """Converts metric coordinates (x, y, z) to image coordinates (d, h, w)."""
    if not isinstance(metric_coords_xyz, torch.Tensor):
        metric_coords_xyz = torch.as_tensor(metric_coords_xyz, dtype=torch.float32)
    if not isinstance(origin_zyx, torch.Tensor):
        origin_zyx = torch.as_tensor(origin_zyx, dtype=torch.float32, device=metric_coords_xyz.device)
    if not isinstance(spacing_zyx, torch.Tensor):
        spacing_zyx = torch.as_tensor(spacing_zyx, dtype=torch.float32, device=metric_coords_xyz.device)

    # Ensure inputs have correct shapes
    if metric_coords_xyz.dim() != 2 or metric_coords_xyz.shape[1] != 3:
        raise ValueError(f"metric_coords_xyz must have shape (N, 3), got {metric_coords_xyz.shape}")
    if origin_zyx.shape != (3,) or spacing_zyx.shape != (3,):
        raise ValueError("origin_zyx and spacing_zyx must have shape (3,)")

    # Flip metric coords from (x, y, z) to (z, y, x) to match origin/spacing order
    metric_coords_zyx = metric_coords_xyz.flip(-1) # Shape: (N, 3)

    # Calculate image coordinates (voxel indices)
    # image_coord = (metric_coord - origin) / spacing
    # Broadcasting handles (N, 3) - (3,) / (3,) -> (N, 3)
    image_coords_zyx = (metric_coords_zyx - origin_zyx) / spacing_zyx

    # Result is in (z, y, x) order, which corresponds to (d, h, w) indices
    return image_coords_zyx


def find_all_contours_in_meas(scan_ref, scan_meas, spacing, origin, contours_dict):
    """
    Finds transformed contours for multiple ROIs and returns original and
    transformed contours in both metric and image coordinates.

    Args:
        scan_ref (torch.Tensor): Reference scan volume (D, H, W).
        scan_meas (torch.Tensor): Measurement scan volume (D, H, W).
        spacing (tuple or list or torch.Tensor): Voxel spacing (sz, sy, sx).
        origin (tuple or list or torch.Tensor): Scan origin in metric coordinates (oz, oy, ox).
        contours_dict (dict): Dictionary mapping ROI names (str) to original
                               contour points (torch.Tensor or list, shape (N, 3), metric coords x, y, z).

    Returns:
        dict: A dictionary containing four sub-dictionaries:
              'original_metric': {roi_name: tensor (N, 3) metric (x, y, z)}
              'transformed_metric': {roi_name: tensor (N, 3) metric (x, y, z)}
              'original_image': {roi_name: tensor (N, 3) image (d, h, w)}
              'transformed_image': {roi_name: tensor (N, 3) image (d, h, w)}
    """
    results = {
        'original_metric': {},
        'transformed_metric': {},
        'original_image': {},
        'transformed_image': {}
    }

    # Ensure spacing and origin are tensors on the correct device
    # Keep them in (z, y, x) order as required by metric_to_image_coords
    device = scan_ref.device
    spacing_zyx = torch.as_tensor(spacing, dtype=torch.float32, device=device)
    origin_zyx = torch.as_tensor(origin, dtype=torch.float32, device=device)

    for roi_name, contours_xyz_metric_orig in contours_dict.items():
        print(f"\n--- Processing ROI: {roi_name} ---")
        # Ensure original contours are tensor on the correct device
        contours_xyz_metric_orig = torch.as_tensor(contours_xyz_metric_orig, dtype=torch.float32, device=device)
        if contours_xyz_metric_orig.dim() != 2 or contours_xyz_metric_orig.shape[1] != 3:
            print(f"Warning: Skipping ROI '{roi_name}' due to invalid contour shape: {contours_xyz_metric_orig.shape}")
            continue

        # Store original metric coordinates
        results['original_metric'][roi_name] = contours_xyz_metric_orig

        # Find transformed metric coordinates using the optimization function
        transformed_contours_xyz_metric = find_contours_in_meas_my(
            scan_ref, scan_meas, spacing_zyx, origin_zyx, contours_xyz_metric_orig)
        results['transformed_metric'][roi_name] = transformed_contours_xyz_metric

        # Convert original metric coordinates to image coordinates (d, h, w)
        original_contours_dhw_image = metric_to_image_coords(
            contours_xyz_metric_orig, origin_zyx, spacing_zyx)
        results['original_image'][roi_name] = original_contours_dhw_image

        # Convert transformed metric coordinates to image coordinates (d, h, w)
        transformed_contours_dhw_image = metric_to_image_coords(
            transformed_contours_xyz_metric, origin_zyx, spacing_zyx)
        results['transformed_image'][roi_name] = transformed_contours_dhw_image

        print(f"Finished ROI: {roi_name}")
        print(f"  Original metric sample: {results['original_metric'][roi_name][0]}")
        print(f"  Transformed metric sample: {results['transformed_metric'][roi_name][0]}")
        print(f"  Original image sample: {results['original_image'][roi_name][0]}")
        print(f"  Transformed image sample: {results['transformed_image'][roi_name][0]}")


    return results

# --- Example of how to use the updated function ---
# Note: This requires load_contours_from_txt and actual scan/contour data

# if __name__ == '__main__':
#     # 1. Load contours
#     contour_dir = 'path/to/your/contour/textfiles' # <<< CHANGE THIS
#     try:
#         _, tensor_contours_dict = load_contours_from_txt(contour_dir) # Dict: roi_name -> tensor (N, 3) (x, y, z)
#         if not tensor_contours_dict:
#              raise ValueError(f"No contours loaded from {contour_dir}")
#     except Exception as e:
#         print(f"Error loading contours: {e}. Exiting.")
#         exit()

#     # 2. Load scans and metadata (replace with your actual loading logic)
#     try:
#         # scan_ref = load_my_scan_data('path/to/reference_scan') # Shape (D, H, W)
#         # scan_meas = load_my_scan_data('path/to/measurement_scan') # Shape (D, H, W)
#         # spacing = load_my_spacing('path/to/reference_scan') # Tuple/List (sz, sy, sx)
#         # origin = load_my_origin('path/to/reference_scan') # Tuple/List (oz, oy, ox)

#         # Dummy data for demonstration:
#         D, H, W = 64, 128, 128
#         print(f"Using dummy scan data of shape ({D}, {H}, {W})")
#         scan_ref = torch.rand(D, H, W) * 100 # Reference
#         # Create a slightly shifted/scaled measurement scan for testing
#         grid_shift = torch.tensor([[[0.05, 0, 0, -5], # Shift x slightly
#                                     [0, 1.0, 0, 0], # Keep y
#                                     [0, 0, 1.0, 0]]], dtype=torch.float32) # Keep z
#         grid = F.affine_grid(grid_shift, (1, 1, D, H, W), align_corners=True)
#         scan_meas = F.grid_sample(scan_ref.view(1, 1, D, H, W), grid, align_corners=True).squeeze()

#         spacing = (3.0, 1.5, 1.5) # Example spacing (z, y, x)
#         origin = (-90.0, -150.0, -150.0) # Example origin (z, y, x)

#     except Exception as e:
#         print(f"Error loading scan data/metadata: {e}. Exiting.")
#         exit()

#     # 3. Set device
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
#     scan_ref = scan_ref.to(device)
#     scan_meas = scan_meas.to(device)
#     # Ensure contours in the dict are moved to device within the main function

#     # 4. Run alignment for all contours
#     print(f"\nAligning all contours...")
#     all_contours_results = find_all_contours_in_meas(
#         scan_ref,
#         scan_meas,
#         spacing,
#         origin,
#         tensor_contours_dict # Pass the whole dictionary
#     )

#     print("\n--- Results Summary ---")
#     for roi_name in all_contours_results['original_metric']:
#         print(f"ROI: {roi_name}")
#         print(f"  Num points: {all_contours_results['original_metric'][roi_name].shape[0]}")
#         # Access results like:
#         # original_metric_coords = all_contours_results['original_metric'][roi_name]
#         # transformed_image_coords = all_contours_results['transformed_image'][roi_name]
#         # print(f"  Transformed Image Coords Sample:\n{transformed_image_coords[:3]}")

#     # You can now use the 'all_contours_results' dictionary which contains
#     # 'original_metric', 'transformed_metric', 'original_image', 'transformed_image'
#     # each being a dictionary mapping roi_name to the corresponding tensor.