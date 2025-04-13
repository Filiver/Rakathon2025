import os
import numpy as np
import torch
import matplotlib.pyplot as plt
# Add other necessary imports if missing


def visualize_all_contours_from_txt(dir_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import numpy as np
    import hashlib

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {}

    def get_color_for_label(label):
        if label not in color_map:
            # Generate a consistent hash-based color from label
            hash_val = int(hashlib.sha256(label.encode()).hexdigest(), 16)
            r = (hash_val % 256) / 255.0
            g = ((hash_val >> 8) % 256) / 255.0
            b = ((hash_val >> 16) % 256) / 255.0
            color_map[label] = (r, g, b)
        return color_map[label]
    all_points = {}
    for filename in os.listdir(dir_path):
        print(f"Processing file: {filename}")
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    roi_name, points_str = line.strip().split(':', 1)
                    points = [list(map(float, p.split(','))) for p in points_str.strip(
                    ).split() if len(p.split(',')) == 3]
                    points = np.array(points)
                    if points.size == 0:
                        continue
                    ax.plot(points[:, 0], points[:, 1], points[:, 2],
                            label=roi_name, color=get_color_for_label(roi_name))
                    if roi_name not in all_points:
                        all_points[roi_name] = []
                    all_points[roi_name].extend(points.tolist())

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Contours from TXT Files")
    plt.tight_layout()
    plt.show()
    return all_points


def visualize_all_contours_from_dict(dict, spacing, origin):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import numpy as np
    import hashlib

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {}

    def get_color_for_label(label):
        if label not in color_map:
            # Generate a consistent hash-based color from label
            hash_val = int(hashlib.sha256(label.encode()).hexdigest(), 16)
            r = (hash_val % 256) / 255.0
            g = ((hash_val >> 8) % 256) / 255.0
            b = ((hash_val >> 16) % 256) / 255.0
            color_map[label] = (r, g, b)
        return color_map[label]
    all_points = {}
    """
    for filename in os.listdir(dir_path):
        print(f"Processing file: {filename}")
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    roi_name, points_str = line.strip().split(':', 1)
                    points = [list(map(float, p.split(','))) for p in points_str.strip(
                    ).split() if len(p.split(',')) == 3]
                    points = np.array(points)
                    if points.size == 0:
                        continue
                    ax.plot(points[:, 0], points[:, 1], points[:, 2],
                            label=roi_name, color=get_color_for_label(roi_name))
                    if roi_name not in all_points:
                        all_points[roi_name] = []
                    all_points[roi_name].extend(points.tolist())
    """
    for roi_name, points in dict.items():
        points = np.array(points)
        if points.size == 0:
            continue
        # print(points[:, :2])
        # print(points[:, :2]-origin[1:])
        # print((points[:, :2] - origin[1:]) / spacing[1:])
        ij = ((points[:, :2] - origin[1:]) / spacing[1:]).astype(int)
        # print(ij[:, :2])
        # input()
        ax.plot(ij[:, 0], ij[:, 1], points[:, 2],
                label=roi_name, color=get_color_for_label(roi_name))
        if roi_name not in all_points:
            all_points[roi_name] = []
        all_points[roi_name].extend(points.tolist())
    ax.legend()
    ax.axis('equal')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Contours")
    # set equal axes
    plt.tight_layout()
    plt.show()
    return all_points


def visualize_two_contour_dicts(dict1, dict2, spacing, origin):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def transform_points(points):
        points = np.array(points)
        if points.size == 0:
            return None
        ij = ((points[:, :2] - origin[1:]) / spacing[1:]).astype(int)
        return ij, points[:, 2]

    def plot_dict(dict_data, color):
        for roi_name, points in dict_data.items():
            transformed = transform_points(points)
            if transformed is None:
                continue
            ij, z = transformed
            ax.plot(ij[:, 0], ij[:, 1], z, label=roi_name, color=color)

    plot_dict(dict1, color='red')
    plot_dict(dict2, color='blue')

    ax.axis('equal')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Overlay of Two 3D Contour Sets")
    plt.tight_layout()
    ax.legend()
    plt.show()


def visualize_all_contours_from_dict2(contours_dict, scan_data, spacing, origin):
    """
    Visualizes contours from a dictionary onto scan slices.

    Args:
        contours_dict (dict): Dictionary containing contour data, expected to have
                              keys like 'transformed_image' mapping roi_name to
                              torch.Tensor (N, 3) of (d, h, w) image coordinates.
        scan_data (np.ndarray): The 3D scan volume (D, H, W).
        spacing (array-like): Voxel spacing (sz, sy, sx).
        origin (array-like): Scan origin (oz, oy, ox).
    """
    # Ensure scan_data is numpy
    if isinstance(scan_data, torch.Tensor):
        scan_data = scan_data.cpu().numpy()

    # Ensure origin/spacing are numpy
    origin_np = np.asarray(origin)
    spacing_np = np.asarray(spacing)

    if scan_data.ndim != 3:
        print(
            f"Error: scan_data must be 3D (D, H, W), got shape {scan_data.shape}")
        return
    if origin_np.shape != (3,) or spacing_np.shape != (3,):
        print(
            f"Error: Origin shape {origin_np.shape} or Spacing shape {spacing_np.shape} is not (3,).")
        return

    # --- Select which coordinates to visualize ---
    # Change this key if you want to visualize original_metric, transformed_metric etc.
    coord_key = 'transformed_image'
    contour_sub_dict = contours_dict.get(coord_key)

    if contour_sub_dict is None:
        print(f"Error: Key '{coord_key}' not found in contours_dict.")
        return
    if not isinstance(contour_sub_dict, dict):
        print(f"Error: Value for '{coord_key}' is not a dictionary.")
        return
    if not contour_sub_dict:
        print(f"Warning: No ROIs found in contours_dict['{coord_key}'].")
        # return # Or continue to show empty plots

    num_slices = scan_data.shape[0]  # D dimension
    # Adjust layout as needed
    plt.figure(figsize=(15, 5 * ((num_slices + 4) // 5)))

    slice_has_contour = {i: False for i in range(num_slices)}
    # slice_idx -> {roi_name: np.array(N, 2)}
    contours_per_slice = {i: {} for i in range(num_slices)}

    # --- Pre-process contours for plotting ---
    for roi_name, points_tensor in contour_sub_dict.items():
        if not isinstance(points_tensor, torch.Tensor):
            print(
                f"Warning: Skipping ROI '{roi_name}' - value is not a PyTorch tensor.")
            continue

        if points_tensor.numel() == 0:
            # print(f"Info: Skipping ROI '{roi_name}' - tensor is empty.") # Less verbose
            continue

        try:
            points_np = points_tensor.cpu().numpy()
        except Exception as e:
            print(
                f"Warning: Skipping ROI '{roi_name}' - error converting tensor to NumPy: {e}")
            continue

        # **** Critical Check for dimensionality and shape ****
        if points_np.ndim != 2 or points_np.shape[1] != 3:
            print(
                f"Warning: Skipping ROI '{roi_name}' - NumPy array shape is {points_np.shape}, expected (N, 3).")
            continue
        # **** End Critical Check ****

        # points_np contains (d, h, w) coordinates (image indices)
        # Group points by slice index (d)
        for point in points_np:
            slice_idx = int(round(point[0]))  # d coordinate is the slice index
            if 0 <= slice_idx < num_slices:
                # Get h, w coordinates (indices 1 and 2) for plotting on the slice
                hw_coords = point[1:]  # Shape (2,)
                if roi_name not in contours_per_slice[slice_idx]:
                    contours_per_slice[slice_idx][roi_name] = []
                contours_per_slice[slice_idx][roi_name].append(hw_coords)
                slice_has_contour[slice_idx] = True

    # Convert lists to numpy arrays for easier plotting
    for slice_idx in contours_per_slice:
        for roi_name in contours_per_slice[slice_idx]:
            if contours_per_slice[slice_idx][roi_name]:  # Check if list is not empty
                contours_per_slice[slice_idx][roi_name] = np.array(
                    contours_per_slice[slice_idx][roi_name])
            else:
                contours_per_slice[slice_idx][roi_name] = np.empty(
                    (0, 2))  # Use empty array if no points

    # --- Plotting ---
    plot_index = 1
    plotted_legend = False
    for slice_idx in range(num_slices):
        # Optionally skip slices with no contours or always show the slice
        # if not slice_has_contour[slice_idx]:
        #      continue

        # Adjust grid as needed
        ax = plt.subplot(5, (num_slices + 4) // 5, plot_index)
        # Display the Z slice (axial view)
        ax.imshow(scan_data[slice_idx], cmap='gray', aspect='equal')
        ax.set_title(f'Slice {slice_idx}')
        ax.set_xlabel('W index')
        ax.set_ylabel('H index')

        rois_on_slice = contours_per_slice[slice_idx]
        slice_had_roi = False
        for roi_name, hw_points in rois_on_slice.items():
            # hw_points is now (N, 2) with (h, w) coordinates
            if hw_points.ndim == 2 and hw_points.shape[0] > 0 and hw_points.shape[1] == 2:
                # Plot expects (x, y), which corresponds to (w, h) indices for imshow
                ax.plot(hw_points[:, 1], hw_points[:, 0], '.',
                        label=roi_name, markersize=2)  # Plot w vs h
                slice_had_roi = True
            # else: # Debugging if needed
                # print(f"Debug: ROI {roi_name} on slice {slice_idx} has unexpected shape {hw_points.shape if isinstance(hw_points, np.ndarray) else 'Not Array'}")

        # Add legend only to the first plot that actually has contours
        if slice_had_roi and not plotted_legend:
            ax.legend()
            plotted_legend = True

        plot_index += 1

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import os
    dir = "pointclouds_by_rs/SAMPLE_004"
    subdirs = os.listdir(dir)
    for subdir in subdirs:
        if subdir != "RS.1.2.246.352.221.52794105832653520384075859529424384185__RS.1.2.246.352.221.57475698521031836325890889930332779148__RS.1.2.246.352.221.530968562667814550516230413739928631461__RS.1.2.246.352.221.534409961817902190914559599786692832400":
            continue
        print(f"Processing subdirectory: {subdir}")
        all_points = visualize_all_contours_from_txt(
            os.path.join(dir, subdir, "txt"))
        for roi_name, points_list in all_points.items():
            print(f"ROI: {roi_name}")
            print(points_list)
            input()
