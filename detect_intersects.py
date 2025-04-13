from shapely.geometry import Polygon
import torch  # Assuming tensors are PyTorch tensors

dangerous = [("spinalcord", "ptv_low"), ("spinalcord", "ptv_high"),
             ("ptv_low", "ctv_low")]


def detect_intersect(contours_meas_torch_dict, overlap_threshold_percent=0.0):
    """
    Detects intersections between specified contour pairs within a single dataset
    and quantifies the overlap.

    Args:
        contours_meas_torch_dict (dict): Dictionary containing contour data,
                                         expected to have a "binned_z_transform" key
                                         where the value is {contour_name: {slice_idx: tensor}}.
                                         Coordinates are assumed to be in mm.
        overlap_threshold_percent (float): Minimum overlap percentage (relative to the
                                           larger contour) to consider as an intersection.
                                           Defaults to 0.0 (any overlap).

    Returns:
        dict: A dictionary where keys are the dangerous pairs (tuples) and
              values are lists of tuples, each containing
              (slice_index, overlap_percentage, intersection_area_mm2).
              Only includes pairs and slices where overlap exceeds the threshold.
    """
    intersections = {}
    # Check if the main key exists
    if "binned_z_transform" not in contours_meas_torch_dict:
        print("Error: 'binned_z_transform' key not found in input dictionary.")
        return intersections

    binned_data = contours_meas_torch_dict["binned_z_transform"]

    for pair in dangerous:
        # Check if both contours in the pair exist in the data
        if pair[0] not in binned_data or pair[1] not in binned_data:
            print(
                f"Warning: Contours for pair {pair} not found in 'binned_z_transform'. Skipping.")
            continue

        contours1_dict = binned_data[pair[0]]
        contours2_dict = binned_data[pair[1]]

        # Find common slices for this pair
        common_slices = set(contours1_dict.keys()) & set(contours2_dict.keys())

        # Store tuples of (slice_idx, overlap_percent, intersection_area_mm2)
        intersecting_slices_data = []
        for slice_idx in common_slices:
            # Ensure slice exists for both (redundant check due to common_slices, but safe)
            if slice_idx not in contours1_dict or slice_idx not in contours2_dict:
                continue

            points1_tensor = contours1_dict[slice_idx]
            points2_tensor = contours2_dict[slice_idx]

            # Convert tensors to lists of tuples for Shapely
            try:
                points1_list = points1_tensor.cpu().numpy().tolist()
                points2_list = points2_tensor.cpu().numpy().tolist()
            except Exception as e:
                print(
                    f"Error converting tensors to lists for slice {slice_idx}, pair {pair}: {e}")
                continue

            # Need at least 3 points to form a polygon
            if len(points1_list) < 3 or len(points2_list) < 3:
                continue

            try:
                poly1 = Polygon(points1_list)
                poly2 = Polygon(points2_list)

                # Check if the polygons are valid and attempt to fix if not
                if not poly1.is_valid:
                    poly1 = poly1.buffer(0)
                if not poly2.is_valid:
                    poly2 = poly2.buffer(0)

                # Ensure polygons are still valid after buffer(0) and have non-zero area
                if not poly1.is_valid or not poly2.is_valid or poly1.area == 0 or poly2.area == 0:
                    continue

                # Check intersection and calculate overlap percentage
                if poly1.intersects(poly2):
                    intersection_poly = poly1.intersection(poly2)
                    # Ensure intersection is a Polygon
                    if isinstance(intersection_poly, Polygon):
                        intersection_area_mm2 = intersection_poly.area
                    else:
                        intersection_area_mm2 = 0.0

                    # Calculate overlap relative to the area of the LARGER polygon
                    larger_area = max(poly1.area, poly2.area)
                    if larger_area > 0:
                        overlap_percent = (
                            intersection_area_mm2 / larger_area) * 100
                    else:
                        overlap_percent = 0.0

                    if overlap_percent >= overlap_threshold_percent:
                        intersecting_slices_data.append(
                            (slice_idx, overlap_percent, intersection_area_mm2))

            except Exception as e:
                print(
                    f"Error processing polygons on slice {slice_idx} for pair {pair}: {e}")
                continue

        if intersecting_slices_data:
            # Sort by slice index
            intersections[pair] = sorted(
                intersecting_slices_data, key=lambda x: x[0])

    return intersections


def compare_contour_sets(binned_z_transform, binned_z_original, overlap_threshold_percent=0.0):
    """
    Compares contours with the same name between two different contour sets
    (e.g., transformed vs. original) and quantifies their overlap per slice
    and overall volume overlap.

    Args:
        binned_z_transform (dict): Dictionary where keys are contour names (str)
                                   and values are dicts mapping slice indices (int)
                                   to contour point tensors (PyTorch Tensor).
                                   Coordinates are assumed to be in mm.
        binned_z_original (dict): Dictionary with the same structure as
                                  binned_z_transform, representing the second set
                                  of contours (e.g., original).
        overlap_threshold_percent (float): Minimum per-slice overlap percentage (relative
                                           to the larger contour) to consider for inclusion
                                           in the slice list. Defaults to 0.0.

    Returns:
        dict: A dictionary where keys are the common contour names (str) found
              in both input dictionaries. Values are tuples:
              (list_of_slice_data, overall_volume_overlap_percent).
              - list_of_slice_data: List of tuples, each containing
                (slice_index, overlap_percentage, intersection_area_mm2).
                Only includes slices where per-slice overlap exceeds the threshold.
              - overall_volume_overlap_percent: Float representing the percentage
                calculated as (sum of intersection areas across all common slices) /
                (sum of original contour areas across all common slices) * 100.
                Returns 0.0 if the sum of original areas is zero.
    """
    comparison_results = {}

    # Find common contour names (keys) in both dictionaries
    common_contour_names = set(binned_z_transform.keys()) & set(
        binned_z_original.keys())

    if not common_contour_names:
        print("Warning: No common contour names found between the two input dictionaries.")
        return comparison_results

    for contour_name in common_contour_names:
        # Safely get the dictionaries for the current contour name
        contours_transform_dict = binned_z_transform.get(contour_name, {})
        contours_original_dict = binned_z_original.get(contour_name, {})

        # Find common slices for this specific contour name
        common_slices = set(contours_transform_dict.keys()) & set(
            contours_original_dict.keys())

        # Store tuples of (slice_idx, overlap_percent, intersection_area_mm2)
        slice_comparison_data = []
        total_intersection_area = 0.0
        total_original_area = 0.0

        for slice_idx in common_slices:
            # Check if slice_idx exists in both dictionaries for the current contour
            if slice_idx not in contours_transform_dict or slice_idx not in contours_original_dict:
                continue

            points_transform_tensor = contours_transform_dict[slice_idx]
            points_original_tensor = contours_original_dict[slice_idx]

            # Convert tensors to lists of tuples for Shapely
            try:
                points_transform_list = points_transform_tensor.cpu().numpy().tolist()
                points_original_list = points_original_tensor.cpu().numpy().tolist()
            except Exception as e:
                print(
                    f"Error converting tensors to lists for slice {slice_idx}, contour '{contour_name}': {e}")
                continue

            # Need at least 3 points to form a polygon
            if len(points_transform_list) < 3 or len(points_original_list) < 3:
                continue

            try:
                poly_transform = Polygon(points_transform_list)
                poly_original = Polygon(points_original_list)

                # Check if the polygons are valid and attempt to fix if not
                if not poly_transform.is_valid:
                    poly_transform = poly_transform.buffer(0)
                if not poly_original.is_valid:
                    poly_original = poly_original.buffer(0)

                # Ensure polygons are still valid after buffer(0) and have non-zero area
                # We need original area even if transform area is zero for volume calc
                if not poly_original.is_valid or poly_original.area == 0:
                    continue  # Skip slice if original is invalid or has no area

                # Add original area to total for volume calculation
                total_original_area += poly_original.area

                # Check transform polygon validity for intersection calculation
                if not poly_transform.is_valid or poly_transform.area == 0:
                    continue  # Skip intersection calculation if transform is invalid/zero area

                # Check intersection and calculate overlap percentage and area
                current_intersection_area_mm2 = 0.0
                if poly_transform.intersects(poly_original):
                    intersection_poly = poly_transform.intersection(
                        poly_original)
                    if isinstance(intersection_poly, Polygon):
                        current_intersection_area_mm2 = intersection_poly.area

                # Add current intersection area to total for volume calculation
                total_intersection_area += current_intersection_area_mm2

                # Calculate per-slice overlap percentage relative to the LARGER polygon
                larger_area = max(poly_transform.area, poly_original.area)
                if larger_area > 0:
                    overlap_percent = (
                        current_intersection_area_mm2 / larger_area) * 100
                else:
                    overlap_percent = 0.0

                # Add to slice list only if per-slice threshold is met
                if overlap_percent >= overlap_threshold_percent:
                    slice_comparison_data.append(
                        (slice_idx, overlap_percent, current_intersection_area_mm2))

            except Exception as e:
                print(
                    f"Error processing polygons on slice {slice_idx} for contour '{contour_name}': {e}")
                continue

        # Calculate overall volume overlap percentage after processing all slices for the contour
        overall_volume_overlap_percent = 0.0
        if total_original_area > 0:
            overall_volume_overlap_percent = (
                total_intersection_area / total_original_area) * 100

        # Store results if there's any slice data or if volume overlap is meaningful
        # (Adjust this condition based on whether you want entries even with 0 overlap)
        if slice_comparison_data or overall_volume_overlap_percent > 0:
            # Sort slice data by slice index
            sorted_slice_data = sorted(
                slice_comparison_data, key=lambda x: x[0])
            comparison_results[contour_name] = (
                sorted_slice_data, overall_volume_overlap_percent)

    return comparison_results

# Example Usage for detect_intersect (assuming contours_meas_torch_dict is populated):
# threshold = 5.0
# intersections_found = detect_intersect(contours_meas_torch_dict, overlap_threshold_percent=threshold)
# if intersections_found:
#     print(f"Found intersections between dangerous pairs (overlap > {threshold}%):")
#     for pair, slice_data in intersections_found.items():
#         print(f"  {pair[0]} and {pair[1]} intersect on slices:")
#         for slice_idx, overlap, area in slice_data: # Unpack the tuple
#             print(f"    Slice {slice_idx}: {overlap:.2f}% overlap, {area:.2f} mm^2 area")
# else:
#     print(f"No intersections found between dangerous pairs with overlap > {threshold}%.")


# Example Usage for compare_contour_sets:
# Assuming binned_z_transform_data and binned_z_original_data are populated dicts
# with the structure described in the docstring.
# threshold_comp = 1.0 # Per-slice threshold
# comparison = compare_contour_sets(binned_z_transform_data, binned_z_original_data, overlap_threshold_percent=threshold_comp)
# if comparison:
#     print(f"Comparison Results (Per-slice overlap threshold >= {threshold_comp}%):")
#     for contour_name, (slice_data, volume_overlap) in comparison.items():
#         print(f"  Contour '{contour_name}':")
#         print(f"    Overall Volume Overlap: {volume_overlap:.2f}%")
#         if slice_data:
#             print(f"    Slices meeting threshold:")
#             for slice_idx, overlap, area in slice_data:
#                 print(f"      Slice {slice_idx}: {overlap:.2f}% overlap, {area:.2f} mm^2 area")
#         else:
#             print("    No individual slices met the overlap threshold.")
# else:
#     print(f"No common contours found or no overlaps detected.")
