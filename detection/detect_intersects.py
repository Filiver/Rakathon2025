from shapely.geometry import Polygon
import torch
from collections import defaultdict

dangerous = [("spinalcord", "ptv_low"), ("spinalcord", "ptv_high"),
             ("ptv_low", "parotid_l"), ("ptv_high", "parotid_l"),
             ("ptv_low", "parotid_r"), ("ptv_high", "parotid_r"),
             ("ptv_low", "esophagus"), ("ptv_high", "esophagus"),
             ("ptv_low", "ctv_low"), ("ptv_high", "ctv_low"),
             ("ptv_low", "ctv_high"), ("ptv_high", "ctv_high")]


def detect_intersect(contours_meas_torch_dict):
    """
    Detects intersections between specified contour pairs for each slice and
    calculates overall volume intersection percentages.

    Args:
        contours_meas_torch_dict (dict): Dictionary containing contour data,
                                         expected to have a "binned_z_transform" key
                                         where the value is {contour_name: {slice_idx: tensor}}.
                                         Coordinates are assumed to be in mm.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries. Each dictionary corresponds to a unique slice index
                    found across the relevant contours. Inside each slice dictionary:
                    - Keys are strings representing the dangerous pair (e.g., "contour1/contour2").
                    - Values are dictionaries: {'percentage_cover': float, 'mm2_cover': float}.
                      'percentage_cover' is the intersection area as a percentage of the
                      *larger* contour's area on that slice. 'mm2_cover' is the absolute
                      intersection area in mm^2.
            - dict: A dictionary containing the overall volume intersection percentages for each pair.
                    - Keys are the pair strings (e.g., "contour1/contour2").
                    - Values are dictionaries: {'volume_percentage_cover': float}.
                      This percentage is calculated as (total intersection area across all
                      common slices for the pair) / (total area of the *first* contour
                      in the pair across all common slices) * 100.
                      Returns an empty dictionary if no relevant contours or slices are found.
    """
    results_list = []
    slice_results = defaultdict(dict)
    volume_totals = defaultdict(
        lambda: {'total_intersection': 0.0, 'total_ref_area': 0.0})
    all_unique_slices = set()
    volume_percentage_results = {}  # Initialize volume results dict

    # Check if the main key exists
    if "binned_z_transform" not in contours_meas_torch_dict:
        print("Error: 'binned_z_transform' key not found in input dictionary.")
        return results_list, volume_percentage_results  # Return empty list and dict

    binned_data = contours_meas_torch_dict["binned_z_transform"]

    # --- Pre-computation: Find all unique slices across relevant contours ---
    relevant_contours = set(c for pair in dangerous for c in pair)
    for contour_name in relevant_contours:
        if contour_name in binned_data:
            all_unique_slices.update(binned_data[contour_name].keys())
        else:
            # Print warning only once per missing contour relevant to dangerous pairs
            print(
                f"Warning: Contour '{contour_name}' needed for dangerous pairs analysis is missing.")

    if not all_unique_slices:
        print("Warning: No slices found for any relevant contours.")
        return results_list, volume_percentage_results  # Return empty list and dict

    sorted_slice_indices = sorted(list(all_unique_slices))

    # --- Iterate through slices and then pairs ---
    for slice_idx in sorted_slice_indices:
        current_slice_dict = {}  # Dictionary for the current slice

        for pair in dangerous:
            contour1_name, contour2_name = pair
            pair_str = f"{contour1_name}/{contour2_name}"

            # Check if both contours exist in the main data structure
            if contour1_name not in binned_data or contour2_name not in binned_data:
                # Warning printed during pre-computation if contour is missing entirely
                continue

            contours1_dict = binned_data[contour1_name]
            contours2_dict = binned_data[contour2_name]

            # Check if both contours exist for the *current slice*
            if slice_idx not in contours1_dict or slice_idx not in contours2_dict:
                continue  # This pair doesn't have data on this specific slice

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

                # Accumulate area for the first contour (reference for volume %) if valid and non-zero
                # This happens regardless of intersection on this slice, for slices where both exist
                if poly1.is_valid and poly1.area > 0:
                    volume_totals[pair_str]['total_ref_area'] += poly1.area

                # Ensure polygons are still valid after buffer(0) and have non-zero area for intersection check
                if not poly1.is_valid or not poly2.is_valid or poly1.area == 0 or poly2.area == 0:
                    continue  # Cannot calculate intersection or meaningful overlap

                # Check intersection and calculate overlap percentage
                current_intersection_area_mm2 = 0.0
                percentage_cover = 0.0
                if poly1.intersects(poly2):
                    intersection_poly = poly1.intersection(poly2)
                    # Ensure intersection is a Polygon or MultiPolygon (sum areas if Multi)
                    if isinstance(intersection_poly, Polygon):
                        current_intersection_area_mm2 = intersection_poly.area
                    # Handle MultiPolygon, GeometryCollection etc.
                    elif hasattr(intersection_poly, 'geoms'):
                        current_intersection_area_mm2 = sum(
                            p.area for p in intersection_poly.geoms if isinstance(p, Polygon))

                    if current_intersection_area_mm2 > 1e-9:  # Use tolerance for floating point
                        # Calculate per-slice overlap relative to the area of the LARGER polygon
                        # Areas > 0 checked above
                        larger_area = max(poly1.area, poly2.area)
                        percentage_cover = (
                            current_intersection_area_mm2 / larger_area) * 100

                        # Add slice intersection data
                        current_slice_dict[pair_str] = {
                            'percentage_cover': percentage_cover,
                            'mm2_cover': current_intersection_area_mm2
                        }

                        # Add the intersection area of this slice to the total sum for volume calc
                        volume_totals[pair_str]['total_intersection'] += current_intersection_area_mm2
                    # else: intersection area is effectively zero, don't add to dict

            except Exception as e:
                print(
                    f"Error processing polygons on slice {slice_idx} for pair {pair}: {e}")
                continue

        # Add the dictionary for the current slice to the main slice_results dict
        # It might be empty if no intersections occurred on this slice for any pair
        slice_results[slice_idx] = current_slice_dict

    # --- Format the final output list ---
    for idx in sorted_slice_indices:
        # Append slice dict (could be empty)
        results_list.append(slice_results[idx])

    # --- Calculate final volume percentages ---
    # volume_percentage_results = {} # Moved initialization up
    for pair_str, totals in volume_totals.items():
        total_intersection = totals['total_intersection']
        total_ref_area = totals['total_ref_area']

        volume_percent = 0.0
        if total_ref_area > 0:
            volume_percent = (total_intersection / total_ref_area) * 100

        # Only include pairs that actually had some reference area
        if total_ref_area > 0 or total_intersection > 0:
            volume_percentage_results[pair_str] = {
                'volume_percentage_cover': volume_percent}

    # results_list.append(volume_percentage_results) # Don't append volume results to the list

    return results_list, volume_percentage_results  # Return list and dict separately


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
                                           in the slice dictionary. Defaults to 0.0.

    Returns:
        tuple: A tuple containing:
            - list: A list of dictionaries. Each dictionary corresponds to a unique common slice index
                    found across the common contours. Inside each slice dictionary:
                    - Keys are the common contour names (str, e.g., "parotid_r").
                    - Values are dictionaries: {'percentage_cover': float, 'mm2_cover': float}.
                      'percentage_cover' is the intersection area as a percentage of the
                      *larger* contour's area on that slice. 'mm2_cover' is the absolute
                      intersection area in mm^2. An empty dictionary {} is added for slices
                      with no overlaps meeting the threshold.
            - dict: A dictionary containing the overall volume overlap percentages for each common contour.
                    - Keys are the common contour names (str).
                    - Values are dictionaries: {'volume_percentage_cover': float}.
                      This percentage is calculated as (sum of intersection areas across all common slices) /
                      (sum of original contour areas across all common slices) * 100.
                      Returns an empty dictionary if no common contours or slices are found.
    """
    results_list = []
    slice_results = defaultdict(dict)
    volume_totals = defaultdict(
        lambda: {'total_intersection': 0.0, 'total_original_area': 0.0})
    all_common_slices = set()
    volume_percentage_results = {}

    # Find common contour names (keys) in both dictionaries
    common_contour_names = set(binned_z_transform.keys()) & set(
        binned_z_original.keys())

    if not common_contour_names:
        print("Warning: No common contour names found between the two input dictionaries.")
        return results_list, volume_percentage_results

    # --- Pre-computation: Find all unique common slices across common contours ---
    first_contour = True  # Flag to initialize the set
    for contour_name in common_contour_names:
        # Safely get the dictionaries for the current contour name
        contours_transform_dict = binned_z_transform.get(contour_name, {})
        contours_original_dict = binned_z_original.get(contour_name, {})
        current_common_slices = set(contours_transform_dict.keys()) & set(
            contours_original_dict.keys())

        if first_contour:
            all_common_slices = current_common_slices
            first_contour = False
        else:
            # We need slices common to *all* compared contours for a consistent list length,
            # OR slices common to *any* pair if we want to report any comparison.
            # Let's take the union of common slices for *each* contour pair.
            all_common_slices.update(current_common_slices)

    if not all_common_slices:
        print("Warning: No common slices found for any common contours.")
        return results_list, volume_percentage_results

    sorted_slice_indices = sorted(list(all_common_slices))

    # --- Iterate through slices and then contours ---
    for slice_idx in sorted_slice_indices:
        current_slice_dict = {}  # Dictionary for the current slice

        for contour_name in common_contour_names:
            # Safely get the dictionaries for the current contour name
            contours_transform_dict = binned_z_transform.get(contour_name, {})
            contours_original_dict = binned_z_original.get(contour_name, {})

            # Check if this contour exists on this specific slice in both sets
            if slice_idx not in contours_transform_dict or slice_idx not in contours_original_dict:
                continue  # This contour doesn't have data on this slice in both sets

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

                # Accumulate original area for volume calculation if valid and non-zero
                if poly_original.is_valid and poly_original.area > 0:
                    volume_totals[contour_name]['total_original_area'] += poly_original.area

                # Ensure both polygons are valid and have non-zero area for intersection check
                if not poly_transform.is_valid or not poly_original.is_valid or poly_transform.area == 0 or poly_original.area == 0:
                    continue  # Cannot calculate intersection or meaningful overlap

                # Check intersection and calculate overlap percentage and area
                current_intersection_area_mm2 = 0.0
                percentage_cover = 0.0
                if poly_transform.intersects(poly_original):
                    intersection_poly = poly_transform.intersection(
                        poly_original)
                    # Handle Polygon and MultiPolygon cases for area calculation
                    if isinstance(intersection_poly, Polygon):
                        current_intersection_area_mm2 = intersection_poly.area
                    # Handle MultiPolygon, GeometryCollection etc.
                    elif hasattr(intersection_poly, 'geoms'):
                        current_intersection_area_mm2 = sum(
                            p.area for p in intersection_poly.geoms if isinstance(p, Polygon))

                    # Add current intersection area to total for volume calculation
                    if current_intersection_area_mm2 > 1e-9:
                        volume_totals[contour_name]['total_intersection'] += current_intersection_area_mm2

                        # Calculate per-slice overlap percentage relative to the LARGER polygon
                        # Areas > 0 checked above
                        larger_area = max(poly_transform.area,
                                          poly_original.area)
                        percentage_cover = (
                            current_intersection_area_mm2 / larger_area) * 100

                        # Add to slice dict only if per-slice threshold is met
                        if percentage_cover >= overlap_threshold_percent - 1e-9:
                            current_slice_dict[contour_name] = {
                                'percentage_cover': percentage_cover,
                                'mm2_cover': current_intersection_area_mm2
                            }
                        # Handle threshold = 0 case explicitly: add if any positive intersection
                        elif overlap_threshold_percent == 0.0:  # Already covered by > 1e-9 check
                            current_slice_dict[contour_name] = {
                                'percentage_cover': percentage_cover,
                                'mm2_cover': current_intersection_area_mm2
                            }

            except Exception as e:
                print(
                    f"Error processing polygons on slice {slice_idx} for contour '{contour_name}': {e}")
                continue

        # Add the dictionary for the current slice to the main results list
        # It might be empty if no overlaps occurred on this slice meeting the threshold
        results_list.append(current_slice_dict)

    # --- Calculate final volume percentages ---
    for contour_name, totals in volume_totals.items():
        total_intersection = totals['total_intersection']
        total_original = totals['total_original_area']

        volume_percent = 0.0
        if total_original > 0:
            volume_percent = (total_intersection / total_original) * 100

        # Only include contours that had some original area or intersection area
        if total_original > 0 or total_intersection > 0:
            volume_percentage_results[contour_name] = {
                'volume_percentage_cover': volume_percent}

    return results_list, volume_percentage_results


# Example Usage for detect_intersect (assuming contours_meas_torch_dict is populated):
# slice_data_list, volume_summary = detect_intersect(contours_meas_torch_dict)
# if slice_data_list is not None and volume_summary is not None: # Check if function returned valid data
#     print("Intersection Analysis Results:")
#     print("-" * 30)
#     print("Overall Volume Intersection Percentages (relative to first contour in pair):")
#     if volume_summary:
#         for pair_str, data in volume_summary.items():
#             print(f"  {pair_str}: {data['volume_percentage_cover']:.2f}%")
#     else:
#         print("  No significant volume overlaps detected.")
#     print("-" * 30)
#     print("Per-Slice Intersection Details:")
#     found_slice_intersections = False
#     # To print slice indices correctly, we need them. The function currently doesn't return them.
#     # We could modify the function to return sorted_slice_indices or return slice_results dict directly.
#     # For now, using list index as before:
#     for i, slice_dict in enumerate(slice_data_list):
#         if slice_dict:
#             found_slice_intersections = True
#             # If you need the actual slice index, you'd need to modify the function's return value.
#             # Example: print(f"  Slice Index {sorted_slice_indices[i]}:")
#             print(f"  Slice (List Index {i}):")
#             for pair_str, data in slice_dict.items():
#                 print(f"    {pair_str}: {data['percentage_cover']:.2f}% overlap, {data['mm2_cover']:.2f} mm^2 area")
#     if not found_slice_intersections:
#          print("  No per-slice intersections found.")
# else:
#      print("Could not perform intersection analysis (input data missing or invalid).")


# Example Usage for compare_contour_sets (assuming binned_z_transform and binned_z_original are populated):
# slice_comparison_list, volume_comparison_summary = compare_contour_sets(binned_z_transform, binned_z_original, overlap_threshold_percent=5.0)
# if slice_comparison_list is not None and volume_comparison_summary is not None:
#     print("\nContour Comparison Results:")
#     print("-" * 30)
#     print("Overall Volume Overlap Percentages (Intersection / Original):")
#     if volume_comparison_summary:
#         for contour_name, data in volume_comparison_summary.items():
#             print(f"  {contour_name}: {data['volume_percentage_cover']:.2f}%")
#     else:
#         print("  No significant volume overlaps detected.")
#     print("-" * 30)
#     print(f"Per-Slice Comparison Details (Threshold: {overlap_threshold_percent}%):")
#     found_slice_overlaps = False
#     # Need sorted_slice_indices from function if we want to print actual slice numbers
#     # For now, using list index:
#     for i, slice_dict in enumerate(slice_comparison_list):
#         if slice_dict:
#             found_slice_overlaps = True
#             # print(f"  Slice Index {sorted_slice_indices[i]}:") # If indices were returned
#             print(f"  Slice (List Index {i}):")
#             for contour_name, data in slice_dict.items():
#                 print(f"    {contour_name}: {data['percentage_cover']:.2f}% overlap, {data['mm2_cover']:.2f} mm^2 area")
#     if not found_slice_overlaps:
#          print(f"  No per-slice overlaps found meeting the {overlap_threshold_percent}% threshold.")
# else:
#      print("Could not perform contour comparison (input data missing or invalid).")
