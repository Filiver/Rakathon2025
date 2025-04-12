# ... existing code in the cell ...
import os
import numpy as np
import matplotlib.pyplot as plt
import pydicom

dcm_file_path = "data/radioprotect/Rakathon Data/SAMPLE_001/RS.1.2.246.352.221.53086809173815688567595866456863246500.dcm"

# Load the DICOM file
try:
    dcm = pydicom.dcmread(dcm_file_path)
except Exception as e:
    print(f"Error loading DICOM file: {e}")
    exit()

# --- Create output directory ---
output_dir = "contours"
os.makedirs(output_dir, exist_ok=True)
print(f"Saving contour plots to: {os.path.abspath(output_dir)}")

# --- Iterate through all contours ---
contours_plotted = 0

# Check if ROIContourSequence exists
if hasattr(dcm, "ROIContourSequence"):
    for i, roi_contour in enumerate(dcm.ROIContourSequence):
        # Check if ContourSequence exists within the ROI
        if hasattr(roi_contour, "ContourSequence"):
            for j, contour_sequence_item in enumerate(roi_contour.ContourSequence):
                # Check if ContourGeometricType is CLOSED_PLANAR
                if hasattr(contour_sequence_item, "ContourGeometricType") and contour_sequence_item.ContourGeometricType == "CLOSED_PLANAR":
                    # Check if ContourData and ContourImageSequence exist
                    if (
                        hasattr(contour_sequence_item, "ContourData")
                        and hasattr(contour_sequence_item, "ContourImageSequence")
                        and len(contour_sequence_item.ContourImageSequence) > 0
                        and hasattr(contour_sequence_item.ContourImageSequence[0], "ReferencedSOPInstanceUID")
                    ):
                        contour_data = contour_sequence_item.ContourData
                        ref_sop_uid = contour_sequence_item.ContourImageSequence[
                            0].ReferencedSOPInstanceUID

                        print(
                            f"\nProcessing ROI {i+1}, Contour {j+1} referencing SOP UID: {ref_sop_uid}")

                        # Reshape contour data into (N, 3) array
                        points = np.array(contour_data).reshape((-1, 3))

                        # --- Construct the path to the referenced CT file ---
                        rtstruct_dir = os.path.dirname(dcm_file_path)
                        ct_filename = f"CT.{ref_sop_uid}.dcm"
                        ct_file_path = os.path.join(rtstruct_dir, ct_filename)

                        print(f"Attempting to load CT file: {ct_file_path}")

                        # Load the referenced CT image
                        try:
                            ct_dcm = pydicom.dcmread(ct_file_path)
                            if hasattr(ct_dcm, "pixel_array"):
                                ct_image = ct_dcm.pixel_array

                                # --- Plotting ---
                                plt.figure(figsize=(8, 8))
                                plt.imshow(ct_image, cmap="gray")
                                # Plot contour points (x, y). Add the first point to the end to close the loop.
                                plt.plot(np.append(points[:, 0], points[0, 0]), np.append(
                                    points[:, 1], points[0, 1]), "r-", label=f"ROI {i+1} Contour {j+1}")  # Red line
                                plt.title(
                                    f"CT Slice with RTSTRUCT Contour\n(SOP UID: {ref_sop_uid})")
                                plt.xlabel("X coordinate")
                                plt.ylabel("Y coordinate")
                                plt.legend()
                                # Ensure aspect ratio is correct
                                plt.axis('equal')

                                # --- Save the plot ---
                                output_filename = f"contour_roi_{i+1}_contour_{j+1}_ct_{ref_sop_uid}.png"
                                output_path = os.path.join(
                                    output_dir, output_filename)
                                plt.savefig(output_path)
                                plt.close()  # Close the figure to free memory
                                print(f"Saved plot to: {output_path}")
                                contours_plotted += 1

                            else:
                                print(
                                    f"Error: CT file {ct_filename} does not contain pixel data.")

                        except FileNotFoundError:
                            print(
                                f"Error: Referenced CT file not found at {ct_file_path}")
                        except Exception as e:
                            print(
                                f"Error loading or plotting CT file {ct_filename}: {e}")
                        # --- End Plotting/Saving Block ---
else:
    print("RTSTRUCT file does not contain ROIContourSequence.")

print(f"\nFinished processing. Plotted and saved {contours_plotted} contours.")
# ... rest of the cell (if any) ...
