import pydicom
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Polygon, Point


def get_contour_points(rs_ds):
    # Extract structure set ROI contours
    contours = []
    for roi_contour in rs_ds.ROIContourSequence:
        if hasattr(roi_contour, "ContourSequence"):
            for contour in roi_contour.ContourSequence:
                points = np.array(contour.ContourData).reshape(-1, 3)
                contours.append(points)
    return contours


def create_mask(contours, ct_slice_ds):
    # Extract image parameters
    image_position = np.array(ct_slice_ds.ImagePositionPatient)  # (x, y, z)
    # (row_spacing, col_spacing)
    pixel_spacing = np.array(ct_slice_ds.PixelSpacing)
    rows, cols = ct_slice_ds.Rows, ct_slice_ds.Columns

    mask = np.zeros((rows, cols), dtype=np.uint8)

    for contour in contours:
        print(f"Contour: {contour}")
        if not np.isclose(contour[0, 2], image_position[2]):
            continue  # Different slice

        # Convert contour coords to pixel indices
        pixel_coords = []
        for x, y, _ in contour:
            col = int(round((x - image_position[0]) / pixel_spacing[0]))
            row = int(round((y - image_position[1]) / pixel_spacing[1]))
            pixel_coords.append((col, row))

        # Rasterize polygon (shapely+point-inside)
        poly = Polygon(pixel_coords)
        for row in range(rows):
            for col in range(cols):
                if poly.contains(Point(col, row)):
                    mask[row, col] = 1

    return mask


# Example usage
rs_path = "radioprotect/data/SAMPLE_001/RS.1.2.246.352.221.4639420678005016246395232927368615560.dcm"
ct_path = "radioprotect/data/SAMPLE_001/CT.1.2.246.352.221.50911002929711127858269804207128222.dcm"

rs_ds = pydicom.dcmread(rs_path)
ct_ds = pydicom.dcmread(ct_path)

contours = get_contour_points(rs_ds)
mask = create_mask(contours, ct_ds)

plt.imshow(ct_ds.pixel_array, cmap="gray")
plt.imshow(mask, cmap="Reds", alpha=0.4)
plt.title("Tumor Mask Overlay")
plt.axis("off")
plt.show()
