import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import SimpleITK as sitk
import scipy.ndimage as ndi
from organize_new import process_sample

ROI_intrested = [
    re.compile(r"^gtv.*"),
    re.compile(r"^ptv.*"),
    re.compile(r"^ctv.*"),
    re.compile(r"body"),
    re.compile(r"spinalcord"),
    re.compile(r"parotid*"),
    re.compile(r"submandibular*"),
    re.compile(r"esophagus*"),
    re.compile(r"glnd_submand*"),
]


def parse_report_file(report_path):
    import re
    rs_to_cts = {}
    current_rs = None

    with open(report_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        line = line.strip()
        rs_match = re.search(r'RS File:\s*(.+\.dcm)', line)
        if rs_match:
            current_rs = rs_match.group(1)
            rs_to_cts[current_rs] = []
        elif "Files (Count:" in line and current_rs:
            ct_files = re.findall(r'(CT\.[^,\s]+\.dcm)', line)
            rs_to_cts[current_rs].extend(ct_files)

    print(f"Found {len(rs_to_cts)} RS files with CT references.")
    return rs_to_cts


def load_rtstruct_contours(rtstruct_path):
    rs = pydicom.dcmread(rtstruct_path)

    roi_names = {}
    for item in rs.StructureSetROISequence:
        if hasattr(item, "ROINumber") and hasattr(item, "ROIName"):
            roi_names[item.ROINumber] = item.ROIName

    contour_map = {}

    for roi in rs.ROIContourSequence:
        roi_number = roi.ReferencedROINumber
        roi_name = roi_names.get(roi_number, f"ROI_{roi_number}").lower()

        if not any(pattern.match(roi_name) for pattern in ROI_intrested):
            continue

        if not hasattr(roi, "ContourSequence"):
            continue  # Skip if there's no ContourSequence

        for contour in roi.ContourSequence:
            if not hasattr(contour, "ContourImageSequence"):
                continue

            sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            coords = np.array(contour.ContourData).reshape(-1, 3)

            if roi_name not in contour_map:
                contour_map[sop_uid] = []
                #contour_map[roi_name] = coords

            contour_map[sop_uid].append((roi_name, coords))
            #contour_map[roi_name] = np.vstack((contour_map[roi_name], coords))

    return contour_map


def find_ct_by_sop(directory):
    """Scan for all CT files and index them by SOPInstanceUID."""
    sop_to_ct = {}
    for root, _, files in os.walk(directory):
        for file in files:
            path = os.path.join(root, file)
            try:
                ds = pydicom.dcmread(path)
                if ds.Modality == "CT":
                    sop_to_ct[ds.SOPInstanceUID] = ds
            except:
                continue
    return sop_to_ct


def load_ordered_ct_series_from_directory(dicom_dir):
    """
    Loads and returns a list of ordered pydicom Datasets using SimpleITK's SeriesReader.
    """
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)

    if not dicom_files:
        raise RuntimeError(f"No readable DICOM series found in {dicom_dir}.")

    # Return the same list as used by reader, but loaded via pydicom for metadata access
    ordered_datasets = []
    for file in dicom_files:
        try:
            ds = pydicom.dcmread(file)
            ordered_datasets.append(ds)
        except Exception as e:
            print(f"⚠️ Could not read DICOM file {file}: {e}")
    return ordered_datasets


def generate_volume_point_cloud(ct_datasets, rgb=True, mask=True):
    import matplotlib.cm as cm
    points_list = []
    valid_cts = []

    for ds in ct_datasets:
        try:
            _ = ds.ImagePositionPatient[2]
            _ = ds.PixelSpacing
            valid_cts.append(ds)
        except AttributeError:
            print(
                f"⚠️ Skipping slice without spatial metadata: {getattr(ds, 'SOPInstanceUID', 'UNKNOWN')}")

    if not valid_cts:
        raise ValueError("🚫 No valid CT slices found with spatial metadata!")

    valid_cts = sorted(valid_cts, key=lambda ds: ds.ImagePositionPatient[2])

    for ds in valid_cts:
        img = ds.pixel_array.astype(np.int16)

        # Step 1: Threshold image to exclude air and bed
        voxel_mask = (img < -500) | (img > 500)
        if not np.any(voxel_mask):
            continue

        # Step 2: If mask is enabled, keep only largest connected component
        if mask:
            labeled, num = ndi.label(voxel_mask)
            if num == 0:
                continue
            sizes = ndi.sum(voxel_mask, labeled, range(1, num + 1))
            voxel_mask = (labeled == (np.argmax(sizes) + 1))
        else:
            voxel_mask = np.ones_like(voxel_mask, dtype=bool)

        rows, cols = img.shape
        spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
        origin = np.array(ds.ImagePositionPatient)

        x = np.arange(cols)
        y = np.arange(rows)
        xx, yy = np.meshgrid(x, y)
        # print(f"xx: {xx.shape}, yy: {yy.shape}")

        coords_x = origin[0] + xx * spacing[0]
        coords_y = origin[1] + yy * spacing[1]  # flipped
        coords_z = np.full_like(coords_x, origin[2])

        coords = np.stack((coords_x, coords_y, coords_z), axis=-1)
        masked_coords = coords[voxel_mask]

        if rgb:
            img_normalized = np.clip(
                (img.astype(np.float32) + 300) / 1500, 0, 1)
            colored_img = cm.get_cmap("viridis")(
                img_normalized)[:, :, :3]  # RGB
            masked_rgb = (colored_img[voxel_mask] * 255).astype(np.uint8)
            cloud = np.hstack((masked_coords, masked_rgb))
        else:
            cloud = masked_coords

        points_list.append(cloud)
    print(f"Generated {len(points_list)} slices of point cloud data.")

    return np.vstack(points_list)


def generate_point_cloud(ds, rgb=True):
    """
    Converts a single DICOM CT slice into a point cloud with real-world coordinates.
    Optionally includes grayscale value as RGB.
    """
    img = ds.pixel_array.astype(np.int16)
    spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
    origin = np.array(ds.ImagePositionPatient)

    # Get image dimensions
    rows, cols = img.shape

    # Create pixel grid
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)

    # Convert to real-world coordinates (x, y, z in mm)
    coords_x = origin[0] + xx * spacing[0]
    coords_y = origin[1] + yy * spacing[1]
    coords_z = np.full_like(coords_x, origin[2])  # constant slice Z

    # Flatten everything
    points = np.stack((coords_x, coords_y, coords_z), axis=-1).reshape(-1, 3)

    if rgb:
        # Normalize grayscale value for RGB
        norm = np.clip((img + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
        values = np.stack([norm, norm, norm], axis=-1).reshape(-1, 3)
        return np.hstack((points, values))  # shape: [N, 6]
    else:
        return points  # shape: [N, 3]


def generate_point_cloud(ds, rgb=True):
    """
    Converts a single DICOM CT slice into a point cloud with real-world coordinates.
    Optionally includes grayscale value as RGB.
    """
    img = ds.pixel_array.astype(np.int16)
    spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
    origin = np.array(ds.ImagePositionPatient)

    # Get image dimensions
    rows, cols = img.shape

    # Create pixel grid
    x = np.arange(cols)
    y = np.arange(rows)
    xx, yy = np.meshgrid(x, y)

    # Convert to real-world coordinates (x, y, z in mm)
    coords_x = origin[0] + xx * spacing[0]
    coords_y = origin[1] + yy * spacing[1]
    coords_z = np.full_like(coords_x, origin[2])  # constant slice Z

    # Flatten everything
    points = np.stack((coords_x, coords_y, coords_z), axis=-1).reshape(-1, 3)

    if rgb:
        # Normalize grayscale value for RGB
        norm = np.clip((img + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
        values = np.stack([norm, norm, norm], axis=-1).reshape(-1, 3)
        return np.hstack((points, values))  # shape: [N, 6]
    else:
        return points  # shape: [N, 3]


def process_all_rs(directory, output_dir="plots"):
    img_dir = os.path.join(output_dir, "imgs")
    txt_dir = os.path.join(output_dir, "txt")

    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(txt_dir, exist_ok=True)

    rtstruct_files = [os.path.join(dp, f) for dp, dn, filenames in os.walk(directory)
                      for f in filenames if f.lower().endswith(".dcm")]

    sop_to_ct = find_ct_by_sop(directory)
    print(f"🧠 Indexed {len(sop_to_ct)} CT files.")

    for rs_path in rtstruct_files:
        try:
            print(f"\n📄 Processing RS file: {rs_path}")
            contours = load_rtstruct_contours(rs_path)

            for sop_uid, contour_list in contours.items():
                if sop_uid not in sop_to_ct:
                    print(f"⚠️ SOP UID {sop_uid} not found in CT scans.")
                    continue

                ct_ds = sop_to_ct[sop_uid]
                plot_ct_with_contours(ct_ds, contour_list, img_dir, txt_dir)

        except Exception as e:
            print(f"💥 Error processing {rs_path}: {e}")

    # Generate and save point cloud
    ct_datasets = load_ordered_ct_series_from_directory(directory)
    volume_cloud = generate_volume_point_cloud(ct_datasets)
    cloud_path = os.path.join(output_dir, "full_volume_pointcloud.npy")
    np.save(cloud_path, volume_cloud)
    print(f"💾 Saved full volume point cloud: {cloud_path}")
    visualize_point_cloud(volume_cloud, num_points=200000)  # adjustable


def visualize_point_cloud(cloud, num_points=10000):
    """
    Visualize a sample of the point cloud using matplotlib.
    """
    if cloud.shape[0] > num_points:
        indices = np.random.choice(
            cloud.shape[0], size=num_points, replace=False)
        cloud = cloud[indices]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    x, y, z = cloud[:, 0], cloud[:, 1], cloud[:, 2]
    if cloud.shape[1] == 6:
        colors = cloud[:, 3:] / 255.0
        ax.scatter(x, y, z, c=colors, s=1)
    else:
        ax.scatter(x, y, z, color='gray', s=1)

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Point Cloud Preview")
    plt.tight_layout()
    plt.show()


def load_ordered_ct_series_from_directory(dicom_dir):
    reader = sitk.ImageSeriesReader()
    dicom_files = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not dicom_files:
        raise RuntimeError(f"No readable DICOM series found in {dicom_dir}.")
    ordered_datasets = [pydicom.dcmread(file) for file in dicom_files]
    return ordered_datasets


def plot_ct_with_contours(ds, contours, img_dir, txt_dir):
    img = ds.pixel_array.astype(np.int16)
    img = np.clip((img + 1000) / 2000 * 255, 0, 255).astype(np.uint8)
    spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
    origin = np.array(ds.ImagePositionPatient)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")
    txt_lines = []

    for roi_name, coords in contours:
        # Plotting
        ij = ((coords[:, :2] - origin[:2]) / spacing[:2]).astype(int)
        ax.plot(ij[:, 0], ij[:, 1], label=roi_name, linewidth=1)

        # Saving coordinates as real-world 3D coordinates (x,y,z)
        points_str = " ".join(f"{x:.2f},{y:.2f},{z:.2f}" for x, y, z in coords)
        txt_lines.append(f"{roi_name}: {points_str}")

    if contours:
        ax.legend(fontsize="x-small", loc="lower right")

    ax.axis("off")
    plt.title(f"SOP UID: {ds.SOPInstanceUID}")
    png_path = os.path.join(img_dir, f"{ds.SOPInstanceUID}.png")
    txt_path = os.path.join(txt_dir, f"{ds.SOPInstanceUID}.txt")
    plt.savefig(png_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))


def process_each_rs_separately(base_dir, report_path, output_dir="pointclouds_by_rs", point_cloud=False):
    output_dir = os.path.join(output_dir, os.path.basename(base_dir))
    print(f"Output directory: {output_dir}")
    rs_mapping = process_sample(base_dir)
    values = rs_mapping.values()
    cts = []
    rs = []
    for v in values:
        if "CT" not in v:
            continue
        cts.append(v["CT"])
        rs.append(v["RTSTRUCT"])
    print(f"Found {len(rs_mapping)} RS files with CT references.")
    print(f"Found {len(cts)} CT files with RS references.")

    for rs_files, ct_files in zip(rs, cts):
        ct_paths = []
        rs_paths = []
        rs_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file in rs_files:
                    # rs_path = os.path.join(root, file)
                    rs_paths.append(os.path.join(root, file))
                elif file in ct_files:
                    ct_paths.append(os.path.join(root, file))
        print(f"Found {len(rs_paths)} RS files and {len(ct_paths)} CT files.")

        if rs_paths and ct_paths:
            print(
                f"Processing RS: {len(rs_paths)} files and {len(ct_paths)} CT files.")
            rs_files_combined = "__".join(
                [os.path.basename(path) for path in rs_paths])
            rs_files_combined = rs_files_combined.replace(
                ".dcm", "").replace(" ", "_")
            rs_output_dir = os.path.join(
                output_dir, rs_files_combined.replace(".dcm", "").replace(" ", "_"))
            os.makedirs(rs_output_dir, exist_ok=True)
            img_dir = os.path.join(rs_output_dir, "imgs")
            txt_dir = os.path.join(rs_output_dir, "txt")

            os.makedirs(img_dir, exist_ok=True)
            os.makedirs(txt_dir, exist_ok=True)

            ct_datasets = [pydicom.dcmread(path) for path in sorted(
                ct_paths, key=lambda x: pydicom.dcmread(x).ImagePositionPatient[2])]

            contour_map = {}
            for rs_path in rs_paths:
                new_contours = load_rtstruct_contours(rs_path)
                for sop_uid, contours in new_contours.items():
                    if sop_uid not in contour_map:
                        contour_map[sop_uid] = []
                    contour_map[sop_uid].extend(contours)

                # contour_map = load_rtstruct_contours(rs_path)
            for ds in ct_datasets:
                sop_uid = ds.SOPInstanceUID
                contours = contour_map.get(sop_uid, [])
                plot_ct_with_contours(
                    ds, contours, img_dir, txt_dir)

            if point_cloud:
                point_cloud = generate_volume_point_cloud(
                    ct_datasets, mask=False)
                visualize_point_cloud(point_cloud, num_points=10000)

                out_path = os.path.join(
                    rs_output_dir, f"{rs_files_combined}_full_volume.npy")
                np.save(out_path, point_cloud)
                point_cloud = generate_volume_point_cloud(
                    ct_datasets, mask=True)
                out_path = os.path.join(
                    rs_output_dir, f"{rs_files_combined}_masked_volume.npy")
                np.save(out_path, point_cloud)

                print(f"Saved point cloud: {out_path}")
        else:
            print(f"Missing RS or CT files for RS: {rs_files}")

def process_rt_ct_pairs(base_dir, cts, rs):
    for rs_files, ct_files in zip(rs, cts):
        ct_paths = []
        rs_paths = []
        rs_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file in rs_files:
                    # rs_path = os.path.join(root, file)
                    rs_paths.append(os.path.join(base_dir, "RT", file))
                elif file in ct_files:
                    ct_paths.append(os.path.join(base_dir, "CT", file))
        print(f"Found {len(rs_paths)} RS files and {len(ct_paths)} CT files.")

        if rs_paths and ct_paths:
            print(
                f"Processing RS: {len(rs_paths)} files and {len(ct_paths)} CT files.")
            
            ct_datasets = [pydicom.dcmread(path) for path in sorted(
                ct_paths, key=lambda x: pydicom.dcmread(x).ImagePositionPatient[2])]

            contour_map = {}
            for rs_path in rs_paths:
                new_contours = load_rtstruct_contours(rs_path)
                for sop_uid, contours in new_contours.items():
                    for roi_name, coords in contours:
                        if roi_name not in contour_map:
                            contour_map[roi_name] = coords
                        else:
                            contour_map[roi_name] = np.vstack((contour_map[roi_name], coords))
            return contour_map
        else:
            print(f"Missing RS or CT files for RS: {rs_files}")


if __name__ == "__main__":
    data_dir = "data/radioprotect/Rakathon Data/SAMPLE_004"
    report_txt = "data/radioprotect/Rakathon Data Organized/SAMPLE_004_report.txt"
    process_each_rs_separately(data_dir, report_txt)
