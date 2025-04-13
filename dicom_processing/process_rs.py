import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import regex as re
import SimpleITK as sitk
import scipy.ndimage as ndi
import matplotlib.cm as cm

ROI_intrested = [
    re.compile(r"^gtv.*"),
    re.compile(r"^ptv.*"),
    re.compile(r"^ctv.*"),
    re.compile(r"body"),
    re.compile(r"spinalcord"),
    re.compile(r"parotid*"),
    re.compile(r"submandibular*"),
    re.compile(r"ezophagus*"),
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


def visualize_point_cloud(cloud, num_points=10000):
    """
    Visualize a sample of the point cloud using matplotlib.
    """
    print(f"Visualizing {cloud.shape[0]} points in the point cloud.")
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


def load_ordered_ct_series(ct_paths):
    datasets = []
    for path in ct_paths:
        ds = pydicom.dcmread(path)
        datasets.append(ds)
    datasets.sort(key=lambda x: x.ImagePositionPatient[2])
    return datasets


def generate_volume_point_cloud(ct_datasets, rgb=True, mask=False):
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

    valid_cts.sort(key=lambda ds: ds.ImagePositionPatient[2])

    for ds in valid_cts:
        img = ds.pixel_array.astype(np.int16)

        voxel_mask = (img < -500) | (img > 500)
        if not np.any(voxel_mask):
            continue

        if mask:
            labeled, num = ndi.label(voxel_mask)
            if num == 0:
                continue
            sizes = ndi.sum(voxel_mask, labeled, range(1, num + 1))
            voxel_mask = (labeled == (np.argmax(sizes) + 1))
        else:
            voxel_mask = np.ones_like(img, dtype=bool)

        rows, cols = img.shape
        spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
        origin = np.array(ds.ImagePositionPatient)

        xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
        print(f"Processing slice with shape: {img.shape} and origin: {origin}")

        coords = np.stack((origin[0] + xx * spacing[0],
                           origin[1] + yy * spacing[1],
                           np.full_like(xx, origin[2])), axis=-1)

        masked_coords = coords[voxel_mask]

        if rgb:
            img_normalized = np.clip(
                (img.astype(np.float32) + 300) / 1500, 0, 1)
            colored_img = cm.get_cmap("viridis")(img_normalized)[:, :, :3]
            masked_rgb = (colored_img[voxel_mask] * 255).astype(np.uint8)
            cloud = np.hstack((masked_coords, masked_rgb))
        else:
            cloud = masked_coords

        points_list.append(cloud)

    print(f"Generated {len(points_list)} slices of point cloud data.")

    return np.vstack(points_list)


def process_each_rs_separately(base_dir, report_path, output_dir="pointclouds_by_rs"):
    rs_mapping = parse_report_file(report_path)
    os.makedirs(output_dir, exist_ok=True)

    for rs_file, ct_files in rs_mapping.items():
        ct_paths = []
        rs_path = None
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file == rs_file:
                    rs_path = os.path.join(root, file)
                elif file in ct_files:
                    ct_paths.append(os.path.join(root, file))

        if rs_path and ct_paths:
            print(f"Processing RS: {rs_file} with {len(ct_paths)} CT files")
            ct_datasets = load_ordered_ct_series(ct_paths)
            point_cloud = generate_volume_point_cloud(ct_datasets)
            visualize_point_cloud(point_cloud)
            out_path = os.path.join(output_dir, f"{rs_file}_pointcloud.npy")
            np.save(out_path, point_cloud)
            print(f"Saved point cloud: {out_path}")
        else:
            print(f"Missing RS or CT files for RS: {rs_file}")


if __name__ == "__main__":
    data_dir = "data/radioprotect/Rakathon Data/SAMPLE_004"
    report_txt = "data/radioprotect/Rakathon Data Organized/SAMPLE_004_report.txt"
    process_each_rs_separately(data_dir, report_txt)
