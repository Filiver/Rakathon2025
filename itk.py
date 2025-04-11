import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np


def load_rtstruct_contours(rtstruct_path):
    rs = pydicom.dcmread(rtstruct_path)

    # Map SOPInstanceUID to a list of contours
    contour_map = {}

    for roi in rs.ROIContourSequence:
        roi_name = getattr(roi, "ROIDisplayName", "Unnamed ROI")

        for contour in roi.ContourSequence:
            if not hasattr(contour, "ContourImageSequence"):
                continue

            sop_uid = contour.ContourImageSequence[0].ReferencedSOPInstanceUID
            coords = np.array(contour.ContourData).reshape(-1, 3)

            if sop_uid not in contour_map:
                contour_map[sop_uid] = []

            contour_map[sop_uid].append((roi_name, coords))

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


def plot_ct_with_contours(ds, contours, output_path):
    img = ds.pixel_array.astype(np.int16)
    img = np.clip((img + 1000) / 2000 * 255, 0, 255).astype(np.uint8)

    spacing = list(ds.PixelSpacing) + [ds.SliceThickness]
    origin = np.array(ds.ImagePositionPatient)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(img, cmap="gray")

    txt_lines = []

    for roi_name, coords in contours:
        ij = ((coords[:, :2] - origin[:2]) / spacing[:2]).astype(int)
        ax.plot(ij[:, 0], ij[:, 1], label=roi_name, linewidth=1)

        points_str = " ".join(f"{x},{y}" for x, y in ij)
        txt_lines.append(f"{roi_name}: {points_str}")

    if contours:
        ax.legend(fontsize="x-small", loc="lower right")

    ax.axis("off")
    plt.title(f"SOP UID: {ds.SOPInstanceUID}")

    # Save image
    plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    # Save text file with same basename
    txt_path = output_path.replace(".png", ".txt")
    with open(txt_path, "w") as f:
        f.write("\n".join(txt_lines))

    print(f"Saved: {output_path}")
    print(f"Saved: {txt_path}")


def process_all_rs(directory, output_dir="plots"):
    os.makedirs(output_dir, exist_ok=True)

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
                out_path = os.path.join(output_dir, f"{sop_uid}.png")
                plot_ct_with_contours(ct_ds, contour_list, out_path)

        except Exception as e:
            print(f"💥 Error processing {rs_path}: {e}")


if __name__ == "__main__":
    data_path = "data/radioprotect/Rakathon Data/SAMPLE_001"  # 👑 ← Update this!
    process_all_rs(data_path)
