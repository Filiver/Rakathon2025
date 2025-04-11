import os
import pydicom
import numpy as np
from PIL import Image


def save_dicom_image(ds, output_path):
    pixel_array = ds.pixel_array.astype(np.float32)
    if 'RescaleSlope' in ds and 'RescaleIntercept' in ds:
        pixel_array = pixel_array * \
            float(ds.RescaleSlope) + float(ds.RescaleIntercept)

    pixel_array -= np.min(pixel_array)
    pixel_array /= np.max(pixel_array)
    pixel_array *= 255
    image = Image.fromarray(pixel_array.astype(np.uint8))
    image.save(output_path)


def extract_metadata(ds):
    def get(tag):
        return ds.get(tag, '')

    return {
        'AcquisitionDate': str(get((0x0008, 0x0022))),
        'AcquisitionTime': str(get((0x0008, 0x0032))),
        'FrameOfReferenceUID': str(get((0x0020, 0x0052))),
        'SeriesNumber': str(get((0x0020, 0x0011))),
        'AcquisitionNumber': str(get((0x0020, 0x0012))),
        'InstanceNumber': str(get((0x0020, 0x0013))),
        'ImagePositionPatient': str(get((0x0020, 0x0032))),
        'ImageOrientationPatient': str(get((0x0020, 0x0037))),
    }


def process_directory(root_dir):
    output_dir = os.path.join(root_dir, "loaded")
    os.makedirs(output_dir, exist_ok=True)

    metadata_lines = []

    for dirpath, _, filenames in os.walk(root_dir):
        for file in filenames:
            if not file.lower().endswith(".dcm"):
                continue

            filepath = os.path.join(dirpath, file)
            try:
                ds = pydicom.dcmread(filepath)

                # Create unique filename in output_dir
                filename_wo_ext = os.path.splitext(file)[0]
                out_img_path = os.path.join(
                    output_dir, f"{filename_wo_ext}.png")
                save_dicom_image(ds, out_img_path)

                meta = extract_metadata(ds)
                metadata_lines.append(
                    f"{file}:\n"
                    f"  Acquisition Date: {meta['AcquisitionDate']}\n"
                    f"  Acquisition Time: {meta['AcquisitionTime']}\n"
                    f"  Frame of Reference UID: {meta['FrameOfReferenceUID']}\n"
                    f"  Series Number: {meta['SeriesNumber']}\n"
                    f"  Acquisition Number: {meta['AcquisitionNumber']}\n"
                    f"  Instance Number: {meta['InstanceNumber']}\n"
                    f"  Image Position (Patient): {meta['ImagePositionPatient']}\n"
                    f"  Image Orientation (Patient): {meta['ImageOrientationPatient']}\n"
                )

            except Exception as e:
                print(f"Error processing {filepath}: {e}")

    metadata_path = os.path.join(output_dir, "SAMPLE_001_metadata.txt")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata_lines))


if __name__ == "__main__":
    process_directory("radioprotect/data/SAMPLE_001")
