import os
import SimpleITK as sitk
import shutil
from datetime import datetime
from collections import defaultdict
import sys
from pathlib import Path
import numpy as np  # Added for axial check
import pydicom  # Import pydicom
from pydicom.errors import InvalidDicomError  # Import specific error

HERE = Path(__file__).parent.parent

# --- Configuration ---
# Set the main directory containing your sample folders (e.g., SAMPLE_001, SAMPLE_002)
MAIN_SAMPLES_DIRECTORY = HERE / "data/radioprotect/Rakathon Data"
# Set the base directory where the organized structure will be created
ORGANIZED_OUTPUT_DIRECTORY = HERE / "data/radioprotect/Organized_CT_Data_Axial"
# Set to True to copy files, False to print actions without copying (for testing)
PERFORM_COPY = True
# ------------------


def parse_dicom_date_time(date_str, time_str):
    """Safely parse DICOM date and time strings."""
    if not date_str:
        return None, None
    try:
        dt_obj = datetime.strptime(date_str, "%Y%m%d")
        if time_str:
            # Handle fractional seconds if present
            time_str = time_str.split(".")[0]
            if len(time_str) == 6:  # HHMMSS
                dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M%S")
            elif len(time_str) == 4:  # HHMM
                dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H%M")
            elif len(time_str) == 2:  # HH
                dt_obj = datetime.strptime(f"{date_str}{time_str}", "%Y%m%d%H")

        return dt_obj.date(), dt_obj.time()
    except ValueError:
        # Fallback if only date is valid
        try:
            return datetime.strptime(date_str, "%Y%m%d").date(), None
        except ValueError:
            return None, None


def is_axial_scan(image_orientation_patient):
    """Checks if the image orientation corresponds to an axial scan."""
    if not image_orientation_patient or len(image_orientation_patient) != 6:
        return False
    try:
        row_vector = np.array(image_orientation_patient[0:3])
        col_vector = np.array(image_orientation_patient[3:6])
        # Calculate the cross product
        cross_product = np.cross(row_vector, col_vector)
        # Check if the cross product is predominantly along the Z-axis
        return abs(cross_product[2]) > 0.95  # Allow for some tolerance
    except Exception:
        return False


def get_dicom_info(filepath):
    """
    Extracts relevant information from a DICOM file. Uses pydicom to check modality
    and read RTSTRUCT info, uses SimpleITK for CT image info.

    Returns:
        tuple: (file_type, date_obj, time_obj, frame_uid, instance_number, sop_instance_uid, is_axial, filepath) or None
        file_type: 'CT', 'RTSTRUCT', or None
        frame_uid: FrameOfReferenceUID for CT, ReferencedFrameOfReferenceUID for RTSTRUCT
        is_axial: Boolean (only relevant for CT)
    """
    try:
        # Step 1: Try reading with pydicom first to get Modality safely
        ds = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)  # force=True might help with slightly non-conformant files

        modality = getattr(ds, "Modality", "").upper()
        sop_instance_uid = getattr(ds, "SOPInstanceUID", None)

        # --- Check if Modality is RTSTRUCT ---
        if modality == "RTSTRUCT":
            referenced_frame_uid = None
            # Extract ReferencedFrameOfReferenceUID using pydicom
            if hasattr(ds, "ReferencedFrameOfReferenceSequence"):
                ref_seq = ds.ReferencedFrameOfReferenceSequence
                if ref_seq and len(ref_seq) > 0:
                    # Access the FrameOfReferenceUID within the first item of the sequence
                    if hasattr(ref_seq[0], "FrameOfReferenceUID"):
                        referenced_frame_uid = ref_seq[0].FrameOfReferenceUID

            # Fallback: Check direct tag (less common for RTSTRUCT)
            if not referenced_frame_uid and hasattr(ds, "FrameOfReferenceUID"):
                referenced_frame_uid = ds.FrameOfReferenceUID

            if referenced_frame_uid:
                return ("RTSTRUCT", None, None, referenced_frame_uid, None, sop_instance_uid, None, filepath)
            else:
                # print(f"Warning: File {os.path.basename(filepath)} has Modality RTSTRUCT but could not find ReferencedFrameOfReferenceUID.")
                return None

        # --- If not RTSTRUCT, check if it's CT ---
        elif modality == "CT":
            # Step 2: Now use SimpleITK for CT image-specific info
            try:
                reader = sitk.ImageFileReader()
                reader.SetFileName(filepath)
                reader.ReadImageInformation()
                metadata_keys = reader.GetMetaDataKeys()

                # Verify SOP Class UID if desired (optional, as Modality is CT)
                # sop_class_uid = reader.GetMetaData("0008|0016").strip() if "0008|0016" in metadata_keys else None
                # if sop_class_uid != SOP_CLASS_UID_CT: return None # Stricter check

                frame_uid = reader.GetMetaData("0020|0052").strip() if "0020|0052" in metadata_keys else None
                if not frame_uid:
                    return None

                instance_num_str = reader.GetMetaData("0020|0013").strip() if "0020|0013" in metadata_keys else None
                instance_number = int(instance_num_str) if instance_num_str and instance_num_str.isdigit() else None

                orientation_str = reader.GetMetaData("0020|0037").strip() if "0020|0037" in metadata_keys else None
                orientation = [float(x) for x in orientation_str.split("\\")] if orientation_str and "\\" in orientation_str else None
                is_axial = is_axial_scan(orientation)

                # Extract Date/Time using SimpleITK metadata
                best_date, best_time = None, None
                priority_tags = [
                    ("0008|0022", "0008|0032"),
                    ("0008|0021", "0008|0031"),
                    ("0008|0020", "0008|0030"),
                    ("0008|0023", "0008|0033"),
                ]
                for date_tag, time_tag in priority_tags:
                    date_str = reader.GetMetaData(date_tag).strip() if date_tag in metadata_keys else None
                    time_str = reader.GetMetaData(time_tag).strip() if time_tag in metadata_keys else None
                    d, t = parse_dicom_date_time(date_str, time_str)
                    if d:
                        best_date, best_time = d, t
                        break
                if not best_date:
                    return None

                # Use SOPInstanceUID from pydicom if available, otherwise try SimpleITK
                final_sop_instance_uid = (
                    sop_instance_uid if sop_instance_uid else (reader.GetMetaData("0008|0018").strip() if "0008|0018" in metadata_keys else None)
                )

                return ("CT", best_date, best_time, frame_uid, instance_number, final_sop_instance_uid, is_axial, filepath)

            except Exception as sitk_e:
                # Error reading CT with SimpleITK after pydicom identified it as CT
                print(f"Error reading CT image info from {os.path.basename(filepath)} with SimpleITK: {sitk_e}", file=sys.stderr)
                return None
        else:
            # Not an RTSTRUCT or a CT
            return None

    except InvalidDicomError:
        # Not a valid DICOM file according to pydicom
        # print(f"Skipping invalid DICOM file: {os.path.basename(filepath)}")
        return None
    except AttributeError as ae:
        # pydicom read successful, but required attribute (like Modality) missing
        # print(f"Missing required DICOM attribute in {os.path.basename(filepath)}: {ae}")
        return None
    except Exception as e:
        # General error during pydicom read or initial processing
        print(f"Error reading initial metadata from {os.path.basename(filepath)} with pydicom: {e}", file=sys.stderr)
        return None


def process_sample_directory(sample_dir, output_base_dir):
    """
    Processes a single sample directory, organizing axial CT files and associated RTSTRUCTs
    by date and frame UID, with CT/RT subdirs and prefixed frame names.
    """
    sample_name = os.path.basename(sample_dir)
    print(f"\nProcessing sample: {sample_name}")
    output_sample_dir = os.path.join(output_base_dir, sample_name)

    # Group files:
    # ct_files_grouped: {date: {frame_uid: [(instance_number, filepath)]}} - Only AXIAL CTs
    # rt_files_grouped: {referenced_frame_uid: [filepath]}
    ct_files_grouped = defaultdict(lambda: defaultdict(list))
    rt_files_grouped = defaultdict(list)
    files_processed = 0
    axial_ct_files_found = 0
    rt_struct_files_found = 0

    for filename in os.listdir(sample_dir):
        # Allow files without .dcm extension if they might be DICOM
        # if not filename.lower().endswith(".dcm"):
        #     continue

        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            continue

        files_processed += 1
        info = get_dicom_info(filepath)

        if info:
            file_type, date_obj, _, frame_uid, instance_number, _, is_axial, fpath = info

            if file_type == "CT":
                if is_axial and date_obj and frame_uid:
                    # Use a large number if instance_number is None for sorting purposes
                    sort_key = instance_number if instance_number is not None else float("inf")
                    ct_files_grouped[date_obj][frame_uid].append((sort_key, fpath))
                    axial_ct_files_found += 1
                # else: # Optional: Log non-axial or CTs missing info
                #     if not is_axial: print(f"  Skipping non-axial CT: {filename}")
                #     elif not date_obj: print(f"  Skipping CT with no date: {filename}")
                #     elif not frame_uid: print(f"  Skipping CT with no FrameUID: {filename}")

            elif file_type == "RTSTRUCT":
                if frame_uid:  # frame_uid here is the ReferencedFrameOfReferenceUID
                    rt_files_grouped[frame_uid].append(fpath)
                    rt_struct_files_found += 1

    print(f"  Scanned {files_processed} files.")
    print(f"  Found {axial_ct_files_found} axial CT files with required metadata.")
    print(f"  Found {rt_struct_files_found} RTSTRUCT files with referenced FrameUID.")

    if not ct_files_grouped:
        print("  No suitable axial CT files found to organize.")
        # Still check if there are RT structs to potentially warn about orphans?
        if rt_files_grouped:
            print(f"  Warning: Found {rt_struct_files_found} RTSTRUCT files but no corresponding axial CT series to organize them with.")
        return

    # Create organized structure and copy files
    for date_obj, frame_groups in sorted(ct_files_grouped.items()):
        date_str = date_obj.strftime("%Y-%m-%d")
        output_date_dir = os.path.join(output_sample_dir, date_str)

        for frame_uid, ct_file_list in sorted(frame_groups.items()):
            num_ct_slices = len(ct_file_list)
            prefix = "ref_" if num_ct_slices >= 100 else "meas_"
            safe_frame_uid_part = frame_uid.replace(".", "_").replace("-", "_")  # Basic sanitization
            frame_dir_name = f"{prefix}{safe_frame_uid_part}"
            output_frame_dir = os.path.join(output_date_dir, frame_dir_name)
            output_ct_dir = os.path.join(output_frame_dir, "CT")
            output_rt_dir = os.path.join(output_frame_dir, "RT")

            associated_rt_files = rt_files_grouped.get(frame_uid, [])

            print(f"  Organizing {num_ct_slices} axial CT slices for Date: {date_str}, FrameUID: ...{frame_uid[-12:]} -> {frame_dir_name}")
            if associated_rt_files:
                print(f"    Found {len(associated_rt_files)} associated RTSTRUCT file(s).")

            if PERFORM_COPY:
                try:
                    # Create CT directory first
                    os.makedirs(output_ct_dir, exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory {output_ct_dir}: {e}", file=sys.stderr)
                    continue  # Skip this frame group if CT dir creation fails

                # Create RT directory only if needed
                if associated_rt_files:
                    try:
                        os.makedirs(output_rt_dir, exist_ok=True)
                    except OSError as e:
                        print(f"Error creating directory {output_rt_dir}: {e}", file=sys.stderr)
                        # Continue processing CT even if RT dir fails? Or skip? Let's continue CT.
                        associated_rt_files = []  # Prevent trying to copy RT files

            # Sort CT files by instance number (primary key) then filepath (secondary, for stability)
            ct_file_list.sort()

            # Copy CT files
            for _, filepath in ct_file_list:
                dest_filename = os.path.basename(filepath)
                dest_path = os.path.join(output_ct_dir, dest_filename)
                if PERFORM_COPY:
                    try:
                        shutil.copy2(filepath, dest_path)
                    except Exception as e:
                        print(f"Error copying CT {filepath} to {dest_path}: {e}", file=sys.stderr)
                else:
                    print(f"    Would copy CT {dest_filename} to {output_ct_dir}")

            # Copy RTSTRUCT files if they exist and dir was created
            if associated_rt_files:
                for rt_filepath in associated_rt_files:
                    dest_filename = os.path.basename(rt_filepath)
                    dest_path = os.path.join(output_rt_dir, dest_filename)
                    if PERFORM_COPY:
                        try:
                            shutil.copy2(rt_filepath, dest_path)
                        except Exception as e:
                            print(f"Error copying RTSTRUCT {rt_filepath} to {dest_path}: {e}", file=sys.stderr)
                    else:
                        print(f"    Would copy RTSTRUCT {dest_filename} to {output_rt_dir}")

    # Optional: Report on RT structs that didn't match any processed CT frame UID
    processed_ct_frame_uids = set()
    for frame_groups in ct_files_grouped.values():
        processed_ct_frame_uids.update(frame_groups.keys())

    orphan_rt_count = 0
    for frame_uid, rt_list in rt_files_grouped.items():
        if frame_uid not in processed_ct_frame_uids:
            orphan_rt_count += len(rt_list)
            # for rt_path in rt_list:
            #     print(f"  Warning: RTSTRUCT {os.path.basename(rt_path)} references FrameUID {frame_uid}, but no corresponding axial CT series was organized.")
    if orphan_rt_count > 0:
        print(
            f"  Warning: Found {orphan_rt_count} RTSTRUCT file(s) referencing FrameUIDs for which no corresponding axial CT series were organized in this sample."
        )


def process_all_samples(main_dir, output_dir):
    """
    Processes all subdirectories (samples) in the main directory.
    """
    if not os.path.isdir(main_dir):
        print(f"Error: Main samples directory not found: {main_dir}", file=sys.stderr)
        return

    if not PERFORM_COPY:
        print("\n--- Running in DRY RUN mode. No files will be copied. ---")

    print(f"\nStarting CT organization process.")
    print(f"Input directory: {main_dir}")
    print(f"Output directory: {output_dir}")

    for item in sorted(os.listdir(main_dir)):  # Sort for consistent processing order
        item_path = os.path.join(main_dir, item)
        if os.path.isdir(item_path):
            # Check if item_path looks like a sample directory (e.g., starts with SAMPLE_)
            if item.upper().startswith("SAMPLE_"):
                process_sample_directory(item_path, output_dir)
            else:
                print(f"\nSkipping directory (doesn't look like a sample): {item}")

    print("\nCT organization process finished.")
    if not PERFORM_COPY:
        print("--- Reminder: DRY RUN mode was active. No files were copied. ---")


if __name__ == "__main__":
    # Basic check if paths are default/placeholder
    if "/path/to/" in str(MAIN_SAMPLES_DIRECTORY) or "/path/to/" in str(ORGANIZED_OUTPUT_DIRECTORY):
        print("Please update MAIN_SAMPLES_DIRECTORY and ORGANIZED_OUTPUT_DIRECTORY variables in the script.", file=sys.stderr)
    else:
        process_all_samples(MAIN_SAMPLES_DIRECTORY, ORGANIZED_OUTPUT_DIRECTORY)
