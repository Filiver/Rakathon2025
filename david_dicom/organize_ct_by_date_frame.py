import os
import SimpleITK as sitk
import shutil
from datetime import datetime
from collections import defaultdict
import sys

# --- Configuration ---
# Set the main directory containing your sample folders (e.g., SAMPLE_001, SAMPLE_002)
MAIN_SAMPLES_DIRECTORY = "/Users/davidkorcak/Documents/Rakathon2025/data/radioprotect/Rackaton Data"
# Set the base directory where the organized structure will be created
ORGANIZED_OUTPUT_DIRECTORY = "/Users/davidkorcak/Documents/Rakathon2025/data/radioprotect/Organized_CT_Data"
# Set to True to copy files, False to print actions without copying (for testing)
PERFORM_COPY = True
# -------------------


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


def get_dicom_info(filepath):
    """
    Extracts relevant information from a DICOM file using SimpleITK.

    Returns:
        tuple: (date_obj, time_obj, frame_uid, modality, instance_number, sop_instance_uid) or None
    """
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(filepath)
        reader.ReadImageInformation()

        modality = reader.GetMetaData("0008|0060").strip().upper() if "0008|0060" in reader.GetMetaDataKeys() else ""
        if modality != "CT":
            return None  # Only process CT files

        # Extract Frame of Reference UID
        frame_uid = reader.GetMetaData("0020|0052").strip() if "0020|0052" in reader.GetMetaDataKeys() else None
        if not frame_uid:
            # print(f"Warning: Missing FrameOfReferenceUID in {os.path.basename(filepath)}. Skipping.")
            return None

        # Extract Instance Number for sorting slices
        instance_num_str = reader.GetMetaData("0020|0013").strip() if "0020|0013" in reader.GetMetaDataKeys() else None
        instance_number = None
        if instance_num_str:
            try:
                instance_number = int(instance_num_str)
            except ValueError:
                # print(f"Warning: Could not parse InstanceNumber '{instance_num_str}' in {os.path.basename(filepath)}.")
                pass  # Keep instance_number as None if parsing fails

        # Extract SOP Instance UID
        sop_instance_uid = reader.GetMetaData("0008|0018").strip() if "0008|0018" in reader.GetMetaDataKeys() else None

        # --- Extract Date and Time (Prioritize Acquisition > Series > Study > Content) ---
        best_date, best_time = None, None
        priority_tags = [
            ("0008|0022", "0008|0032"),  # Acquisition Date/Time
            ("0008|0021", "0008|0031"),  # Series Date/Time
            ("0008|0020", "0008|0030"),  # Study Date/Time
            ("0008|0023", "0008|0033"),  # Content Date/Time
        ]

        for date_tag, time_tag in priority_tags:
            date_str = reader.GetMetaData(date_tag).strip() if date_tag in reader.GetMetaDataKeys() else None
            time_str = reader.GetMetaData(time_tag).strip() if time_tag in reader.GetMetaDataKeys() else None
            d, t = parse_dicom_date_time(date_str, time_str)
            if d:  # If we found a valid date, use it and stop searching
                best_date, best_time = d, t
                break

        if not best_date:
            # print(f"Warning: Could not determine date for {os.path.basename(filepath)}. Skipping.")
            return None

        return best_date, best_time, frame_uid, modality, instance_number, sop_instance_uid

    except Exception as e:
        # print(f"Error reading metadata from {os.path.basename(filepath)}: {e}")
        return None


def process_sample_directory(sample_dir, output_base_dir):
    """
    Processes a single sample directory, organizing CT files by date and frame UID.
    """
    sample_name = os.path.basename(sample_dir)
    print(f"\nProcessing sample: {sample_name}")
    output_sample_dir = os.path.join(output_base_dir, sample_name)

    # Group files: {date: {frame_uid: [(instance_number, filepath)]}}
    ct_files_grouped = defaultdict(lambda: defaultdict(list))
    files_processed = 0
    ct_files_found = 0

    for filename in os.listdir(sample_dir):
        if not filename.lower().endswith(".dcm"):
            continue

        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            continue

        files_processed += 1
        info = get_dicom_info(filepath)

        if info:
            date_obj, _, frame_uid, modality, instance_number, _ = info
            if modality == "CT" and date_obj and frame_uid:
                # Use a large number if instance_number is None for sorting purposes
                sort_key = instance_number if instance_number is not None else float("inf")
                ct_files_grouped[date_obj][frame_uid].append((sort_key, filepath))
                ct_files_found += 1

    print(f"  Scanned {files_processed} files. Found {ct_files_found} CT files with required metadata.")

    if not ct_files_grouped:
        print("  No suitable CT files found to organize.")
        return

    # Create organized structure and copy files
    for date_obj, frame_groups in sorted(ct_files_grouped.items()):
        date_str = date_obj.strftime("%Y-%m-%d")
        output_date_dir = os.path.join(output_sample_dir, date_str)

        for frame_uid, file_list in sorted(frame_groups.items()):
            # Sanitize frame_uid for directory name if needed (though usually safe)
            safe_frame_uid_name = f"frame_uid_{frame_uid.replace('.', '_')}"
            output_frame_dir = os.path.join(output_date_dir, safe_frame_uid_name)

            print(f"  Organizing {len(file_list)} files for Date: {date_str}, FrameUID: ...{frame_uid[-12:]}")

            if PERFORM_COPY:
                try:
                    os.makedirs(output_frame_dir, exist_ok=True)
                except OSError as e:
                    print(f"Error creating directory {output_frame_dir}: {e}", file=sys.stderr)
                    continue  # Skip this group if directory creation fails

            # Sort files by instance number (primary key) then filepath (secondary, for stability)
            file_list.sort()

            for _, filepath in file_list:
                dest_filename = os.path.basename(filepath)
                dest_path = os.path.join(output_frame_dir, dest_filename)

                if PERFORM_COPY:
                    try:
                        # print(f"    Copying {dest_filename} to {output_frame_dir}")
                        shutil.copy2(filepath, dest_path)  # copy2 preserves metadata
                    except Exception as e:
                        print(f"Error copying {filepath} to {dest_path}: {e}", file=sys.stderr)
                else:
                    print(f"    Would copy {os.path.basename(filepath)} to {output_frame_dir}")


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
    if "/path/to/" in MAIN_SAMPLES_DIRECTORY or "/path/to/" in ORGANIZED_OUTPUT_DIRECTORY:
        print("Please update MAIN_SAMPLES_DIRECTORY and ORGANIZED_OUTPUT_DIRECTORY variables in the script.", file=sys.stderr)
    else:
        process_all_samples(MAIN_SAMPLES_DIRECTORY, ORGANIZED_OUTPUT_DIRECTORY)
