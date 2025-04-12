import os
import pydicom
from pydicom.errors import InvalidDicomError
from collections import defaultdict
from datetime import datetime


def get_dicom_date(ds):
    """Attempts to extract a date from common DICOM date tags."""
    for tag in ["StudyDate", "SeriesDate", "ContentDate"]:
        date_str = ds.get(tag)
        if date_str:
            try:
                # DICOM date format is YYYYMMDD
                return datetime.strptime(date_str, "%Y%m%d").date()
            except ValueError:
                continue
    return None  # Or raise an error, or return a default date


def generate_report(sample_dir, output_dir):
    """
    Generates a report for a single sample directory containing DICOM files.

    Args:
        sample_dir (str): Path to the sample directory.
        output_dir (str): Path to the directory where the report will be saved.
    """
    rs_files_info = []
    ct_files_info = {}  # Map SOPInstanceUID -> {filename, date, frame_uid, ds}
    all_referenced_ct_uids = set()
    all_ct_sop_instance_uids = set()

    print(f"Processing sample: {os.path.basename(sample_dir)}")

    for filename in os.listdir(sample_dir):
        filepath = os.path.join(sample_dir, filename)
        if not os.path.isfile(filepath):
            continue

        try:
            ds = pydicom.dcmread(filepath, stop_before_pixels=True)
            modality = ds.get("Modality", "").upper()
            sop_instance_uid = ds.get("SOPInstanceUID")
            if not sop_instance_uid:
                # print(f"Warning: Missing SOPInstanceUID in {filename}. Skipping.")
                continue

            file_date = get_dicom_date(ds)

            if modality == "RTSTRUCT" and filename.upper().startswith("RS"):
                referenced_uids_in_rs = set()
                # Extract referenced SOP Instance UIDs from contours
                if "ROIContourSequence" in ds:
                    for roi_contour in ds.ROIContourSequence:
                        if "ContourSequence" in roi_contour:
                            for contour in roi_contour.ContourSequence:
                                if "ContourImageSequence" in contour:
                                    for contour_image in contour.ContourImageSequence:
                                        ref_uid = contour_image.get(
                                            "ReferencedSOPInstanceUID")
                                        if ref_uid:
                                            referenced_uids_in_rs.add(ref_uid)
                                            all_referenced_ct_uids.add(ref_uid)

                rs_files_info.append({"filename": filename, "date": file_date,
                                     "ds": ds, "referenced_uids": referenced_uids_in_rs})

            elif modality == "CT" and filename.upper().startswith("CT"):
                frame_uid = ds.get("FrameOfReferenceUID")
                if not frame_uid:
                    # print(f"Warning: Missing FrameOfReferenceUID in CT file {filename}. Skipping.")
                    continue
                if sop_instance_uid in ct_files_info:
                    # print(f"Warning: Duplicate SOPInstanceUID {sop_instance_uid} found in {filename} and {ct_files_info[sop_instance_uid]['filename']}. Keeping first.")
                    continue

                ct_files_info[sop_instance_uid] = {
                    "filename": filename, "date": file_date, "frame_uid": frame_uid, "ds": ds}
                all_ct_sop_instance_uids.add(sop_instance_uid)

        except InvalidDicomError:
            # print(f"Skipping non-DICOM or invalid file: {filename}")
            pass
        except Exception as e:
            print(f"Error processing file {filename}: {e}")

    # Sort RS files by date (handle None dates by placing them last)
    rs_files_info.sort(key=lambda x: x["date"]
                       if x["date"] else datetime.max.date())

    report_lines = []
    report_lines.append(f"Report for Sample: {os.path.basename(sample_dir)}")
    report_lines.append("=" * 30)
    report_lines.append("RTSTRUCT Files and Related CT Scans (Sorted by Date)")
    report_lines.append("-" * 30)

    processed_ct_for_rs = set()  # Keep track of CTs listed under RS files

    if not rs_files_info:
        report_lines.append("No RTSTRUCT files found.")
    else:
        for rs_info in rs_files_info:
            rs_date_str = rs_info["date"].strftime(
                "%Y-%m-%d") if rs_info["date"] else "Unknown Date"
            report_lines.append(
                f"\nDate: {rs_date_str} -> RS File: {rs_info['filename']}")

            related_cts_by_frame = defaultdict(list)
            for ref_uid in rs_info["referenced_uids"]:
                if ref_uid in ct_files_info:
                    ct_info = ct_files_info[ref_uid]
                    related_cts_by_frame[ct_info["frame_uid"]].append(
                        ct_info["filename"])
                    # Mark this CT as processed
                    processed_ct_for_rs.add(ref_uid)

            if not related_cts_by_frame:
                report_lines.append(
                    "  -> No related CT files found or CT files missing FrameOfReferenceUID.")
            else:
                report_lines.append(
                    "  -> Related CT Scans (Grouped by Frame of Reference UID):")
                # Sort groups by Frame UID for consistent output
                for frame_uid, ct_filenames in sorted(related_cts_by_frame.items()):
                    # Sort filenames within the group
                    ct_filenames.sort()
                    report_lines.append(f"    -> Frame UID: {frame_uid}")
                    report_lines.append(
                        f"       Files (Count: {len(ct_filenames)}): {', '.join(ct_filenames)}")

    report_lines.append("\n" + "=" * 30)
    report_lines.append("Unrelated CT Scans (Grouped by Date and Frame UID)")
    report_lines.append("-" * 30)

    # {date: {frame_uid: [filenames]}}
    unrelated_ct_groups = defaultdict(lambda: defaultdict(list))
    unrelated_ct_found = False

    # Iterate through all found CT files
    for sop_uid, ct_info in ct_files_info.items():
        # Check if this CT was referenced by *any* RS file processed earlier
        if sop_uid not in all_referenced_ct_uids:
            ct_date = ct_info["date"]
            frame_uid = ct_info["frame_uid"]
            filename = ct_info["filename"]
            unrelated_ct_groups[ct_date][frame_uid].append(filename)
            unrelated_ct_found = True

    if not unrelated_ct_found:
        report_lines.append("No unrelated CT scans found.")
    else:
        # Sort groups first by date (None dates last), then by Frame UID
        sorted_dates = sorted(unrelated_ct_groups.keys(),
                              key=lambda d: d if d else datetime.max.date())
        for date in sorted_dates:
            date_str = date.strftime("%Y-%m-%d") if date else "Unknown Date"
            frame_groups = unrelated_ct_groups[date]
            report_lines.append(f"\nDate: {date_str}")
            # Sort by Frame UID within the date
            for frame_uid, filenames in sorted(frame_groups.items()):
                # Sort filenames within the group
                filenames.sort()
                report_lines.append(f"  -> Frame UID: {frame_uid}")
                report_lines.append(
                    f"     Files (Count: {len(filenames)}): {', '.join(filenames)}")

    # Write report to file
    report_filename = f"{os.path.basename(sample_dir)}_report.txt"
    report_filepath = os.path.join(output_dir, report_filename)
    os.makedirs(output_dir, exist_ok=True)
    try:
        with open(report_filepath, "w") as f:
            f.write("\n".join(report_lines))
        print(f"Report generated: {report_filepath}")
    except IOError as e:
        print(f"Error writing report file {report_filepath}: {e}")


def process_all_samples(main_dir, output_dir):
    """
    Processes all subdirectories (samples) in the main directory.

    Args:
        main_dir (str): Path to the main directory containing sample subdirectories.
        output_dir (str): Path to the directory where reports will be saved.
    """
    if not os.path.isdir(main_dir):
        print(f"Error: Main directory not found: {main_dir}")
        return

    print(f"Starting processing in main directory: {main_dir}")
    print(f"Reports will be saved to: {output_dir}")

    for item in os.listdir(main_dir):
        item_path = os.path.join(main_dir, item)
        if os.path.isdir(item_path):
            generate_report(item_path, output_dir)

    print("Processing finished.")


# --- Configuration ---
# Set the main directory containing your sample folders
MAIN_SAMPLES_DIRECTORY = "data/radioprotect/Rakathon Data"
# Set the directory where you want to save the reports
REPORTS_OUTPUT_DIRECTORY = "data/radioprotect/Rakathon Data Organized"
# -------------------

if __name__ == "__main__":
    # Basic check if paths are placeholder
    if "/path/to/" in MAIN_SAMPLES_DIRECTORY or "/path/to/" in REPORTS_OUTPUT_DIRECTORY:
        print("Please update MAIN_SAMPLES_DIRECTORY and REPORTS_OUTPUT_DIRECTORY variables in the script.")
    else:
        process_all_samples(MAIN_SAMPLES_DIRECTORY, REPORTS_OUTPUT_DIRECTORY)
