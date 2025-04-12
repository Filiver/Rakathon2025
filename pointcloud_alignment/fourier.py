import torch
import torch.nn.functional as F
import SimpleITK as sitk
import numpy as np
import moviepy
from torch_helpers import maxloc
import pydicom

# dont touch this until necessary
BODY_BBOX = (150, 0, 350, 512)
REGISTRATION_SHAPE = (512, 768)
HISTOGRAM_BINS = 2**12
REFERENCE_WINDOW_CENTER = 195
REFERENCE_WINDOW_WIDTH = 800
MEASUREMENT_WINDOW_CENTER = None
MEASUREMENT_WINDOW_WIDTH = None
APPLY_LOG_ON_REFERENCE = True
REFERENCE_LOG_SCALER = 5.0
APPLY_LOG_ON_MEASUREMENT = True
MEASUREMENT_LOG_SCALER = 5.0


def equalize_colored_volume(volume):
    histc = torch.histc(volume, bins=HISTOGRAM_BINS)
    histc = histc.cumsum(dim=0)
    histc = torch.where(histc > 0, histc, 1e-6)  # Keep this safeguard
    cdf_min = torch.min(histc[histc > 1e-7])  # Find min non-zero value robustly
    cdf_max = histc[-1]
    if cdf_max == cdf_min:
        return torch.zeros_like(volume, dtype=torch.float32)  # Return 0s or handle as appropriate
    cdf = (histc - cdf_min) / (cdf_max - cdf_min)
    cdf.clamp_(0, 1)
    scaled_volume = ((volume * (HISTOGRAM_BINS - 1)).long()).clamp(0, HISTOGRAM_BINS - 1)
    return cdf[scaled_volume]


def dicom_filenames_from_dir(scan_dir):
    """
    Get the DICOM filenames from the specified directory.
    """
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(scan_dir)
    return dicom_names


def get_scan_image_from_filenames(filenames):
    """
    Load DICOM scans from a list of filenames.
    """
    reader = sitk.ImageSeriesReader()
    reader.MetaDataDictionaryArrayUpdateOn()
    reader.LoadPrivateTagsOn()
    reader.SetFileNames(filenames)
    image = reader.Execute()
    return image, reader


def normalize_CT_frame(dcm, _window_center=None, _window_width=None, apply_log=False, log_scaler=1.0):
    """
    Normalize a single CT frame using the metadata.
    """
    # Get the rescale slope and intercept from the metadata
    frame = dcm.pixel_array.astype(np.float32)
    rescale_slope = float(dcm.RescaleSlope)
    rescale_intercept = float(dcm.RescaleIntercept)
    # HU frame
    hu_frame = frame * rescale_slope + rescale_intercept
    # Window
    window_center = float(dcm.WindowCenter) if _window_center is None else _window_center
    window_width = float(dcm.WindowWidth) if _window_width is None else _window_width
    # Calculate the min and max values for the window
    min_val = window_center - window_width / 2
    max_val = window_center + window_width / 2
    hu_frame = np.clip(hu_frame, min_val, max_val)
    # Normalize the frame to [0, 1]
    normalized_frame = (hu_frame - min_val) / (max_val - min_val)
    if apply_log:
        # Apply logarithmic scaling
        denom = np.log(1 + log_scaler)
        normalized_frame = np.log(1 + normalized_frame * log_scaler) / denom
    return normalized_frame.astype(np.float32)


def raw_data_from_image(
    image: sitk.Image, reader: sitk.ImageSeriesReader, _window_width=None, _window_center=None, _apply_log=False, _log_scaler=1.0
) -> dict:
    """
    Convert SimpleITK image to a numpy array.
    """
    normed_frames = []
    for f in reader.GetFileNames():
        dcm = pydicom.dcmread(f)
        normed_frames.append(normalize_CT_frame(dcm, _window_center, _window_width, _apply_log, _log_scaler))
    return {
        "array": np.stack(normed_frames, axis=0),
        "spacing": image.GetSpacing(),
        "origin": image.GetOrigin(),
    }


def mask_non_body_parts(image: np.ndarray, body_bbox: tuple) -> np.ndarray:
    """
    Clip the image to the specified bounding box.
    """
    assert image.ndim == 3, "Image must be a 3D tensor"
    new_imgs = np.zeros_like(image, dtype=np.float32)
    min_val = image.min()  # darkest
    for i in range(image.shape[0]):
        # Create a mask for the body bounding box
        mask = np.zeros_like(image[i], dtype=np.float32)
        mask[body_bbox[0] : body_bbox[2], body_bbox[1] : body_bbox[3]] = 1.0
        # Apply the mask to the image
        new_imgs[i] = image[i] * mask + min_val * (1 - mask)
    return new_imgs


def rescale_volume(volume: torch.Tensor, original_spacing: tuple, new_spacing: tuple) -> torch.Tensor:
    """Rescale the volume to a new spacing."""
    assert volume.ndim == 3, "Volume must be a 3D tensor"
    # Flip the scaling coefficients to (z,y,x)
    original_scale = torch.tensor(original_spacing[::-1])
    new_scale = torch.tensor(new_spacing[::-1])
    # Extents of the scan
    original_extents = torch.tensor(volume.shape) * original_scale
    # Shape of the reference scan when resampled to the measurement scan's resolution
    new_shape = (torch.round(original_extents / new_scale).long()).tolist()
    # Interpolate the reference scan to match the measurement scan's resolution
    # First interpolate the 2D slices with bicubic, the Z axis with trilinear
    # Convert volume to float once
    volume_float = volume.float()
    # Interpolate each 2D slice using bicubic
    interp_2d_slices = [
        torch.nn.functional.interpolate(
            volume_float[i].unsqueeze(0).unsqueeze(0),  # Add batch and channel dims
            size=new_shape[1:],  # Target H, W
            mode="bicubic",
            align_corners=False,
        ).clamp(0.0, 1.0)
        for i in range(volume.shape[0])  # Iterate through depth
    ]
    # Stack the interpolated slices
    interp_2d_only = torch.stack(interp_2d_slices, dim=2)  # Shape: (original_depth, new_height, new_width)
    reference_interp = torch.nn.functional.interpolate(interp_2d_only, size=new_shape, mode="trilinear", align_corners=False).clamp(0.0, 1.0)
    return reference_interp.squeeze()


def zero_pad_volume_to_shape(volume: torch.Tensor, target_shape: tuple) -> tuple[torch.Tensor, list]:
    """Pad the volume to the target shape."""
    # Calculate padding sizes for each dimension
    padding = []
    for dim, target_dim in zip(volume.shape[::-1], target_shape[::-1]):
        dim_delta = target_dim - dim
        padding.extend([dim_delta // 2, dim_delta - dim_delta // 2])
    # Pad the volume
    padded_volume = F.pad(volume, tuple(padding), mode="constant", value=0)
    return padded_volume, padding


def volumes_to_video(ref: torch.Tensor, meas: torch.Tensor, filename: str, fps: int = 15):
    """
    Creates an HDR video from a 3D volume tensor.

    Args:
        volume (torch.Tensor): The input 3D volume (depth, height, width).
        filename (str): The name of the output video file (e.g., 'output.mp4').
        fps (int): Frames per second for the video.
    """
    depth, height, width = ref.shape

    def get_frame(t):
        """Returns the frame corresponding to time t as 10-bit uint16 for HDR."""
        frame_index = int(t * fps)
        if frame_index >= depth:
            frame_index = depth - 1
        ref_slice = ref[frame_index]
        meas_slice = meas[frame_index]
        abs_diff = torch.abs(ref_slice - meas_slice)
        ref_slice_int = (ref_slice * 255).byte().numpy()
        meas_slice_int = (meas_slice * 255).byte().numpy()
        diff_slice_int = (abs_diff * 255).byte().numpy()
        slice_int = np.concatenate((ref_slice_int, meas_slice_int, diff_slice_int), axis=1)
        frame_rgb = np.stack([slice_int] * 3, axis=-1)
        return frame_rgb

    duration = depth / fps
    clip = moviepy.VideoClip(
        get_frame,
        duration=duration,
    )
    clip.write_videofile(filename, fps=fps, codec="libx264")
    clip.close()


def register_volumes_fft(ref_volume, meas_volume, device="cpu"):
    """
    Register two volumes using Fourier transform.
    """
    # Compute the Fourier transforms
    ref_fft = torch.fft.rfftn(ref_volume.to(device))
    meas_fft = torch.fft.rfftn(meas_volume.to(device))
    # Compute the cross-power spectrum
    cross_power_spectrum = (ref_fft * torch.conj(meas_fft)) / (torch.abs(ref_fft) + 1e-6)
    # Compute the inverse Fourier transform to get the registered volume
    registered_volume = torch.fft.irfftn(cross_power_spectrum)
    registered_volume.abs_()
    registered_volume = torch.fft.fftshift(registered_volume)
    return maxloc(registered_volume)


def crop_volume_to_center(volume: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    d, h, w = target_shape
    dact, hact, wact = volume.shape
    # Calculate the center of the original volume
    center_d = dact // 2
    center_h = hact // 2
    center_w = wact // 2
    # Calculate the start and end indices for cropping
    start_d = center_d - d // 2
    end_d = start_d + d
    start_h = center_h - h // 2
    end_h = start_h + h
    start_w = center_w - w // 2
    end_w = start_w + w
    # Crop the volume
    cropped_volume = volume[start_d:end_d, start_h:end_h, start_w:end_w]
    return cropped_volume


def transform_meas_to_ref(translation_tensor, meas_volume):
    """
    Apply the translation to the measurement volume.
    """
    # Create a grid of indices for the measurement volume
    d, h, w = meas_volume.shape
    spaces = [
        torch.linspace(-1, 1, w),
        torch.linspace(-1, 1, h),
        torch.linspace(-1, 1, d),
    ]
    grid = torch.meshgrid(spaces, indexing="ij")
    grid = torch.stack(grid, dim=-1).float()
    grid = grid.permute(2, 1, 0, 3)  # Change to (N, C, H, W) format
    # Translate the grid by the translation tensor
    # Reorder translation from (z, y, x) to (x, y, z)
    translation_z, translation_y, translation_x = translation_tensor
    translation_grid_order = torch.tensor([translation_x, translation_y, translation_z], dtype=torch.float)
    # Normalize translation to grid space (-1 to 1)
    dims = torch.tensor([w, h, d], dtype=torch.float)  # x, y, z sizes
    scaling_factor = 2 / (dims - 1)  # Scaling for each axis
    translation_normalized = translation_grid_order * scaling_factor
    # Apply translation (subtract to shift volume in the positive direction)
    translated_grid = grid - translation_normalized.view(1, 1, 1, 3)
    # Interpolate the measurement volume at the translated grid locations
    transformed_meas_volume = F.grid_sample(meas_volume.unsqueeze(0).unsqueeze(0), translated_grid.unsqueeze(0), mode="bilinear", align_corners=False)
    return transformed_meas_volume.squeeze(0).squeeze(0)


def meas_to_ref_final_step(meas_volume, meas_spacing, ref_spacing, ref_shape):
    meas_rescaled_to_ref = rescale_volume(meas_volume, meas_spacing, ref_spacing)
    meas_rescaled_to_ref_padded, padding = zero_pad_volume_to_shape(meas_rescaled_to_ref, ref_shape)
    return meas_rescaled_to_ref_padded


def align_measurement_to_reference_scan(reference_filenames, measurement_filenames, save_videos=False) -> dict[str, torch.Tensor]:
    """Align the measurement scan to the reference scan, return the aligned measurement scan in reference space.
    The returned data's coordinate ordering is (z, y, x) and the spacing is in (z,y,x).

    Args:
        reference_filenames (list[str]): CT filenames of the reference scan.
        measurement_filenames (_type_): CT filenames of the measurement scan.
        save_videos (bool, optional): Whether to save videos of the registration process. Defaults to False.

    Returns:
        dict[str, torch.Tensor]: key: "reference" (reference scan), "measurement" (aligned measurement scan), "spacing" (spacing of the reference scan).
    """
    # load data - receive filenames
    reference_scan_image, reference_reader = get_scan_image_from_filenames(reference_filenames)
    measurement_scan_image, measurement_reader = get_scan_image_from_filenames(measurement_filenames)
    # convert to numpy array
    print("Converting to numpy array...")
    reference_scan = raw_data_from_image(
        reference_scan_image, reference_reader, REFERENCE_WINDOW_WIDTH, REFERENCE_WINDOW_CENTER, APPLY_LOG_ON_REFERENCE, REFERENCE_LOG_SCALER
    )
    measurement_scan = raw_data_from_image(
        measurement_scan_image,
        measurement_reader,
        MEASUREMENT_WINDOW_WIDTH,
        MEASUREMENT_WINDOW_CENTER,
        APPLY_LOG_ON_MEASUREMENT,
        MEASUREMENT_LOG_SCALER,
    )
    # mask the non-body parts
    print("Masking non-body parts...")
    reference_scan["array"] = mask_non_body_parts(reference_scan["array"], BODY_BBOX)
    # take the volumes as torch tensors
    ref_volume = torch.from_numpy(reference_scan["array"])  # Keep as original type for equalize
    meas_volume = torch.from_numpy(measurement_scan["array"])  # Keep as original type for equalize
    # equalize the volumes - input should match expectation of equalize_colored_volume
    print("Equalizing volumes...")
    equalized_reference_volume = equalize_colored_volume(ref_volume)  # Output is float [0,1]
    equalized_measurement_volume = equalize_colored_volume(meas_volume)  # Output is float [0,1]
    # rescale the reference volume to the measurement volume's resolution
    print("Rescaling reference volume to measurement volume's resolution...")
    rescaled_reference_volume = rescale_volume(equalized_reference_volume.float(), reference_scan["spacing"], measurement_scan["spacing"])
    # crop" the volumes to the same shape
    print("Old reference volume shape:", rescaled_reference_volume.shape)
    print("New reference volume shape:", rescaled_reference_volume.shape)
    print("Padding and cropping non-interesting parts...")
    target_shape = (equalized_reference_volume.shape[0], *REGISTRATION_SHAPE)
    # zero pad the measurement volume to the reference volume's shape
    padded_measurement_volume, padding = zero_pad_volume_to_shape(equalized_measurement_volume, target_shape)
    cropped_reference_volume = crop_volume_to_center(rescaled_reference_volume, target_shape)
    # save the volumes as videos
    if save_videos:
        print("Saving videos before registration...")
        volumes_to_video(cropped_reference_volume, padded_measurement_volume, "reference_measurement_diff.mp4", fps=10)
    # register the volumes
    print("Registering volumes...")
    translation = register_volumes_fft(cropped_reference_volume, padded_measurement_volume, "mps")
    vol_ctr = (torch.tensor(target_shape) - 1) / 2
    translation_tensor = (torch.tensor(translation, dtype=torch.float32) - vol_ctr).floor()
    print("Translation:", translation_tensor)
    # Apply the translation to the measurement volume
    print("Transforming measurement volume...")
    transformed_meas_volume = transform_meas_to_ref(translation_tensor, padded_measurement_volume)
    # Save the transformed measurement volume as a video
    if save_videos:
        print("Saving videos after registration...")
        volumes_to_video(cropped_reference_volume, transformed_meas_volume, "registered_measurement_diff.mp4", fps=10)
    # Transform the translation vector to the coordinate system of the reference volume
    translation_ref = torch.tensor(translation, dtype=torch.float32)  # pixels spaced in the reference volume
    translation_ref = (
        translation_ref * torch.tensor(reference_scan["spacing"]) / torch.tensor(measurement_scan["spacing"])
    )  # pixel-space in the reference volume
    meas_translated_to_ref = meas_to_ref_final_step(
        transformed_meas_volume, measurement_scan["spacing"], reference_scan["spacing"], equalized_reference_volume.shape
    )
    if save_videos:
        print("Saving videos after final transformation in reference space...")
        volumes_to_video(equalized_reference_volume, meas_translated_to_ref, "final_registered_measurement_diff.mp4", fps=10)
    return {
        "reference": equalized_reference_volume,
        "measurement": meas_translated_to_ref,
        "spacing": torch.tensor(reference_scan["spacing"][::-1]),
    }


if __name__ == "__main__":
    import argparse

    DEFAULT_REFERENCE = "data/radioprotect/Organized_CT_Data/SAMPLE_001/2023-06-05/frame_uid_1_2_246_352_221_559666980133719263215614360979762074268"
    DEFAULT_MEASUREMENT = (
        "data/radioprotect/Organized_CT_Data/SAMPLE_001/2023-06-21/frame_uid_1_2_246_352_221_523526543250385987917834924930119139461"
    )

    parser = argparse.ArgumentParser(description="Fourier Transform Registration")
    parser.add_argument("--reference", type=str, help="Path to the reference scan directory", default=DEFAULT_REFERENCE)
    parser.add_argument("--measurement", type=str, help="Path to the measurement scan directory", default=DEFAULT_MEASUREMENT)
    args = parser.parse_args()
    # Align the measurement scan to the reference scan
    ref_filenames = dicom_filenames_from_dir(args.reference)
    meas_filenames = dicom_filenames_from_dir(args.measurement)
    res = align_measurement_to_reference_scan(ref_filenames, meas_filenames, save_videos=True)
    print(res["reference"].shape)
    print(res["measurement"].shape)
    print(res["spacing"])
