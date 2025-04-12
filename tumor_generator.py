# tumor_sphere_generator.py

import numpy as np

def generate_sphere(num_points=1000) -> np.ndarray:
    """
    Generate 3D points on the surface of a sphere.
    
    Parameters:
        num_points (int): Number of 3D surface points to generate.
    
    Returns:
        np.ndarray: Array of shape (N, 3) with [x, y, z] coordinates.
    """
    # Randomly sample spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi, num_points)  # azimuthal angle
    phi = np.random.uniform(0, np.pi, num_points)        # polar angle

    # Base radius (unit sphere)
    r = 1.0

    # Calculate Cartesian coordinates from spherical coordinates
    x = r * np.sin(phi) * np.cos(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(phi)

    # Return the points on the sphere
    return np.column_stack((x, y, z))


def generate_biconcave_shape_from_sphere(sphere_points: np.ndarray, deformation_scale=0.3) -> np.ndarray:
    """
    Generate 3D points on a biconcave shape (like a red blood cell) based on an input set of points on a sphere.
    
    Parameters:
        sphere_points (np.ndarray): Array of shape (N, 3) with [x, y, z] coordinates on the sphere.
        deformation_scale (float): Degree of flattening deformation to apply.
    
    Returns:
        np.ndarray: Array of shape (N, 3) with [x, y, z] coordinates on the deformed biconcave shape.
    """
    # Copy original sphere points to apply deformation
    deformed_points = sphere_points.copy()

    # Apply deformation (flattening at poles)
    for i in range(sphere_points.shape[0]):
        x, y, z = sphere_points[i]
        # Calculate the polar angle (phi) from the point
        phi = np.arccos(z)  # phi is the angle from the z-axis (polar angle)
        
        # Apply flattening based on phi
        flatten_factor = deformation_scale * np.cos(phi) ** 2  # More flattening near the poles
        deformed_points[i, 2] -= flatten_factor  # Decrease the z-coordinate to create flattening

    return deformed_points


def save_points_to_txt(points: np.ndarray, filename: str):
    """
    Save 3D points to a TXT file with comma-separated values.

    Parameters:
        points (np.ndarray): Array of shape (N, 3).
        filename (str): Output file path.
    """
    np.savetxt(filename, points, delimiter=',', fmt="%.6f")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate two sets of points: one on a sphere and one on a biconcave shape.")
    parser.add_argument("output_file_sphere", help="Output TXT file path for sphere points")
    parser.add_argument("output_file_biconcave", help="Output TXT file path for biconcave points")
    parser.add_argument("--points", type=int, default=1000, help="Number of surface points to generate")
    parser.add_argument("--deformation", type=float, default=0.3, help="Deformation scale (flattening amount)")

    args = parser.parse_args()

    # Generate points on the sphere
    sphere_points = generate_sphere(num_points=args.points)
    
    # Generate the deformed biconcave shape based on the sphere points
    biconcave_points = generate_biconcave_shape_from_sphere(sphere_points, deformation_scale=args.deformation)
    
    # Save the points to files
    save_points_to_txt(sphere_points, args.output_file_sphere)
    save_points_to_txt(biconcave_points, args.output_file_biconcave)

    print(f"Generated sphere points saved to {args.output_file_sphere}")
    print(f"Generated biconcave points saved to {args.output_file_biconcave}")
