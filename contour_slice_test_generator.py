import numpy as np

def generate_clean_ellipse(
    center=(0, 0),
    axes=(50, 30),
    num_points=100,
    angle=0.0,
    z_value=0.0
):
    """
    Generate clean points on the perimeter of an ellipse.

    Args:
        center: (x, y) center of the ellipse.
        axes: (a, b) lengths of the ellipse's semi-axes.
        num_points: Number of points to generate.
        angle: Rotation of the ellipse in radians.
        z_value: Constant z-value for the slice.

    Returns:
        ndarray of shape (num_points, 3) with x, y, z.
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    a, b = axes
    x = a * np.cos(t)
    y = b * np.sin(t)

    # Rotate
    cos_ang = np.cos(angle)
    sin_ang = np.sin(angle)
    x_rot = cos_ang * x - sin_ang * y
    y_rot = sin_ang * x + cos_ang * y

    # Translate and add z
    x_final = x_rot + center[0]
    y_final = y_rot + center[1]
    z_final = np.full_like(x_final, z_value)

    return np.stack((x_final, y_final, z_final), axis=1)

if __name__ == "__main__":
    # Simulate one more circular, one more elliptical
    reference = generate_clean_ellipse(center=(50, 50), axes=(30, 30), num_points=100, z_value=10)
    measured  = generate_clean_ellipse(center=(50, 50), axes=(30, 20), num_points=100, z_value=10)

    import matplotlib.pyplot as plt
    plt.figure(figsize=(6, 6))
    plt.scatter(reference[:, 0], reference[:, 1], label="Reference (circle)", color='blue', alpha=0.6)
    plt.scatter(measured[:, 0], measured[:, 1], label="Measured (ellipse)", color='red', alpha=0.6)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("Simulated Clean Contours")
    plt.tight_layout()
    plt.savefig("./images/simulated_contours.png")
    plt.show()

