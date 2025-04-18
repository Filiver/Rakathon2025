import numpy as np

def generate_ellipse_points(a, b, num_points=100, center=(0, 0), noise=0.0, z=0) -> np.ndarray:
    """
    Generate points on an ellipse in 3D space.
    parameters:
        a (float): Semi-major axis length.
        b (float): Semi-minor axis length.
        num_points (int): Number of points to generate.
        center (tuple): Center of the ellipse in 2D space (x, y).
        noise (float): Standard deviation of Gaussian noise to add to points.
        z (float): Z-coordinate for all points.
    returns:
        np.ndarray: Array of shape (num_points, 3) with [x, y, z] coordinates.  
    """
    t = np.linspace(0, 2 * np.pi, num_points, endpoint=False)
    x = a * np.cos(t) + center[0]
    y = b * np.sin(t) + center[1]
    x += np.random.normal(0, noise, size=num_points)
    y += np.random.normal(0, noise, size=num_points)
    z = np.full_like(x, z)
    return np.stack((x, y, z), axis=1)


if __name__ == "__main__":
    # Simulate one more circular, one more elliptical
    reference = generate_ellipse_points(center=(50, 50), axes=(30, 30), num_points=100, z_value=10)
    measured  = generate_ellipse_points(center=(50, 50), axes=(30, 20), num_points=100, z_value=10)

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

