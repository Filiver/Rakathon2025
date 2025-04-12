from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import os

# Parameters
num_points = 1000  # Number of points to sample
radius = 1.0       # Radius of the sphere
shift_vector = np.array([1.0, 0.2, -0.1])  # Shift to apply
output_dir = "sphere_points"

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# Sample points on the sphere using spherical coordinates
phi = np.random.uniform(0, 2 * np.pi, num_points)  # azimuthal angle
costheta = np.random.uniform(-1, 1, num_points)    # cos of polar angle
theta = np.arccos(costheta)                        # polar angle

x = radius * np.sin(theta) * np.cos(phi)
y = radius * np.sin(theta) * np.sin(phi)
z = radius * np.cos(theta)

# Stack into an array
points = np.vstack((x, y, z)).T

# Save original points to a single file
original_file = os.path.join(output_dir, "original_points.txt")
with open(original_file, 'w') as f:
    for point in points:
        f.write(f"{point[0]},{point[1]},{point[2]}\n")

# Apply shift
shifted_points = points + shift_vector

# Save shifted points to a single file
shifted_file = os.path.join(output_dir, "shifted_points.txt")
with open(shifted_file, 'w') as f:
    for point in shifted_points:
        f.write(f"{point[0]},{point[1]},{point[2]}\n")

print(
    f"Saved all original points to '{original_file}' and shifted points to '{shifted_file}'.")

# visualize


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.title("Sphere Points")
ax.scatter(points[:, 0], points[:, 1], points[:, 2],
           c='b', label='Original Points')
ax.scatter(shifted_points[:, 0], shifted_points[:, 1],
           shifted_points[:, 2], c='r', label='Shifted Points')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.legend()
plt.show()
