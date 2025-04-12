def visualize_all_contours_from_txt(dir_path):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import os
    import numpy as np
    import hashlib

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    color_map = {}

    def get_color_for_label(label):
        if label not in color_map:
            # Generate a consistent hash-based color from label
            hash_val = int(hashlib.sha256(label.encode()).hexdigest(), 16)
            r = (hash_val % 256) / 255.0
            g = ((hash_val >> 8) % 256) / 255.0
            b = ((hash_val >> 16) % 256) / 255.0
            color_map[label] = (r, g, b)
        return color_map[label]

    for filename in os.listdir(dir_path):
        print(f"Processing file: {filename}")
        if filename.endswith(".txt"):
            with open(os.path.join(dir_path, filename), 'r') as file:
                for line in file:
                    if ':' not in line:
                        continue
                    roi_name, points_str = line.strip().split(':', 1)
                    points = [list(map(float, p.split(','))) for p in points_str.strip(
                    ).split() if len(p.split(',')) == 3]
                    points = np.array(points)
                    if points.size == 0:
                        continue
                    ax.plot(points[:, 0], points[:, 1], points[:, 2],
                            label=roi_name, color=get_color_for_label(roi_name))

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Contours from TXT Files")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    visualize_all_contours_from_txt(
        "pointclouds_by_rs/SAMPLE_004/RS.1.2.246.352.221.52794105832653520384075859529424384185__RS.1.2.246.352.221.57475698521031836325890889930332779148__RS.1.2.246.352.221.530968562667814550516230413739928631461__RS.1.2.246.352.221.534409961817902190914559599786692832400/txt")
