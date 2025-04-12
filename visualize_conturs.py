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
    all_points = {}
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
                    if roi_name not in all_points:
                        all_points[roi_name] = []
                    all_points[roi_name].extend(points.tolist())

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Contours from TXT Files")
    plt.tight_layout()
    plt.show()
    return all_points


def visualize_all_contours_from_dict(dict, spacing, origin):
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
    all_points = {}
    """
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
                    if roi_name not in all_points:
                        all_points[roi_name] = []
                    all_points[roi_name].extend(points.tolist())
    """
    for roi_name, points in dict.items():
        points = np.array(points)
        if points.size == 0:
            continue
        # print(points[:, :2])
        # print(points[:, :2]-origin[1:])
        # print((points[:, :2] - origin[1:]) / spacing[1:])
        ij = ((points[:, :2] - origin[1:]) / spacing[1:]).astype(int)
        # print(ij[:, :2])
        # input()
        ax.plot(ij[:, 0], ij[:, 1], points[:, 2],
                label=roi_name, color=get_color_for_label(roi_name))
        if roi_name not in all_points:
            all_points[roi_name] = []
        all_points[roi_name].extend(points.tolist())
    ax.legend()
    ax.axis('equal')

    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("3D Contours from TXT Files")
    # set equal axes
    plt.tight_layout()
    plt.show()
    return all_points


def visualize_two_contour_dicts(dict1, dict2, spacing, origin):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import numpy as np

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    def transform_points(points):
        points = np.array(points)
        if points.size == 0:
            return None
        ij = ((points[:, :2] - origin[1:]) / spacing[1:]).astype(int)
        return ij, points[:, 2]

    def plot_dict(dict_data, color):
        for roi_name, points in dict_data.items():
            transformed = transform_points(points)
            if transformed is None:
                continue
            ij, z = transformed
            ax.plot(ij[:, 0], ij[:, 1], z, label=roi_name, color=color)

    plot_dict(dict1, color='red')
    plot_dict(dict2, color='blue')

    ax.axis('equal')
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")
    ax.set_title("Overlay of Two 3D Contour Sets")
    plt.tight_layout()
    ax.legend()
    plt.show()


if __name__ == "__main__":
    # Example usage
    import os
    dir = "pointclouds_by_rs/SAMPLE_004"
    subdirs = os.listdir(dir)
    for subdir in subdirs:
        if subdir != "RS.1.2.246.352.221.52794105832653520384075859529424384185__RS.1.2.246.352.221.57475698521031836325890889930332779148__RS.1.2.246.352.221.530968562667814550516230413739928631461__RS.1.2.246.352.221.534409961817902190914559599786692832400":
            continue
        print(f"Processing subdirectory: {subdir}")
        all_points = visualize_all_contours_from_txt(
            os.path.join(dir, subdir, "txt"))
        for roi_name, points_list in all_points.items():
            print(f"ROI: {roi_name}")
            print(points_list)
            input()
