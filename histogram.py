import os
import re
from collections import Counter
import matplotlib.pyplot as plt


def plot_frame_of_reference_histogram(metadata_path):
    if not os.path.isfile(metadata_path):
        print(f"File not found: {metadata_path}")
        return

    # Read the full metadata.txt content
    with open(metadata_path, 'r') as f:
        content = f.read()

    # Extract all Frame of Reference UID values using regex
    uids = re.findall(r"Frame of Reference UID:.*?UI:\s*([^\n]+)", content)

    # Count occurrences of each UID
    uid_counts = Counter(uids)

    # Plot histogram
    plt.figure(figsize=(12, 6))
    plt.bar(uid_counts.keys(), uid_counts.values())
    plt.xticks(rotation=90)
    plt.title("Histogram of Frame of Reference UIDs")
    plt.xlabel("Frame of Reference UID")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.show()


# Example usage
metadata_file_path = os.path.join(
    "radioprotect", "data", "SAMPLE_001", "loaded", "SAMPLE_001_metadata.txt")
plot_frame_of_reference_histogram(metadata_file_path)
