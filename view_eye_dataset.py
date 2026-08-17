"""
View BioID dataset: .pgm face images with eye positions from .eye files
"""

import random
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image

DATASET_DIR = "D:\\PBL\\ML_dataset\\BioID-FaceDatabase-V1.2"   # <-- change this
NUM_SAMPLES = 6                              # how many images to preview


def read_eye_file(eye_path):
    """
    BioID .eye files look like:
    #LX	LY	RX	RY
    156	147	326	141
    Returns (left_x, left_y, right_x, right_y)
    """
    with open(eye_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    # skip header line(s) starting with '#'
    data_line = [l for l in lines if not l.startswith("#")][0]
    lx, ly, rx, ry = map(int, data_line.split())
    return lx, ly, rx, ry


def find_pairs(dataset_dir):
    """Match each .pgm with its corresponding .eye file."""
    root = Path(dataset_dir)
    pgm_files = sorted(root.glob("*.pgm"))
    pairs = []
    for pgm in pgm_files:
        eye_file = pgm.with_suffix(".eye")
        if eye_file.exists():
            pairs.append((pgm, eye_file))
    return pairs


def show_samples(pairs, n=6):
    sample = random.sample(pairs, min(n, len(pairs)))
    cols = min(3, len(sample))
    rows = (len(sample) + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if len(sample) > 1 else [axes]

    for ax, (pgm_path, eye_path) in zip(axes, sample):
        img = Image.open(pgm_path)
        ax.imshow(img, cmap="gray")
        ax.set_title(pgm_path.stem, fontsize=9)
        ax.axis("off")

        lx, ly, rx, ry = read_eye_file(eye_path)
        ax.add_patch(patches.Circle((lx, ly), radius=4, color="lime", fill=False, linewidth=2))
        ax.add_patch(patches.Circle((rx, ry), radius=4, color="red", fill=False, linewidth=2))

    # hide any unused subplot axes
    for ax in axes[len(sample):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def print_dataset_stats(pairs):
    print(f"Found {len(pairs)} matched .pgm/.eye pairs")
    if pairs:
        sample_img = Image.open(pairs[0][0])
        print(f"e.g. size={sample_img.size}, mode={sample_img.mode}")


if __name__ == "__main__":
    pairs = find_pairs(DATASET_DIR)
    if not pairs:
        print(f"No matched .pgm/.eye pairs found in {DATASET_DIR}")
    else:
        print_dataset_stats(pairs)
        show_samples(pairs, NUM_SAMPLES)