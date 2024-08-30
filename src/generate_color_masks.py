import colorsys
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def generate_colors(n):
    colors = []  # Start with black as the first color
    for i in range(1, n):  # Start from 1 to keep the first color as black
        hue = i / (n-1)  # Adjust hue calculation to spread over n-1 instead of n
        saturation = 0.5
        value = 0.8
        rgb = colorsys.hsv_to_rgb(hue, saturation, value)
        # Convert to 0-255 RGB values
        rgb_255 = list(int(x * 255) for x in rgb)
        colors.append(rgb_255)
    random.shuffle(colors)
    colors.insert(0, [0,0,0])
    # exit()
    return colors

def get_distinct_colors(num_colors):
    """
    Generate a list of distinct colors, excluding the first color which will be black for the background.
    """
    cmap = plt.get_cmap('tab20b', num_colors + 1)  # Generate an extra color because we will not use one of them
    colors = cmap(range(num_colors + 1))
    # Convert colors from RGBA to RGB, normalizing to 0-255 scale
    colors = [(int(r*255), int(g*255), int(b*255)) for r, g, b, _ in colors][1:]  # Skip the first color
    random.shuffle(colors)
    return colors


def colorize_mask(mask_path, save_path, colors):
    """
    Load a grayscale semantic mask, map each label to a color from the colors list (with label 0 as black), and save the colorized version.
    """
    # Load the mask image
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError("Mask image file not found.")

    # Create an output image initialized with black color
    colored_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)

    # Map each label to its corresponding color
    unique_labels = [x for x in np.unique(mask) if x != 0]
    for label in unique_labels:
        colored_mask[mask == label] = colors[label]

    # Save the colored mask
    cv2.imwrite(save_path, colored_mask, [cv2.IMWRITE_PNG_COMPRESSION, 90])
    print(f"Colored mask saved to {save_path}")



sem_masks = sorted([x for x in Path("../data/results/semantic_masks").glob("*.png")])

save_color_sem_maskdir = Path("../data/results/color_semantic_masks")
save_color_sem_maskdir.mkdir(exist_ok=True, parents=True)



num_classes = 3

sem_colors = get_distinct_colors(num_classes)  # Add black for label 0

for semmask in sem_masks:
    sem_save_path = Path(save_color_sem_maskdir, semmask.name)

    colorize_mask(str(semmask), str(sem_save_path), sem_colors)


in_masks = sorted([x for x in Path("../data/results/instance_masks").glob("*.png")])
save_color_insta_maskdir = Path("../data/results/color_instance_masks")
save_color_insta_maskdir.mkdir(exist_ok=True, parents=True)

num_classes = 350
instacolors = get_distinct_colors(num_classes)
# instacolors = generate_colors(num_classes)
# exit()
# instacolors = [(0, 0, 0)] + get_distinct_colors(num_classes)  # Add black for label 0

for instamask in in_masks:
    insta_save_path = Path(save_color_insta_maskdir, instamask.name)

    colorize_mask(str(instamask), str(insta_save_path), instacolors)

