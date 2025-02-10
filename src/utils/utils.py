import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd


def filter_area(df, lower, upper):
    filtered_dfs = []
    for spec in df.common_name.unique():
        temp = df[df["common_name"] == spec]

        mean = temp.area.describe()["mean"]
        min = temp.area.describe()["min"]
        max = temp.area.describe()["max"]
        per25 = temp.area.describe()["25%"]
        per50 = temp.area.describe()["50%"]
        per75 = temp.area.describe()["75%"]

        if type(lower) is int:
            lower_area_limit = lower
        elif lower is None:
            lower_area_limit = 0
        elif lower == "mean":
            lower_area_limit = mean
        elif lower == "min":
            lower_area_limit = min
        elif lower == "max":
            lower_area_limit = max
        elif lower == "per25":
            lower_area_limit = per25
        elif lower == "per50":
            lower_area_limit = per50
        elif lower == "per75":
            lower_area_limit = per75

        if type(upper) is int:
            upper_area_limit = upper
        elif upper == "mean":
            upper_area_limit = mean
        elif upper == "min":
            upper_area_limit = min
        elif upper == "max":
            upper_area_limit = max
        elif upper == "per25":
            upper_area_limit = per25
        elif upper == "per50":
            upper_area_limit = per50
        elif upper == "per75":
            upper_area_limit = per75

        temp = temp[
            (temp["area"] < upper_area_limit) & (lower_area_limit < temp["area"])
        ]
        filtered_dfs.append(temp)
    filtered_df = pd.concat(filtered_dfs)
    return filtered_df

def get_cutouts(cfg, cutoutids):
    cutoutmeta = [Path(cfg.paths.cutoutdir, x + ".json") for x in cutoutids]
    cutouts = []
    for i in cutoutmeta:
        with open(str(i), 'r') as file:
            cut = json.load(file)
        cutouts.append(cut)
    return cutouts

def get_random_background(cfg):
    # Define the directory you want to search
    backdirectory = Path(cfg.paths.backgrounddir)
    # Define the extensions you want to search for
    extensions = ['*.jpg', '*.jpeg', '*.JPEG', '*.JPG']

    # Use list comprehension to gather all files matching the extensions
    backgroundfiles = [file for ext in extensions for file in backdirectory.glob(ext)]

    # backgroundmeta = list(Path(cfg.data.backgrounddir).glob("*.json"))
    randombackground = random.choice(backgroundfiles)
    # randombackground= create_background_dataclass_from_json(randombackground_json)
    return randombackground


def is_clockwise(contour: np.ndarray) -> bool:
    """Determines if the points in the contour are arranged in clockwise order.
    
    Args:
        contour (np.ndarray): An array of contour points.

    Returns:
        bool: True if contour is clockwise, otherwise False.
    """
    value = 0
    num = len(contour)
    for i in range(num):
        p1 = contour[i]
        p2 = contour[(i + 1) % num]  # Circular indexing
        value += (p2[0][0] - p1[0][0]) * (p2[0][1] + p1[0][1])
    return value < 0

def get_merge_point_idx(contour1: np.ndarray, contour2: np.ndarray) -> Tuple[int, int]:
    """Finds the indices of the closest points between two contours.
    
    Args:
        contour1 (np.ndarray): First contour.
        contour2 (np.ndarray): Second contour.

    Returns:
        Tuple[int, int]: Indices of the closest points in contour1 and contour2 respectively.
    """
    c1 = contour1.reshape(-1, 2)
    c2 = contour2.reshape(-1, 2)
    dist_matrix = np.sum((c1[:, np.newaxis] - c2[np.newaxis, :]) ** 2, axis=2)
    return np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)

def merge_contours(contour1: np.ndarray, contour2: np.ndarray, idx1: int, idx2: int) -> np.ndarray:
    """Merges two contours based on provided indices of closest points.

    Args:
        contour1 (np.ndarray): First contour.
        contour2 (np.ndarray): Second contour.
        idx1 (int): Index of the merge point in contour1.
        idx2 (int): Index of the merge point in contour2.

    Returns:
        np.ndarray: The resulting merged contour.
    """
    new_contour = np.concatenate([
        contour1[:idx1 + 1],
        contour2[idx2:],
        contour2[:idx2 + 1],
        contour1[idx1:]
    ])
    return np.array(new_contour)

def merge_with_parent(contour_parent: np.ndarray, contour: np.ndarray) -> np.ndarray:
    """Merges a contour with its parent contour.

    Args:
        contour_parent (np.ndarray): The parent contour.
        contour (np.ndarray): The contour to merge.

    Returns:
        np.ndarray: The merged contour.
    """
    if not is_clockwise(contour_parent):
        contour_parent = contour_parent[::-1]
    if is_clockwise(contour):
        contour = contour[::-1]
    idx1, idx2 = get_merge_point_idx(contour_parent, contour)
    new_contour = merge_contours(contour_parent, contour, idx1, idx2)
    return new_contour

def normalize_coordinates(coordinates: List[List[int]], width: int, height: int, x, y) -> List[List[float]]:
    """Normalizes coordinates of contours to relative positions based on image dimensions.
    
    Args:
        coordinates (List[List[int]]): List of coordinate lists.
        width (int): Width of the image.
        height (int): Height of the image.

    Returns:
        List[List[float]]: Normalized coordinates as a list of lists.
    """
    normalized_coordinates = []
    for coord_list in coordinates:
        normalized_list = []
        for i, coord in enumerate(coord_list):
            if i % 2 == 0:  # x-coordinate
                adjsuted_x_coord = coord + x
                normalized_list.append(adjsuted_x_coord / width)
            else:           # y-coordinate
                adjsuted_y_coord = coord + y
                normalized_list.append(adjsuted_y_coord / height)
        normalized_coordinates.append(normalized_list)
    return normalized_coordinates

def mask2polygon_holes(image):
    contours, hierarchies = cv2.findContours(image.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        return []
    contours_parent = [contour if hierarchies[0][i][3] < 0 and len(contour) >= 3 else np.array([])
                    for i, contour in enumerate(contours)]
    for i, contour in enumerate(contours):
        parent_idx = hierarchies[0][i][3]
        if parent_idx >= 0 and len(contour) >= 3:
            contour_parent = contours_parent[parent_idx]

            if contour_parent.size > 0:
                contours_parent[parent_idx] = merge_with_parent(contour_parent, contour)
    # contours_parent_tmp = [contour for contour in contours_parent if contour]
    contours_parent_tmp = [contour for contour in contours_parent if contour.size > 0]
    polygons = [contour.flatten().tolist() for contour in contours_parent_tmp]
    return polygons

def is_rectangular(mask, threshold_percentage):

    # Calculate the total number of pixels
    total_pixels = mask.size
    # Calculate the number of non-zero pixels
    non_zero_pixels = np.count_nonzero(mask)

    # Calculate the percentage of non-zero pixels
    filled_percentage = (non_zero_pixels / total_pixels) * 100

    # Check if the filled percentage meets or exceeds the threshold
    is_filled_enough = filled_percentage >= threshold_percentage
    
    return is_filled_enough, filled_percentage