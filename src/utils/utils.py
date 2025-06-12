import os
import json
import random
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
import sqlite3



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

def resolve_image_storage_locations(batch_ids: list[str], cutout_ids: list[str], cfg) -> dict[str, dict[str, int]]:

    data = {}

    primary_storage_base_downloads = secondary_storage_base_downloads = tertiary_storage_base_downloads = 0
    for batch_id, cutout_id in zip(batch_ids, cutout_ids):
        image_filename = f"{cutout_id}.png"

        species = query_for_cutout_metadata(cutout_id, cfg)

        # List of storage locations in order of preference.
        storages = [
            ("primary", Path(Path(cfg.paths.primary_longterm_storage, "semifield-cutouts"), batch_id, image_filename)),
            ("secondary", Path(Path(cfg.paths.secondary_longterm_storage, "semifield-cutouts"), batch_id, image_filename)),
            ("tertiary", Path(Path(cfg.paths.tertiary_longterm_storage, "semifield-cutouts"), batch_id, image_filename))
        ]

        # Try each storage location until the image is found and copied
        primary_storage_base_downloads = 0
        secondary_storage_base_downloads = 0
        tertiary_storage_base_downloads = 0
        for storage_name, storage_path in storages:
            if storage_path.exists():
                if storage_name == "primary":
                    primary_storage_base_downloads = 1
                elif storage_name == "secondary":
                    secondary_storage_base_downloads = 1
                elif storage_name == "tertiary":
                    tertiary_storage_base_downloads = 1
                break  # Exit .

        if species not in data:
            data[species] =  {
                f"primary: {cfg.paths.primary_longterm_storage}": 0,
                f"secondary: {cfg.paths.secondary_longterm_storage}": 0,
                f"tertiary: {cfg.paths.tertiary_longterm_storage}": 0
            }

        data[species][f"primary: {cfg.paths.primary_longterm_storage}"] += primary_storage_base_downloads
        data[species][f"secondary: {cfg.paths.secondary_longterm_storage}"] += secondary_storage_base_downloads
        data[species][f"tertiary: {cfg.paths.tertiary_longterm_storage}"] += tertiary_storage_base_downloads

    return data

def query_for_cutout_metadata(cutout_id, cfg) -> str:

    # Connect to database (READ ONLY)
    conn = sqlite3.connect(f"file:{cfg.paths.sql_database}?mode=ro", uri=True)
    cursor = conn.cursor()

    # Get column names
    cursor.execute("PRAGMA table_info(semif_cutouts);")
    columns = [col[1] for col in cursor.fetchall()]

    # Fetch the single row with the given cutout_id
    cursor.execute("SELECT * FROM semif_cutouts WHERE cutout_id = ?", (cutout_id,))
    row = cursor.fetchone()

    if row is None:
        raise ValueError(f"No entry found for cutout_id: {cutout_id}")

    # Turn the row into a dictionary
    row_dict = dict(zip(columns, row))

    # Attempt to parse JSON fields
    try:
        row_dict['cutout_props'] = json.loads(row_dict['cutout_props'])
        row_dict['category'] = json.loads(row_dict['category'])
    except json.JSONDecodeError as e:
        raise ValueError(f"Failed to parse JSON fields for cutout_id {cutout_id}: {e}")

    # Extract species
    species = row_dict['category']['common_name'].upper()

    return species


def read_recipe(json_file_path) -> tuple[list[str], list[str]]:
    # Load your JSON file
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # Initialize empty lists to store batch_id and cutout_id
    batch_ids = []
    cutout_ids = []

    # Loop through the synthetic images and their cutouts
    for image in data.get("synthetic_images", []):
        for cutout in image.get("cutouts", []):
            batch_ids.append(cutout.get("batch_id"))
            cutout_ids.append(cutout.get("cutout_id"))

    return batch_ids, cutout_ids

def count_all_files(dir_path) -> int:
    total = 0
    for root, dirs, files in os.walk(dir_path):
        total += len(files)
    return total

def clear_directory(dir_path) -> None:
    if not os.path.isdir(dir_path):
        return None
    for entry in os.listdir(dir_path):
        full_path = os.path.join(dir_path, entry)
        if os.path.isfile(full_path) or os.path.islink(full_path):
            os.remove(full_path)
        elif os.path.isdir(full_path):
            # Recursively remove contents
            for root, dirs, files in os.walk(full_path, topdown=False):
                for f in files:
                    os.remove(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(full_path)