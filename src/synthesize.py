"""
This script is designed to process large batches of image data for synthetic image generation with a focus on CPU parallelism.
The image composition tasks involve loading, transforming, and distributing cutout images onto backgrounds, generating synthetic datasets for training.

Key Features:
- **Parallelism**: The script uses Python's `concurrent.futures.ProcessPoolExecutor` to enable CPU parallelism, allowing the processing of multiple image recipes concurrently.
- **CPU Usage**: Given the use of multiple processes, CPU usage is maximized during parallel execution, particularly for computationally heavy image transformations and file I/O operations.
"""

# Standard library imports
import json
import logging
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from pathlib import Path
from typing import List, Tuple, Dict, Union

import albumentations as A
import cv2
import numpy as np
from omegaconf import DictConfig

# from utils.utils import mask2polygon_holes, normalize_coordinates

log = logging.getLogger(__name__)

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

class ImageProcessor:
    """
    A class to handle image processing tasks such as loading images, applying transformations, 
    and overlaying them onto a background.
    """

    def __init__(self,cfg: DictConfig, num_cutouts: int):
        """
        Initialize the image processor with the number of cutouts.

        Args:
            num_cutouts (int): Number of cutouts to process.
        """
        self.cfg = cfg
        self.num_cutouts = num_cutouts
        self.resize_scale = cfg.synthesize.resize_factor  # Added scaling factor

        # Non destructive transformations
        self.transform = A.Compose([
            # A.GaussNoise(p=0.2), 
            A.HorizontalFlip(p=0.5), 
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=1),
            A.Transpose(p=0.5),
            # Weather augmentations
            # A.RandomSunFlare(flare_roi=(0.1, 0.1, 0.9, 0.9),src_radius=50, p=0.2),
            # A.RandomShadow(num_shadows_lower=1, num_shadows_upper=1, shadow_dimension=5, shadow_roi=(0, 0.5, 1, 1), p=1),
            # A.RandomBrightnessContrast(p=0.2),
            # A.RandomScale(scale_limit=self.scale_limit, p=0.5),  # Dynamically set scale_limit
        ])

        self.create_contours = cfg.synthesize.yolo_contour_labels
        self.create_bbox = cfg.synthesize.yolo_bbox_labels
    
    
    def apply_random_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a unique random Albumentations transform to the image, handling transparency.

        Args:
            img (np.ndarray): The image to be transformed.

        Returns:
            np.ndarray: The transformed image.
        """
        # Apply the transformation to the image and mask
        augmented = self.transform(image=img, mask=(img[:, :, -1] > 0).astype(np.uint8))
        img_transformed = augmented['image']
        transformed_mask = augmented['mask']
        combined_image = np.where(transformed_mask[..., None] == 1, img_transformed, 0)

        return combined_image

    def distribute_images(
        self, background: np.ndarray, images: List[np.ndarray],
        cutout_paths: List[str], mode: str = "random", min_visibility: float = 0.9,
        max_retries: int = 10
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[Dict[int, List[List[float]]]]]:
        """
        Distribute images on a background either randomly or in a semi-even grid pattern.

        Args:
            background (np.ndarray): The background image.
            images (List[np.ndarray]): List of images to distribute.
            cutout_paths (List[str]): List of paths to the cutout images.
            mode (str): The distribution mode, either 'random' or 'semi_even'. Defaults to 'random'.

        Returns:
            Tuple: The background, semantic mask, instance mask, and coordinates of placed images.
        """
        bg_height, bg_width = background.shape[:2]
        background_semantic_mask = np.zeros((bg_height, bg_width), dtype=np.uint8)
        background_instance_mask = np.zeros((bg_height, bg_width), dtype=np.uint16)

        yolo_bboxes = []
        coord_results = []
        instance_id = 1
        placed_regions = []  # List to store the coordinates of already placed cutouts

        for _, (img, cutout_metadata) in enumerate(zip(images, cutout_paths)):
            class_id = cutout_metadata['category']['class_id']
            cutout_id = cutout_metadata['cutout_id']

            # Apply transformations to the cutout
            transformed_img = self.apply_random_transform(img)
            cutout_placed = False
            for _ in range(max_retries):
                # Get random coordinates for placement
                roi_x, roi_y, img_x_start, img_y_start, img_x_end, img_y_end = (
                    self.random_coordinates(bg_height, bg_width, transformed_img)
                )
                # Calculate the bounding box for this placement
                cutout_bbox = (
                    roi_x, roi_y, 
                    min(roi_x + (img_x_end - img_x_start), bg_width), 
                    min(roi_y + (img_y_end - img_y_start), bg_height)
                )

                # Check for overlap with previously placed regions
                if not any(self.is_fully_occluded(cutout_bbox, region, min_visibility) for region in placed_regions):
                    # Place the cutout if no excessive overlap is found
                    norm_coords, yolo_bbox = self.overlay_with_alpha(
                        background, background_semantic_mask, background_instance_mask,
                        transformed_img[img_y_start:img_y_end, img_x_start:img_x_end],
                        roi_x, roi_y, class_id, instance_id
                    )

                    coord_results.append({class_id: norm_coords})
                    yolo_bboxes.append(yolo_bbox)
                    placed_regions.append(cutout_bbox)
                    instance_id += 1
                    cutout_placed = True
                    break
            if not cutout_placed:
                log.warning(f"Could not place cutout {cutout_id} after {max_retries} attempts.")


        return background, background_semantic_mask, background_instance_mask, coord_results, yolo_bboxes
    
    def is_fully_occluded(self, new_bbox: Tuple[int, int, int, int], placed_bbox: Tuple[int, int, int, int], min_visibility: float) -> bool:
        """
        Check if the new bounding box is fully occluded by an already placed bounding box.

        Args:
            new_bbox (Tuple[int, int, int, int]): The bounding box of the new cutout.
            placed_bbox (Tuple[int, int, int, int]): The bounding box of an already placed cutout.
            min_visibility (float): Minimum visibility ratio required for the new cutout.

        Returns:
            bool: True if the new cutout is fully occluded, False otherwise.
        """
        xA = max(new_bbox[0], placed_bbox[0])
        yA = max(new_bbox[1], placed_bbox[1])
        xB = min(new_bbox[2], placed_bbox[2])
        yB = min(new_bbox[3], placed_bbox[3])

        # Compute the area of overlap
        overlap_area = max(0, xB - xA) * max(0, yB - yA)

        # Compute the area of the new cutout
        new_area = (new_bbox[2] - new_bbox[0]) * (new_bbox[3] - new_bbox[1])

        # Calculate the overlap ratio
        overlap_ratio = overlap_area / new_area

        # Return True if the overlap ratio exceeds the allowed visibility threshold
        return overlap_ratio > (1 - min_visibility)

    @staticmethod
    def calculate_coordinates(
        tx: int, ty: int, iw: int, ih: int, bw: int, bh: int
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate the coordinates for image placement, ensuring it stays within the background bounds.

        Args:
            tx (int): Top-left x-coordinate for placement.
            ty (int): Top-left y-coordinate for placement.
            iw (int): Width of the image.
            ih (int): Height of the image.
            bw (int): Width of the background.
            bh (int): Height of the background.

        Returns:
            Tuple: Calculated coordinates for ROI and image.
        """
        rx_start = max(tx, 0)
        ry_start = max(ty, 0)
        ix_start = max(0, -tx)
        iy_start = max(0, -ty)
        ix_end = iw - max(0, tx + iw - bw)
        iy_end = ih - max(0, ty + ih - bh)
        return rx_start, ry_start, ix_start, iy_start, ix_end, iy_end

    def random_coordinates(
        self, bg_height: int, bg_width: int, img: np.ndarray
    ) -> Tuple[int, int, int, int, int, int]:
        """
        Calculate random placement coordinates for an image on the background.

        Args:
            bg_height (int): Height of the background.
            bg_width (int): Width of the background.
            img (np.ndarray): The image to be placed.

        Returns:
            Tuple: Calculated coordinates for ROI and image.
        """
        img_height, img_width = img.shape[:2]
        center_x = random.randint(0, bg_width)
        center_y = random.randint(0, bg_height)
        top_left_x = center_x - img_width // 2
        top_left_y = center_y - img_height // 2

        return self.calculate_coordinates(
            top_left_x, top_left_y, img_width, img_height, bg_width, bg_height
        )

    def overlay_with_alpha(
        self, background: np.ndarray, background_semantic_mask: np.ndarray,
        background_instance_mask: np.ndarray, image: np.ndarray, x: int, y: int,
        class_id: int, instance_id: int
    ) -> List[List[float]]:
        """
        Overlay an image with alpha transparency onto the background, updating masks with class and instance IDs.
        Cast shadows based on the cutout.

        Args:
            background (np.ndarray): The background image.
            background_semantic_mask (np.ndarray): The semantic mask for the background.
            background_instance_mask (np.ndarray): The instance mask for the background.
            image (np.ndarray): The cutout image to overlay.
            x (int): X-coordinate for overlay placement.
            y (int): Y-coordinate for overlay placement.
            class_id (int): The class ID for the overlay.
            instance_id (int): The instance ID for the overlay.

        Returns:
            List[List[float]]: Normalized coordinates of the overlay.
        """
        img_height, img_width = image.shape[:2]

        # Extract the alpha channel from the cutout image (assume full opacity if no alpha channel exists)
        if image.shape[2] == 4:
            alpha_channel = image[:, :, 3] / 255.0  # Normalize alpha to range [0, 1]
            rgb_image = image[:, :, :3]  # Extract RGB channels
        else:
            # alpha_channel = np.ones((img_height, img_width), dtype=np.float32)
            alpha_channel = image[:, :, -1] > 0  # This is a 2D array (height, width)
            rgb_image = image

        # Define the region of interest (ROI) in the background
        roi_x_start = max(x, 0)
        roi_y_start = max(y, 0)
        roi_x_end = min(x + img_width, background.shape[1])
        roi_y_end = min(y + img_height, background.shape[0])

        # Crop the cutout to match the ROI (in case it goes beyond the background bounds)
        img_x_start = max(0, -x)
        img_y_start = max(0, -y)
        img_x_end = img_width - max(0, (x + img_width) - background.shape[1])
        img_y_end = img_height - max(0, (y + img_height) - background.shape[0])

        
        cutout_rgb = rgb_image[img_y_start:img_y_end, img_x_start:img_x_end]
        cutout_alpha = alpha_channel[img_y_start:img_y_end, img_x_start:img_x_end]
        roi_background = background[roi_y_start:roi_y_end, roi_x_start:roi_x_end]

         # Create the binary mask for the cutout
        cutout_mask = (cutout_alpha > 0).astype(np.uint8)

        # Blend the cutout with the background using the alpha mask
        cutout_alpha_expanded = cutout_alpha[:, :, np.newaxis]  # Expand alpha to match the 3-channel RGB
        background[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = (
            roi_background * (1 - cutout_alpha_expanded) + cutout_rgb * cutout_alpha_expanded
        ).astype(np.uint8)

        # Update the semantic and instance masks in the ROI
        background_semantic_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = np.where(
            cutout_alpha > 0,
            class_id,
            background_semantic_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
        )

        

        if self.cfg.synthesize.instance_masks:
            background_instance_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end] = np.where(
                cutout_alpha > 0,
                instance_id,
                background_instance_mask[roi_y_start:roi_y_end, roi_x_start:roi_x_end]
            )
        
        yolo_bbox = []
        # Calculate YOLO format bounding box if enabled
        if self.create_bbox:
            yolo_bbox = self.calculate_yolo_bbox(roi_x_start, roi_y_start, roi_x_end, roi_y_end, background, class_id)

        # Return normalized coordinates for YOLO labeling
        norm_coords = []
        if self.create_contours:
            norm_coords = normalize_coordinates(
                mask2polygon_holes((cutout_alpha > 0).astype(np.uint8) * 255),
                background.shape[1], background.shape[0], x, y
            )

        return norm_coords, yolo_bbox

    def calculate_yolo_bbox(self, roi_x_start: int, roi_y_start: int, roi_x_end: int, roi_y_end: int, background: np.ndarray, class_id: int) -> List[float]:
            """
            Calculate YOLO format bounding box (center_x, center_y, width, height) using the ROI.

            Args:
                roi_x_start (int): The starting x-coordinate of the ROI.
                roi_y_start (int): The starting y-coordinate of the ROI.
                roi_x_end (int): The ending x-coordinate of the ROI.
                roi_y_end (int): The ending y-coordinate of the ROI.
                background (np.ndarray): The background image.
                class_id (int): The class ID for the bounding box.

            Returns:
                List[float]: The YOLO format bounding box.
            """
            bbox_x_start = roi_x_start / background.shape[1]
            bbox_y_start = roi_y_start / background.shape[0]
            bbox_width = (roi_x_end - roi_x_start) / background.shape[1]
            bbox_height = (roi_y_end - roi_y_start) / background.shape[0]

            center_x = bbox_x_start + bbox_width / 2
            center_y = bbox_y_start + bbox_height / 2

            return [class_id, center_x, center_y, bbox_width, bbox_height]

class ImageCompositor:
    """
    A class dedicated to managing the composition of images based on specified configurations.
    This class orchestrates the process of loading cutouts, placing them on a background,
    and saving the resultant images and masks.
    """

    def __init__(self, cfg: DictConfig, recipe: Dict):
        """
        Initialize the image compositor with the given recipe.

        Args:
            recipe (Dict): Dictionary containing the background path and cutout metadata.
        """
        self.cfg = cfg
        self.recipe = recipe  # Store the recipe as an attribute for access in other methods
        self.background_path = Path(cfg.paths.backgrounddir, recipe['background_image_id'])
        self.cutout_metadata = recipe['cutouts']
        self.cutout_paths = [Path(cfg.paths.cutoutdir, cutout['cutout_id'] + ".png") for cutout in self.cutout_metadata]
        self.processor = ImageProcessor(cfg, num_cutouts=len(self.cutout_paths))
        self.config_save_dirs(cfg)

    def config_save_dirs(self, cfg: DictConfig) -> None:
        """
        Configure and create necessary directories for saving results.
        """
        self.savedir = Path(cfg.paths.projectdir, "results")
        self.image_savedir = Path(self.savedir, "images")
        self.semantic_savedir = Path(self.savedir, "semantic_masks")
        self.instance_savedir = Path(self.savedir, "instance_masks")
        self.yolo_cont_label_savedir = Path(self.savedir, "yolo_contour_labels")
        self.yolo_bbox_label_savedir = Path(self.savedir, "yolo_bbox_labels")

        for directory in [
            self.image_savedir, self.semantic_savedir,
            self.instance_savedir, self.yolo_cont_label_savedir, 
            self.yolo_bbox_label_savedir
        ]:
            directory.mkdir(exist_ok=True, parents=True)

    def save_bboxes(self, txtpath: Path, yolo_bboxes: List[List[Union[str, float]]]) -> None:
        """
        Save the YOLO formatted bounding boxes to a text file.

        Args:
            txtpath (Path): Path to the output text file.
            yolo_bboxes (List[List[Union[str, float]]]): List of YOLO formatted bounding boxes.
        """
        with open(txtpath, "w") as file:
            for bbox in yolo_bboxes:
                line = " ".join(str(num) for num in bbox)
                file.write(line + "\n")

    def save_contour(self, txtpath: Path, coord_results: List[Dict[int, List[List[float]]]]) -> None:
        """
        Save the normalized coordinates for object detection in YOLO format.

        Args:
            txtpath (Path): Path to the output text file.
            coord_results (List[Dict[int, List[List[float]]]]): List of coordinates for each object in the image.
        """
        with open(txtpath, "w") as file:
            for coord_dict in coord_results:
                for class_id, norm_coords in coord_dict.items():
                    for cnt in norm_coords:
                        line = " ".join(str(num) for num in cnt)
                        file.write(f"{class_id} " + line + "\n")

    def save_data(
        self, image: np.ndarray, semantic_mask: np.ndarray,
        instance_mask: np.ndarray, coord_results: List[Dict[int, List[List[float]]]],
        yolo_bboxes: List[List[Union[str, float]]],
        synthetic_image_id: str  # Add synthetic_image_id as an argument
    ) -> None:
        """
        Save the composed image, masks, and YOLO label using the synthetic_image_id.

        Args:
            image (np.ndarray): The composed image.
            semantic_mask (np.ndarray): The semantic mask of the composed image.
            instance_mask (np.ndarray): The instance mask of the composed image.
            coord_results (List[Dict[int, List[List[float]]]]): Coordinates of the placed cutouts.
            yolo_bboxes (List[List[Union[str, float]]]): Yolo formatted bounding boxes of the placed cutouts.
            synthetic_image_id (str): Unique identifier for the synthetic image.
        """
        # Use synthetic_image_id for unique file naming
        imagesavepath = Path(self.image_savedir, f"{synthetic_image_id}.jpg")
        semanticsavepath = Path(self.semantic_savedir, f"{synthetic_image_id}.png")
        
        cv2.imwrite(str(imagesavepath), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(semanticsavepath), semantic_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        
        if self.cfg.synthesize.instance_masks:
            instancesavepath = Path(self.instance_savedir, f"{synthetic_image_id}.png")
            cv2.imwrite(str(instancesavepath), instance_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        if self.cfg.synthesize.yolo_contour_labels:
            yololabelpath = Path(self.yolo_cont_label_savedir, f"{synthetic_image_id}.txt")
            self.save_contour(yololabelpath, coord_results)

        if self.cfg.synthesize.yolo_bbox_labels:
            yolo_bbox_labelpath = Path(self.yolo_bbox_label_savedir, f"{synthetic_image_id}.txt")
            self.save_bboxes(yolo_bbox_labelpath, yolo_bboxes)

        log.info(f"Image processed and saved successfully as {synthetic_image_id}.")

def resize_image(img: np.ndarray, resize_scale: float) -> np.ndarray:
    """
    Resize the cutout proportionally based on the specified resize scale.

    Args:
        img (np.ndarray): The cutout image to resize.
        resize_scale (float): The scaling factor for resizing.

    Returns:
        np.ndarray: The resized cutout image.
    """
    height, width = img.shape[:2]
    new_size = (int(width * resize_scale), int(height * resize_scale))
    resized_img = cv2.resize(img, new_size, interpolation=cv2.INTER_LINEAR)
    return resized_img
    
# def process_recipe(cfg: DictConfig, json_file: Path) -> None:
def process_recipe(cfg: DictConfig, recipe: Dict, shared_data: Dict) -> None:
    """
    Process a single recipe for synthetic image generation.

    Args:
        cfg (DictConfig): Configuration object.
        recipe (Dict): Recipe containing metadata for synthetic image generation.
        shared_data (Dict): Shared dictionary for pre-loaded cutouts and backgrounds.
    """
    try:
        # Extract background path
        background_path = Path(cfg.paths.backgrounddir, recipe['background_image_id'])
        
        resize_factor = cfg.synthesize.resize_factor # Added resize factor

        # Check if the background is already loaded in shared_data
        if background_path not in shared_data:
            log.info(f"Loading background image {background_path}")
            background_image = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
            if resize_factor != 1.0:
                background_image = resize_image(background_image, resize_factor)
            shared_data[background_path] = background_image
        
        background = shared_data[background_path]  # Get the pre-loaded background image
        
        # Process the cutouts and check if they are in shared_data
        cutout_paths = [Path(cfg.paths.cutoutdir, cutout['cutout_id'] + ".png") for cutout in recipe['cutouts']]
        images = []
        for cutout_path in cutout_paths:
            if cutout_path not in shared_data:
                log.debug(f"Loading cutout image {cutout_path}")
                img = cv2.imread(str(cutout_path), cv2.IMREAD_UNCHANGED)
                # Resize the image based on the resize scale
                if resize_factor != 1.0:
                    img = resize_image(img, resize_factor)
                if img.shape[2] == 4:
                    img = img[:, :, :3]  # Ensure image has three channels if alpha is not needed

                shared_data[cutout_path] = img
            images.append(shared_data[cutout_path])
        
        # Initialize the processor with the number of cutouts
        processor = ImageProcessor(cfg, num_cutouts=len(images))
        
        # Distribute the cutout images on the background
        result, result_semantic_mask, result_instance_mask, coord_results, yolo_bboxes = processor.distribute_images(
            background, images, recipe['cutouts'], mode="random"
        )
        
        # Save the results
        compositor = ImageCompositor(cfg, recipe)
        compositor.save_data(result, result_semantic_mask, result_instance_mask, coord_results, yolo_bboxes, recipe['synthetic_image_id'])
        log.info(f"Synthetic image {recipe['synthetic_image_id']} processed successfully.")
    
    except Exception as exc:
        log.exception(f"Failed to process synthetic image {recipe['synthetic_image_id']}: {exc}")

def main(cfg: DictConfig) -> None:
    log.info("Starting synthetic image generation.")
    json_recipe_path = Path(cfg.paths.projectdir,"recipes", f"{cfg.general.project_name}_{cfg.general.sub_project_name}.json")
    # Load the JSON once and share the data between processes
    with open(json_recipe_path, 'r') as file:
        data = json.load(file)

    synthetic_images = data.get("synthetic_images", [])
    log.info(f"Processing {len(synthetic_images)} synthetic image")
    if len(synthetic_images) == 0:
        log.error("No synthetic images found in recipe.")
        return
             
    
    # Use multiprocessing.Manager to share data like cutouts across processes
    with Manager() as manager:
        shared_data = manager.dict()  # Store shared cutout/background images here
        
        if cfg.synthesize.parallel:
            max_workers = cfg.synthesize.parallel_workers  # Dynamic worker count
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(process_recipe, cfg, recipe, shared_data)
                    for recipe in synthetic_images
                ]

                for future in as_completed(futures):
                    try:
                        future.result()
                        log.info("Recipe processed successfully.")
                    except Exception as exc:
                        log.exception(f"Recipe processing failed: {exc}")
        else:
            # Sequential processing for debugging
            for recipe in synthetic_images:
                try:
                    process_recipe(cfg, recipe, shared_data)
                    log.info(f"Processed recipe {recipe['synthetic_image_id']}")
                except Exception as exc:
                    log.exception(f"Failed to process recipe: {exc}")