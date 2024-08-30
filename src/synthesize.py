import json
import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple, Dict

import albumentations as A
import cv2
import numpy as np
from omegaconf import DictConfig

from utils.utils import mask2polygon_holes, normalize_coordinates

log = logging.getLogger(__name__)

class ImageProcessor:
    """
    A class to handle image processing tasks such as loading images, applying transformations, 
    and overlaying them onto a background.
    """

    def __init__(self, num_cutouts: int):
        """
        Initialize the image processor with the number of cutouts.

        Args:
            num_cutouts (int): Number of cutouts to process.
        """
        self.num_cutouts = num_cutouts
        self.instance_ids = self.num_instances()

        # Define the Albumentations transformations
        self.transform = A.Compose([
            A.GaussNoise(p=0.2), A.HorizontalFlip(p=0.2), A.VerticalFlip(p=0.2),
            A.RandomRotate90(p=0.5), A.RandomBrightnessContrast(p=0.2),
            A.RandomScale(p=0.75, scale_limit=(-0.7, 0), always_apply=True),
        ])

    def num_instances(self) -> List[int]:
        """
        Generate a shuffled list of possible instance IDs based on the number of cutouts.

        Returns:
            List[int]: A list of shuffled instance IDs.
        """
        num_possible_instances = list(range(1, self.num_cutouts + 1))
        random.shuffle(num_possible_instances)
        return num_possible_instances

    def load_image(self, image_path: str, with_alpha: bool = True) -> np.ndarray:
        """
        Load an image from the specified path with an option to include the alpha channel.

        Args:
            image_path (str): Path to the image file.
            with_alpha (bool): Whether to load the alpha channel. Defaults to True.

        Returns:
            np.ndarray: The loaded image.
        """
        flag = cv2.IMREAD_UNCHANGED if with_alpha else cv2.IMREAD_COLOR
        print(str(image_path))
        img = cv2.imread(str(image_path), flag)
        
        if with_alpha and img.shape[2] == 4:
            img = img[:, :, :3]  # Ensure image has three channels if alpha is not needed

        log.debug(f"Loaded image {image_path} with shape {img.shape}")
        return img

    def apply_random_transform(self, img: np.ndarray) -> np.ndarray:
        """
        Apply a unique random Albumentations transform to the image, handling transparency.

        Args:
            img (np.ndarray): The image to be transformed.

        Returns:
            np.ndarray: The transformed image.
        """
        augmented = self.transform(image=img, mask=(img[:, :, -1] > 0).astype(np.uint8))
        img_transformed = augmented['image']
        transformed_mask = augmented['mask']
        combined_image = np.where(transformed_mask[..., None] == 1, img_transformed, 0)

        return combined_image

    def distribute_images(
        self, background: np.ndarray, images: List[np.ndarray],
        cutout_paths: List[str], mode: str = "random"
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
        # TODO: Make this a 16-bit mask for more than 255 instances
        background_instance_mask = np.zeros((bg_height, bg_width), dtype=np.uint16)

        coord_results = []
        instance_id = 1
        for index, (img, cutout_metadata) in enumerate(zip(images, cutout_paths)):
            class_id = cutout_metadata['category']['class_id']

            img = self.apply_random_transform(img)

            if mode == "random":
                roi_x_start, roi_y_start, img_x_start, img_y_start, img_x_end, img_y_end = (
                    self.random_coordinates(bg_height, bg_width, img)
                )
            else:
                raise ValueError(
                    "Unsupported cutout distribution mode. Use 'random'."
                )

            norm_coords = self.overlay_with_alpha(
                background, background_semantic_mask, background_instance_mask,
                img[img_y_start:img_y_end, img_x_start:img_x_end],
                roi_x_start, roi_y_start, class_id, instance_id
            )
            instance_id += 1
            coord_results.append({class_id: norm_coords})

        return background, background_semantic_mask, background_instance_mask, coord_results

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

        Args:
            background (np.ndarray): The background image.
            background_semantic_mask (np.ndarray): The semantic mask for the background.
            background_instance_mask (np.ndarray): The instance mask for the background.
            image (np.ndarray): The image to overlay.
            x (int): X-coordinate for overlay placement.
            y (int): Y-coordinate for overlay placement.
            class_id (int): The class ID for the overlay.
            instance_id (int): The instance ID for the overlay.

        Returns:
            List[List[float]]: Normalized coordinates of the overlay.
        """
        img_height, img_width = image.shape[:2]
        image_mask = image[:, :, -1] > 0  # This is a 2D array (height, width)

        log.debug("Creating contours for the overlay.")
        contours = mask2polygon_holes(image_mask.astype(np.uint8) * 255)
        background_height, background_width = background.shape[:2]
        norm_coords = normalize_coordinates(
            contours, background_width, background_height, x, y
        )

        expanded_mask = image_mask[:, :, np.newaxis]  # Expand dimensions for RGB operations

        log.debug("Overlaying image on the background.")
        background[y: y + img_height, x: x + img_width] = np.where(
            expanded_mask,  # Use expanded mask for RGB channels
            image[:, :, :3],  # RGB values of the image
            background[y: y + img_height, x: x + img_width],  # Existing background
        )

        background_semantic_mask[y: y + img_height, x: x + img_width] = np.where(
            image_mask,  # Use original 2D mask for single channel mask update
            class_id,  # Class ID for this overlay
            background_semantic_mask[y: y + img_height, x: x + img_width],  # Existing mask values
        )

        background_instance_mask[y: y + img_height, x: x + img_width] = np.where(
            image_mask,  # Use original 2D mask for single channel mask update
            instance_id,  # Unique instance ID
            background_instance_mask[y: y + img_height, x: x + img_width],  # Existing mask values
        )

        return norm_coords


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

        self.background_path = Path(cfg.paths.backgrounddir, recipe['background_image_id'])
        self.cutout_metadata = recipe['cutouts']
        self.cutout_paths = [Path(cfg.paths.cutoutdir, cutout['cutout_id'] + ".png") for cutout in self.cutout_metadata]
        self.processor = ImageProcessor(num_cutouts=len(self.cutout_paths))
        self.config_save_dirs(cfg)

    def config_save_dirs(self, cfg: DictConfig) -> None:
        """
        Configure and create necessary directories for saving results.
        """
        self.savedir = cfg.paths.resultsdir
        self.image_savedir = Path(self.savedir, "images")
        self.semantic_savedir = Path(self.savedir, "semantic_masks")
        self.instance_savedir = Path(self.savedir, "instance_masks")
        self.yololabel_savedir = Path(self.savedir, "yolo_labels")

        for directory in [
            self.image_savedir, self.semantic_savedir,
            self.instance_savedir, self.yololabel_savedir
        ]:
            directory.mkdir(exist_ok=True, parents=True)

    def run(self) -> None:
        """
        Execute the image composition process, saving the results.
        """
        background = self.processor.load_image(self.background_path, with_alpha=False)
        log.info("Loading cutout images...")
        images = [
            self.processor.load_image(cutout, with_alpha=True)
            for cutout in self.cutout_paths
        ]
        log.info("Distributing images...")
        result, result_semantic_mask, result_instance_mask, coord_results = (
            self.processor.distribute_images(
                background, images, self.cutout_metadata, mode="random"
            )
        )

        self.save_data(result, result_semantic_mask, result_instance_mask, coord_results)

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
        instance_mask: np.ndarray, coord_results: List[Dict[int, List[List[float]]]]
    ) -> None:
        """
        Save the composed image, masks, and YOLO label with a timestamp to ensure uniqueness.

        Args:
            image (np.ndarray): The composed image.
            semantic_mask (np.ndarray): The semantic mask of the composed image.
            instance_mask (np.ndarray): The instance mask of the composed image.
            coord_results (List[Dict[int, List[List[float]]]]): Coordinates of the placed cutouts.
        """
        timestamp = int(time.time())  # Get current Unix time in seconds
        imagesavepath = Path(self.image_savedir, f"{timestamp}.jpg")
        semanticsavepath = Path(self.semantic_savedir, f"{timestamp}.png")
        instancesavepath = Path(self.instance_savedir, f"{timestamp}.png")
        yololabelpath = Path(self.yololabel_savedir, f"{timestamp}.txt")

        cv2.imwrite(str(imagesavepath), image, [cv2.IMWRITE_JPEG_QUALITY, 100])
        cv2.imwrite(str(semanticsavepath), semantic_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        cv2.imwrite(str(instancesavepath), instance_mask, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        self.save_contour(yololabelpath, coord_results)

        log.info(f"Image processed and saved successfully.")


def process_recipe(cfg: DictConfig, json_file: Path) -> None:
    with open(json_file, 'r') as file:
        recipe = json.load(file)

    compositor = ImageCompositor(cfg, recipe)
    compositor.run()

def main(cfg: DictConfig) -> None:
    log.info("Starting synthetic image generation.")
    json_files = list(Path(cfg.paths.synthetic_metadata).glob("*.json"))
    
    if cfg.synthesize.parallel:
        max_workers = cfg.synthesize.max_workers
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(process_recipe, cfg, json_file) for json_file in json_files]

            for future in as_completed(futures):
                try:
                    future.result()
                    log.info("Recipe processed successfully.")
                except Exception as exc:
                    log.exception(f"Recipe processing failed: {exc}")
    else:
        for json_file in json_files:
            try:
                process_recipe(cfg, json_file)
                log.info("Recipe processed successfully.")
            except Exception as exc:
                log.exception(f"Recipe processing failed: {exc}")