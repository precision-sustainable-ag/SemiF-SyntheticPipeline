"""
This script is designed to process large batches of image data for synthetic image generation with a focus on CPU/GPU parallelism.
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

    def __init__(self,cfg: DictConfig, num_cutouts: int):
        """
        Initialize the image processor with the number of cutouts.

        Args:
            num_cutouts (int): Number of cutouts to process.
        """
        self.cfg = cfg
        self.num_cutouts = num_cutouts
        self.resize_scale = cfg.synthesize.resize_factor  # Added scaling factor
        self.instance_ids = self.num_instances()

        # Calculate the adjusted scale_limit based on resize_scale
        self.scale_limit = self.calculate_dynamic_scale_limit()
        # TODO apply more/sophisticated albumentations transformations (e.g shadows, blur, etc.)
        # TODO set transformations based on config settings
        # Define the Albumentations transformations
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
    
    def calculate_dynamic_scale_limit(self) -> Tuple[float, float]:
        """
        Calculate a dynamic scale limit for Albumentations RandomScale transformation
        based on the resize_scale value. This ensures the RandomScale does not shrink
        the cutout images excessively.

        Returns:
            Tuple[float, float]: The lower and upper scale limits for RandomScale.
        """
        # If the resize_scale is too small, limit the shrinking factor.
        min_scale_limit = max(-0.5, -1.0 + self.resize_scale)  # Minimum allowed scaling
        # TODO incorporate this into the config settings
        max_scale_limit = 0.0  # Keep the upper limit at 0.0 (no further enlargement) 

        log.debug(f"Calculated dynamic scale limits: {min_scale_limit}, {max_scale_limit}")
        return (min_scale_limit, max_scale_limit)
    
    def num_instances(self) -> List[int]:
        """
        Generate a shuffled list of possible instance IDs based on the number of cutouts.

        Returns:
            List[int]: A list of shuffled instance IDs.
        """
        num_possible_instances = list(range(1, self.num_cutouts + 1))
        random.shuffle(num_possible_instances)
        return num_possible_instances

    def create_shadow_from_cutout(self, background: np.ndarray, mask: np.ndarray, cutout_position: Tuple[int, int], shadow_offset: Tuple[int, int], padding_ratio: float = 0.1, shadow_intensity: float = 0.5) -> np.ndarray:
        """
        Create a shadow based on the cutout mask and apply it to the background.
        The shadow will be padded dynamically based on the size of the cutout, considering the position of the cutout.

        Args:
            background (np.ndarray): The background image where the shadow will be applied.
            mask (np.ndarray): The binary mask of the cutout (1 where cutout is present, 0 elsewhere).
            cutout_position (Tuple[int, int]): The (x, y) position of the cutout on the background.
            shadow_offset (Tuple[int, int]): The (x, y) offset for the shadow direction.
            padding_ratio (float): The fraction of the cutout size to use for padding (e.g., 0.1 for 10% of cutout size).
            shadow_intensity (float): The intensity of the shadow, between 0 (no shadow) and 1 (completely black).

        Returns:
            np.ndarray: The background with the shadow applied.
        """
        img_height, img_width = background.shape[:2]
        mask_height, mask_width = mask.shape[:2]
        cutout_x, cutout_y = cutout_position

        # Calculate dynamic padding based on the size of the cutout
        padding_x = int(mask_width * padding_ratio)
        padding_y = int(mask_height * padding_ratio)

        # Calculate the boundaries for the padded region, ensuring they stay within the background
        x_start = max(cutout_x - padding_x, 0)
        y_start = max(cutout_y - padding_y, 0)
        x_end = min(cutout_x + mask_width + padding_x, img_width)
        y_end = min(cutout_y + mask_height + padding_y, img_height)

        # Create a new mask with padding added, ensuring it fits within the background
        padded_mask = np.zeros((y_end - y_start, x_end - x_start), dtype=np.uint8)

        # Place the original mask inside the padded mask, ensuring no out-of-bounds issues
        mask_x_start = max(padding_x - cutout_x, 0)
        mask_y_start = max(padding_y - cutout_y, 0)

        # Calculate how much of the mask can fit into the padded area
        valid_mask_height = min(mask_height, padded_mask.shape[0] - mask_y_start)
        valid_mask_width = min(mask_width, padded_mask.shape[1] - mask_x_start)

        # Copy the valid portion of the mask into the padded mask
        padded_mask[mask_y_start:mask_y_start + valid_mask_height, mask_x_start:mask_x_start + valid_mask_width] = mask[:valid_mask_height, :valid_mask_width]

        # Roll the expanded mask to simulate the shadow offset
        shadow_mask = np.roll(padded_mask, shift=shadow_offset, axis=(0, 1))

        # Apply Gaussian blur to soften the shadow edges
        shadow_mask_blurred = cv2.GaussianBlur(shadow_mask.astype(np.float32), (31, 31), 0)

        # Multiply the shadow mask by the shadow intensity to control darkness
        shadow_mask_blurred = shadow_mask_blurred * shadow_intensity

        # Ensure shadow_mask_blurred has the correct number of channels for RGB application
        shadow_mask_expanded = shadow_mask_blurred[:, :, np.newaxis]

        # Apply the shadow by darkening the region of the background
        background[y_start:y_end, x_start:x_end] = background[y_start:y_end, x_start:x_end] * (1 - shadow_mask_expanded)

        return background
    
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
        background_instance_mask = np.zeros((bg_height, bg_width), dtype=np.uint16)

        coord_results = []
        instance_id = 1
        for _, (img, cutout_metadata) in enumerate(zip(images, cutout_paths)):
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
            coord_results.append({class_id: norm_coords})
            
            instance_id += 1

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
        # Apply shadow before placing the cutout
        # TODO randomize shadow intensity and direction(offset) on image (not cutout) level
        background = self.create_shadow_from_cutout(
            background, cutout_mask, cutout_position=(x, y), 
            shadow_offset=(20, 20), padding_ratio=0.9, shadow_intensity=0.5
        )

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

        # Return normalized coordinates for YOLO labeling
        norm_coords = normalize_coordinates(
            mask2polygon_holes((cutout_alpha > 0).astype(np.uint8) * 255),
            background.shape[1], background.shape[0], x, y
        )

        return norm_coords
    def overlay_with_alpha_old(
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

        expanded_mask = image_mask[:, :, np.newaxis]  # Expand dimensions for RGB operations

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
        
        norm_coords = None
        
        if self.cfg.synthesize.instance_masks:
            background_instance_mask[y: y + img_height, x: x + img_width] = np.where(
            image_mask,  # Use original 2D mask for single channel mask update
            instance_id,  # Unique instance ID
            background_instance_mask[y: y + img_height, x: x + img_width],  # Existing mask values
        )
            log.debug("Creating contours for the overlay.")
            contours = mask2polygon_holes(image_mask.astype(np.uint8) * 255)
            background_height, background_width = background.shape[:2]
            norm_coords = normalize_coordinates(
                contours, background_width, background_height, x, y
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
        self.yololabel_savedir = Path(self.savedir, "yolo_labels")

        for directory in [
            self.image_savedir, self.semantic_savedir,
            self.instance_savedir, self.yololabel_savedir
        ]:
            directory.mkdir(exist_ok=True, parents=True)

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
        synthetic_image_id: str  # Add synthetic_image_id as an argument
    ) -> None:
        """
        Save the composed image, masks, and YOLO label using the synthetic_image_id.

        Args:
            image (np.ndarray): The composed image.
            semantic_mask (np.ndarray): The semantic mask of the composed image.
            instance_mask (np.ndarray): The instance mask of the composed image.
            coord_results (List[Dict[int, List[List[float]]]]): Coordinates of the placed cutouts.
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
        if self.cfg.synthesize.yolo_labels:
            yololabelpath = Path(self.yololabel_savedir, f"{synthetic_image_id}.txt")
            self.save_contour(yololabelpath, coord_results)

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
        result, result_semantic_mask, result_instance_mask, coord_results = processor.distribute_images(
            background, images, recipe['cutouts'], mode="random"
        )
        
        # Save the results
        compositor = ImageCompositor(cfg, recipe)
        compositor.save_data(result, result_semantic_mask, result_instance_mask, coord_results, recipe['synthetic_image_id'])
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