import os
import cv2
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from utils.utils import query_for_cutout_metadata


log = logging.getLogger(__name__)

class CutoutManipulator():
    def __init__(self, cfg):

        self.cfg = cfg

        self.pre_processed_cutout_path = cfg.paths.preprocessed_cutoutdir

        # Grab cutout ids of the cutouts that were downloaded
        self.cutout_path = cfg.paths.cutoutdir
        self.all_cutouts = [f.removesuffix(".png") for f in os.listdir(self.cutout_path) if os.path.isfile(os.path.join(self.cutout_path, f))]

        # Create directory to store preprocess cutouts if it dosent exist
        os.makedirs(str(cfg.paths.preprocessed_cutoutdir), exist_ok=True)

        # Go through tasks
        if cfg.preprocess_cutouts.exg:
            # Load in exg list and make species (common_name) lowercase
            self.exg_list = cfg.preprocess_cutouts.exg
            self.exg_list = [species.lower() for species in self.exg_list]

            # apply exg filtering
            self.apply_exg()
        else:
            log.info("Empty exg list. Skipping exg.")
    

    def apply_exg(self):
        for cutout in self.all_cutouts:
            species = query_for_cutout_metadata(cutout, self.cfg).lower()

            # See if species is submitted for exg filtering, if so apply and save to preprocess cutout path
            if species in self.exg_list:
                img = cv2.imread(f"{self.cutout_path}/{cutout}.png")
                img = ensure_4_channels(img)
                mask = remove_soil_background_exg(img, method='otsu')
                output = apply_exg_mask_rgba(img, mask)
                cv2.imwrite(f"{self.pre_processed_cutout_path}/{cutout}.png", output)


def remove_soil_background_exg(cutout_img: np.ndarray, method: str = 'otsu', fixed_thresh: int = 20) -> np.ndarray:
    """
    Remove soil background using Excess Green (ExG) index and thresholding.

    Args:
        cutout_img (np.ndarray): Input image (BGR, uint8).
        method (str): 'otsu' or 'fixed' thresholding.
        fixed_thresh (int): Threshold value if method='fixed'.

    Returns:
        np.ndarray: Binary mask with plant as foreground (255) and soil as background (0).
    """
    # Convert BGR to float for calculation
    b, g, r, _ = cv2.split(cutout_img.astype('float32'))

    # Compute Excess Green Index (ExG)
    exg = 2 * g - r - b

    # Normalize ExG to [0,255] for thresholding
    exg_norm = cv2.normalize(exg, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX)
    exg_norm = exg_norm.astype(np.uint8)

    # Thresholding
    if method == 'otsu':
        _, mask = cv2.threshold(exg_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method == 'fixed':
        _, mask = cv2.threshold(exg_norm, fixed_thresh, 255, cv2.THRESH_BINARY)
    else:
        raise ValueError("method must be 'otsu' or 'fixed'.")

    return mask

def apply_exg_mask_rgba(cutout_img: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Apply a binary mask to a 4-channel RGBA image to mask out soil background.

        Args:
            cutout_img (np.ndarray): Input image with shape (H, W, 4), dtype uint8.
            mask (np.ndarray): Binary mask with shape (H, W), values 0 (background) or 255 (foreground).

        Returns:
            np.ndarray: Masked RGBA image where background is transparent.
        """
        assert cutout_img.shape[2] == 4, "Input image must have 4 channels (RGBA)."
        assert cutout_img.shape[:2] == mask.shape, "Mask shape must match image spatial dimensions."

        # Copy image to avoid modifying original
        masked_img = cutout_img.copy()

        # Apply mask: set alpha to 0 where mask == 0, keep alpha as-is where mask == 255
        masked_img[:, :, 3] = np.where(mask == 255, masked_img[:, :, 3], 0)

        return masked_img

def ensure_4_channels(img):
    if img.shape[2] == 4:
        # Already 4 channels, do nothing
        return img
    elif img.shape[2] == 3:
        # Add fully opaque alpha channel
        alpha_channel = np.ones((img.shape[0], img.shape[1]), dtype=img.dtype) * 255
        img_4ch = cv2.merge((img, alpha_channel))
        return img_4ch
    else:
        raise ValueError(f"Unexpected number of channels: {img.shape[2]}")

def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    CutoutManipulator(cfg)