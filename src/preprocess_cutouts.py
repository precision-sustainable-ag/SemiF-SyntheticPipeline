import cv2
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig


from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import Manager
from typing import List, Tuple, Dict, Union

import albumentations as A

# from utils.utils import mask2polygon_holes, normalize_coordinates

log = logging.getLogger(__name__)


class CutoutManipulator():
    def __init__(cfg):

        self.cutout_path = cfg.paths.cutoutdir
    
    def resize_cutouts():
        if cutout_metadata['category']['common_name'].lower() in resize_area:
            pixel_resize = resize_area[cutout_metadata['category']['common_name'].lower()]
            print(pixel_resize)
            img = resize_to_target_pixels(img, pixel_resize)   

    def apply_exg():
        pass 
    
    def resize_to_target_pixels(image, target_pixels):
        h, w = image.shape[:2]
        original_pixels = w * h

        # Calculate scale factor to get close to target pixel count
        scale_factor = (target_pixels / original_pixels) ** 0.5
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))

        resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized_image




def main(cfg: DictConfig, resize_area) -> None:
    CutoutManipulator(cfg)