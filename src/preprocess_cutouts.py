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

        if cfg.cutout_filters.scale:
            # Load in resizing dictionary and make keys (common_name) lowercase
            self.resizing_dictionary = cfg.cutout_filters.scale
            self.resizing_dictionary = {k.lower(): v for k, v in self.resizing_dictionary.items()}

            # tasks
            self.scale_cutouts()
        else:
            log.info("Empty resizing dictionary. Skipping resizing.")
    
    def scale_cutouts(self):
        for cutout in self.all_cutouts:
            species = query_for_cutout_metadata(cutout, self.cfg).lower()
            if species in self.resizing_dictionary:
                # Load image
                img = cv2.imread(f"{self.cutout_path}/{cutout}.png")

                # Grab scale factor and pass it into scale to target pixels
                scale_factor = self.resizing_dictionary[species]
                img = self.scale_to_target_pixels(img, scale_factor)   

                cv2.imwrite(f"{self.pre_processed_cutout_path}/{cutout}.png", img)

    def apply_exg(self):
        pass 
    
    def scale_to_target_pixels(self, image, scale_factor):
        h, w = image.shape[:2]
        original_pixels = w * h

        # Calculate scale factor to get close to target pixel count
        new_w = max(1, int(w * scale_factor))
        new_h = max(1, int(h * scale_factor))

        scaled_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return scaled_image

def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    if cfg.cutout_filters.cutout_preprocessing:
        CutoutManipulator(cfg)
    else:
        log.warning("Specified preprocess_cutouts in pipeline but configed to false in cutout_filters")
        log.warning("Skippning this step")