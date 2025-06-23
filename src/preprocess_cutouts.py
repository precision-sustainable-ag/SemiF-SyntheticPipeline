import os
import cv2
import sqlite3
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig
from utils.utils import query_for_cutout_metadata


log = logging.getLogger(__name__)

class CutoutProcessor():
    def __init__(self, cfg) -> None:

        self.cfg = cfg
        db_path = str(cfg.paths.sql_database)

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cursor = conn.cursor()
        self.cursor.execute("PRAGMA table_info(semif_cutouts);")

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
            self.exg_list = [species.upper() for species in self.exg_list]

            # apply exg filtering
            self.apply_exg()
        else:
            log.info("Empty exg list. Skipping exg.")
    
    def apply_exg(self) -> None:
        for cutout in self.all_cutouts:
            species = query_for_cutout_metadata(cutout, self.cursor)

            # See if species is submitted for exg filtering, if so apply and save to preprocess cutout path
            if species in self.exg_list:
                orignal_path = f"{self.cutout_path}/{cutout}.png"
                exg_path = f"{self.pre_processed_cutout_path}/{cutout}.png"
                apply_exg_mask(orignal_path, exg_path)

def apply_exg_mask(input_path: str, output_path: str, exg_threshold: int = 20) -> None:
    """ 
        The exg will set all non green areas black to ensure it works
        with the synthesize script. Synthesize will remove black background
    """ 
    # Load image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    # Convert to float32 for ExG calculation
    img_float = img[:, :, :3].astype(np.float32)

    # Split channels (OpenCV uses BGR)
    B, G, R = cv2.split(img_float)

    # Compute Excess Green Index: ExG = 2G - R - B
    exg = 2 * G - R - B

    # Threshold: keep green areas, set others to black
    mask = exg > exg_threshold

    # Create output image: all black
    out_img = np.zeros_like(img)

    # Copy original color where ExG is high enough
    out_img[mask] = img[mask]
    cv2.imwrite(output_path, out_img)

def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    CutoutProcessor(cfg)