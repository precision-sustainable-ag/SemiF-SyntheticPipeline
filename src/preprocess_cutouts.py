import os
import cv2
import sqlite3
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig

from utils.utils import query_for_cutout_metadata
from utils.pdf import add_grammar_and_capitlization_to_list

log = logging.getLogger(__name__)

class CutoutProcessor():
    def __init__(self, cfg) -> None:

        self.cfg = cfg
        self.preprocess_cutouts = cfg.preprocess_cutouts

        self.common_names = [
            name.upper() for name in self.cfg.cutout_filters.category.common_name
        ]
        db_path = str(cfg.paths.sql_database)

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        self.cursor = conn.cursor()
        self.cursor.execute("PRAGMA table_info(semif_cutouts);")

        # Grab cutout ids of the cutouts that were downloaded
        self.cutout_path = cfg.paths.cutoutdir
        self.all_cutouts = [f.removesuffix(".png") for f in os.listdir(self.cutout_path) if os.path.isfile(os.path.join(self.cutout_path, f))]

        # Create a dictionary; key: cutout_id, value: tuple (image, [processes])
        # Allows us to keep track of the images as they go through
        # possibly multiple processes/filters
        cutout_image_dictionary = {}
        cutout_image_dictionary = self.load_cutout_id_image_dictionary()
        
        log.info(f"Preprocessing requested for {len(cutout_image_dictionary)} cutouts")
        # Loop through cutout_image_dictionary with a progress bar
        for cutout_id in tqdm(cutout_image_dictionary.keys(), desc="Preprocessing cutouts"):
            # extract images and list of preprocesses specified for the cutouts species
            image, preprocesses_and_params = cutout_image_dictionary[cutout_id]

            # perform the preprocessing
            for process_name, param in preprocesses_and_params:
                method = getattr(self, process_name, None)
                if method:
                    image = method(image, param)
                else:
                    raise ValueError(f"Unknown process: {process_name}")

            # save the preprocessed image
            self.save_image(str(f"{self.cutout_path}/{cutout_id}.png"), image)

    def remove_soil(self, image: np.ndarray, exg_threshold_percent: float = 20) -> np.ndarray:
        # Convert to float32 for ExG calculation
        img_float = image[:, :, :3].astype(np.float32)

        # Split channels (OpenCV uses BGR)
        B, G, R = cv2.split(img_float)

        # Compute Excess Green Index: ExG = 2G - R - B
        exg = 2 * G - R - B

        # Normalize ExG to percentage [0, 100]
        exg_min = -510
        exg_max = 510
        exg_percentage = 100 * (exg - exg_min) / (exg_max - exg_min)

        # Threshold: keep green areas, set others to black
        mask = exg_percentage > exg_threshold_percent

        # Create output image: all black
        out_img = np.zeros_like(image)

        if out_img.shape[2] == 4:  # has alpha channel
            out_img[:, :, 3] = 255  # set alpha to fully opaque

        # Copy original color where ExG percentage is high enough
        out_img[mask] = image[mask]

        return out_img

    def save_image(self, output_path: str, image_to_save: np.ndarray) -> None:
        cv2.imwrite(output_path, image_to_save)

    def load_cutout_id_image_dictionary(self) -> dict[str, tuple[np.ndarray, list]]:

        # Look at config to see all species that any form of preprocessing has been requested for
        species_processes_dictionary = {}

        # invert the dictionary from the config
        for preprocess in self.preprocess_cutouts.keys():
            species_list = list(self.preprocess_cutouts[preprocess])
            for species in species_list:
                params = self.preprocess_cutouts[preprocess][species]
                species = species.upper()
                if species not in species_processes_dictionary:
                    species_processes_dictionary[species] = []

                if species in self.common_names:
                    species_processes_dictionary[species].append((preprocess,params)) 
                else:
                    log.warning(f"Requested preprocessing for {species} but did not specify in cfg.cutout_filters.category.common_name")
                    log.warning("Skipping this species")

        if species_processes_dictionary:
            for species in species_processes_dictionary:
                if species_processes_dictionary[species]:  # Check that the value is not empty
                    preprocesses = [p for p, _ in species_processes_dictionary[species]]
                    preprocess_str = add_grammar_and_capitlization_to_list(preprocesses)
                    log.info(f"{species.title()} had {preprocess_str} requested")
    
        # Loop through all cutouts downloaded. If a cutout is downloaded and its species has had
        # a preprocess requested for it load it into the dicionary with its images and the process
        # that have been requested for that particular species
        cutout_image_dictionary = {}
        for cutout in tqdm(self.all_cutouts, desc="Loading images"):
            species = query_for_cutout_metadata(cutout, self.cursor)

            # Check if this cutout id has a preprocess requested, if so load into the dictionary
            if species in species_processes_dictionary.keys():
                img = cv2.imread(str(f"{self.cutout_path}/{cutout}.png"), cv2.IMREAD_UNCHANGED)
                if img is None:
                    raise FileNotFoundError(f"Could not read image: {str(f'{self.cutout_path}/{cutout}.png')}")

                cutout_image_dictionary[cutout] = (img, species_processes_dictionary[species])
                
        return cutout_image_dictionary

def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    CutoutProcessor(cfg)

    