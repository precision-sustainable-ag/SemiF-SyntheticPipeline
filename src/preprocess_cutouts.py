import os
import cv2
import sqlite3
import logging
import numpy as np
from tqdm import tqdm
from typing import Union
from functools import partial
from omegaconf import DictConfig
from multiprocessing import Pool
from omegaconf import DictConfig, ListConfig

from utils.utils import index_cutouts_by_species, add_grammar_and_capitlization_to_list

log = logging.getLogger(__name__)

class CutoutProcessor():
    def __init__(self, cfg) -> None:
        self.cfg = cfg
        self.preprocess_cutouts = cfg.preprocess_cutouts
        self.num_workers = cfg.preprocess_cutouts.num_workers

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

        # perform preprocessing
        self.perform_preprocessing()


    def load_work_items(self,species_group: list[dict],cutout_image_dictionary: dict[str, np.ndarray]) -> list[tuple[str, np.ndarray]]:
        """
        Creates list of (cutout_id, image) tuples.
        """
        work_items = []

        for item in species_group:
            cutout_id = item['cutout_id']
            image_path = os.path.join(self.cutout_path, f"{cutout_id}.png")

            if not os.path.exists(image_path):
                log.warning(f"Image not found for cutout {cutout_id}, skipping")
                continue

            img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            if img is None:
                log.warning(f"Failed to read image for cutout {cutout_id}, skipping")
                continue

            cutout_image_dictionary[cutout_id] = img
            work_items.append((cutout_id, img))

        return work_items

    def perform_preprocessing(self) -> dict[str, tuple[np.ndarray, list]]:
        """
            This function handles all the preprocessing tasks. It will go the config,
            figure out which species need preprocessing and then preprocess those species.
            After preprocessing is done for a species we will overwrite the old images
        """
        # Check which species were downloaded (by looking at the recipe)
        downloaded_species_dict_path = f"{self.cfg.paths.recipesdir}/{self.cfg.project_name}_{self.cfg.sub_name}.json"
        downloaded_species_dict = index_cutouts_by_species(downloaded_species_dict_path)

        # Look at config to see all species that any form of preprocessing has been requested for
        species_processes_dictionary = invert_and_check_species_preprocess_dictionary(self.preprocess_cutouts, self.common_names)

        if species_processes_dictionary:
            for species in species_processes_dictionary:
                species = species.upper()

                if species_processes_dictionary[species]:  # Check that the value is not empty

                    preprocesses_parameter = [(pre, param) for pre, param in species_processes_dictionary[species]]
                    preprocesses_parameter_list = [pre for pre, param in preprocesses_parameter]
                    preprocess_str = add_grammar_and_capitlization_to_list(preprocesses_parameter_list)
                    log.info(f"{species.title()} had {preprocess_str} requested")

                    # Check to see if species was downloaded before performing preprocessing
                    if species not in downloaded_species_dict.keys():
                        log.warning("Requested preprocessing for a species that wasn't downloaded. Skipping.")
                        continue
                    species_group = downloaded_species_dict[species]

                    # add overwrite images task to the preprocess list
                    preprocesses_parameter += [("overwrite_images", self.cfg)]

                    # create a dictionary of cutouts and their images
                    cutout_image_dictionary = {}
                    for process_name, parameter in preprocesses_parameter:
                        process_name = process_name.lower()
                        method = PROCESSING_METHODS.get(process_name)
                        if not method:
                            raise ValueError(f"Unknown process: {process_name}")

                        # Load images only at the first preprocessing step, else load them in from dictionary
                        work_items=[]
                        if not cutout_image_dictionary:
                            work_items = self.load_work_items(species_group, cutout_image_dictionary)
                        else:
                            work_items = [(cutout_id, img) for cutout_id, img in cutout_image_dictionary.items()]

                        with Pool(processes=self.num_workers) as pool:
                            func = partial(
                                process_cutout,
                                process_name=process_name,
                                parameter=parameter
                            )

                            # Run multiprocessing
                            results = list(
                                tqdm(
                                    pool.imap(func, work_items),
                                    total=len(work_items),
                                    desc=f"Processing {process_name} cutouts for {species}"
                                )
                            )

                        # Update the dictionary in the parent process
                        for cutout_id, processed_img in results:
                            cutout_image_dictionary[cutout_id] = processed_img

def remove_soil(img: np.ndarray, cutout_id: str, exg_threshold: float) -> np.ndarray:
    """
        Perform basic EXG
    """

    # Convert to float32 for ExG calculation
    img_float = img[:, :, :3].astype(np.float32)

    # Split channels (OpenCV uses BGR)
    B, G, R = cv2.split(img_float)

    # Compute Excess Green Index: ExG = 2G - R - B
    exg = 2 * G - R - B

    # Threshold: keep green areas, set others to black
    mask = exg > (exg_threshold)

    # Create output image: all black
    out_img = np.zeros_like(img)

    if out_img.shape[2] == 4:  # has alpha channel
        out_img[:, :, 3] = 255  # set alpha to fully opaque

    # Copy original color where ExG percentage is high enough
    out_img[mask] = img[mask]

    return out_img

def overwrite_images(img: np.ndarray, cutout_id: str, cfg: DictConfig) -> np.ndarray:
    """
        Return not used but added to fit into multiprocessing
    """
    cv2.imwrite(f"{cfg.paths.cutoutdir}/{cutout_id}.png", img)
    return img 

def process_cutout(cutout_pair: tuple[str, np.ndarray], process_name: str, parameter: Union[float, list, DictConfig]) -> tuple[str, np.ndarray]:
    """
        Applies a specified preprocessing method to a single cutout image.

        This function is designed to be used with multiprocessing, where each worker
        processes one image at a time. It retrieves the appropriate processing method
        from a registry and applies it to the image, using the given parameters.
    """
    cutout_id, img = cutout_pair

    method = PROCESSING_METHODS.get(process_name)
    if not method:
        raise ValueError(f"Unknown process: {process_name}")

    processed_img = method(img, cutout_id, parameter)
    return cutout_id, processed_img

def invert_and_check_species_preprocess_dictionary(preprocess_cutouts: dict[str, dict[str, list]],common_names: list[str]) -> dict[str, list[tuple[str, list]]]:   
    """
        This function takes the preprocess dictionary in the config and inverts it to make the keys based on species
    """
    species_processes_dictionary = {}
    for preprocess in preprocess_cutouts.keys():
        if preprocess == "num_workers":
            continue

        if not preprocess_cutouts[preprocess]:
            log.error(f"{preprocess} was left empty, skipping")
            continue

        preprocess_upper = preprocess.upper()
        species_data = preprocess_cutouts[preprocess]

        if isinstance(species_data, DictConfig):
            iterable = species_data.items()
        elif isinstance(species_data, ListConfig):
            iterable = ((species, []) for species in species_data)
        else:
            raise ValueError(f"Unsupported type for {preprocess}: {type(species_data)}")

        for species, params in iterable:
            species_upper = species.upper()

            if not common_names or species_upper in (name.upper() for name in common_names):
                if species_upper not in species_processes_dictionary:
                    species_processes_dictionary[species_upper] = []

                existing = species_processes_dictionary[species_upper]
                for existing_preprocess, existing_params in existing:
                    if existing_preprocess.upper() == preprocess_upper and existing_params != params:
                        raise ValueError(
                            f"Conflict: species '{species_upper}' has multiple different params "
                            f"for the same preprocess '{preprocess_upper}':\n"
                            f"- Existing: {existing_params}\n- New: {params}"
                        )

                # Safe to append
                species_processes_dictionary[species_upper].append((preprocess_upper, params))

            else:
                log.warning(
                    f"Requested preprocessing for {species_upper} but did not specify in cfg.cutout_filters.category.common_name"
                )
                log.warning("Skipping this species")

    return species_processes_dictionary

PROCESSING_METHODS = {
    "remove_soil": remove_soil,
    "overwrite_images": overwrite_images,
}

def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    if not cfg.preprocess_cutouts:
        log.error("Empty preprocess dictionary, skipping task")
        return

    CutoutProcessor(cfg)




