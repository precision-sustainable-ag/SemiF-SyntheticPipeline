import cv2
import sqlite3
import logging
import numpy as np
from tqdm import tqdm
from omegaconf import DictConfig
from utils.utils import index_cutouts_by_species, add_grammar_and_capitlization_to_list

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

        # perform preprocessing
        self.perform_preprocessing()
        

    def remove_soil(self, image: np.ndarray, exg_threshold: float) -> np.ndarray:
        """
            Perform basic EXG
        """

        if not (-50 <= exg_threshold <= 50):
            log.error(f"Exceeding threshold limit [-50, 50] at {exg_threshold}, skipping")
            return image


        # Convert to float32 for ExG calculation
        img_float = image[:, :, :3].astype(np.float32)

        # Split channels (OpenCV uses BGR)
        B, G, R = cv2.split(img_float)

        # Compute Excess Green Index: ExG = 2G - R - B
        exg = 2 * G - R - B

        # Threshold: keep green areas, set others to black
        mask = exg > (exg_threshold*10)

        # Create output image: all black
        out_img = np.zeros_like(image)

        if out_img.shape[2] == 4:  # has alpha channel
            out_img[:, :, 3] = 255  # set alpha to fully opaque

        # Copy original color where ExG percentage is high enough
        out_img[mask] = image[mask]

        return out_img

    def save_images(self, cutout_image_dictionary: dict[str, np.ndarray]) -> None:
        log.info("Saving preprocessed images")
        for cutout in tqdm(cutout_image_dictionary.keys(), desc="Saving cutouts"):
            cv2.imwrite(f"{self.cfg.paths.cutoutdir}/{cutout}.png", cutout_image_dictionary[cutout])
            
    def perform_preprocessing(self) -> dict[str, tuple[np.ndarray, list]]:
        """
            This function handles all the preprocessing tasks. It will go the config, figure out which species need preprocessing
            and then preprocess those species.
        """

        # Check which species were downloaded (by looking at the recipe)
        downloaded_species_dict_path = f"{self.cfg.paths.recipesdir}/{self.cfg.project_name}_{self.cfg.sub_name}.json"
        downloaded_species_dict = index_cutouts_by_species(downloaded_species_dict_path)

        # Look at config to see all species that any form of preprocessing has been requested for
        species_processes_dictionary = invert_and_check_species_preprocess_dictionary(self.preprocess_cutouts, self.common_names)

        # create a dictionary of cutouts and their images
        cutout_image_dictionary = {} 

        if species_processes_dictionary:
            for species in species_processes_dictionary:
                species = species.upper()
                if species_processes_dictionary[species]:  # Check that the value is not empty

                    preprocesses_parameter = [(pre, param) for pre, param in species_processes_dictionary[species]]
                    preprocesses_parameter_list = [pre for pre, param in preprocesses_parameter]
                    preprocess_str = add_grammar_and_capitlization_to_list(preprocesses_parameter_list)
                    log.info(f"{species.title()} had {preprocess_str} requested")

                    # Check to see if species was downlaoded before performing preprocessing
                    if species not in downloaded_species_dict.keys():
                        log.warning("Requested preprocessing for a species that wasnt downloaded. Skipping.")
                        continue
                    species_group = downloaded_species_dict[species]
                    for cutout in tqdm(species_group, desc=f"Processing cutouts for {species}"):
                        for process_name, parameter in preprocesses_parameter:
                            process_name = process_name.lower()

                            if not cutout["cutout_id"] in cutout_image_dictionary.keys():
                                img = cv2.imread(str(f'{self.cutout_path}/{cutout["cutout_id"]}.png'), cv2.IMREAD_UNCHANGED)
                                if img is None:
                                    raise FileNotFoundError(f"Could not read image: {str(f'{self.cutout_path}/{cutout}.png')}")
                                cutout_image_dictionary[cutout["cutout_id"]] = img


                                method = getattr(self, process_name, None)
                                if method:
                                    cutout_image_dictionary[cutout["cutout_id"]] = method(cutout_image_dictionary[cutout["cutout_id"]], parameter)
                                else:
                                    raise ValueError(f"Unknown process: {process_name}")

        self.save_images(cutout_image_dictionary)


def invert_and_check_species_preprocess_dictionary(preprocess_cutouts: dict[str, dict[str, list]],common_names: list[str]) -> dict[str, list[tuple[str, list]]]:   
    """
        This function takes the preprocess dictionary in the config and inverts it to make the keys based on species
    """
    species_processes_dictionary = {}
    for preprocess in preprocess_cutouts.keys():

        if not preprocess_cutouts[preprocess]:
            log.error(f"{preprocess} was left empty, skipping")
            continue

        preprocess_upper = preprocess.upper()
        species_list = list(preprocess_cutouts[preprocess])
        
        for species in species_list:
            species_upper = species.upper()
            params = preprocess_cutouts[preprocess][species]

            if species_upper in (name.upper() for name in common_names):
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


def main(cfg: DictConfig) -> None:
    log.info("Reached cutout preprocessing task")

    if not cfg.preprocess_cutouts:
        log.error("Empty preprocess dictionary, skipping task")
        return

    CutoutProcessor(cfg)

    