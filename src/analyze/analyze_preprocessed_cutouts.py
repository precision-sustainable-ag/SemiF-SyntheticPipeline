import os
import cv2
import random
import logging
import numpy as np
from omegaconf import DictConfig

from move_cutouts import CutoutDownloader
from utils.utils import clear_directory, index_cutouts_by_species
from utils.pdf import PDFDrafter, add_grammar_and_capitlization_to_list
from preprocess_cutouts import invert_and_check_species_preprocess_dictionary

log = logging.getLogger(__name__)

import sys

class PreprocessAnalyzer():
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.db_path = str(cfg.paths.sql_database)
        self.original_cutout_path = cfg.paths.cutoutdir

        recipe_file = f"{cfg.paths.recipesdir}/{cfg.project_name}_{cfg.sub_name}.json"
        self.cutouts_indexed_by_species = index_cutouts_by_species(recipe_file)

        self.species_cutout_dict = {}

        self.preprocess_cutouts = cfg.preprocess_cutouts

        self.common_names = [
            name.upper() for name in self.cfg.cutout_filters.category.common_name
        ]
    
    def build_description(self, report: PDFDrafter) -> str:

        # Grab list of preprocesses
        preprocesses_list = list(set(preprocess.lower().replace('_', ' ') for preprocess in self.cfg.preprocess_cutouts.keys()))
        preprocesses_str = add_grammar_and_capitlization_to_list(preprocesses_list)
        
        # Grab list of species
        seen = set()
        species_list = []
        for preprocess in self.cfg.preprocess_cutouts:
            for species in self.cfg.preprocess_cutouts[preprocess]:
                species = species.upper()
                if species not in seen:
                    seen.add(species)
                    species_list.append(species)
        species_str = add_grammar_and_capitlization_to_list(species_list)

        # Make description string
        technique_or_techniques = "techniques" if (len(preprocesses_list) > 1) else "technique"
        description = (
            f"This report details the application of {technique_or_techniques} preprocessing to the species {species_str}, "
            f"as specified in {preprocesses_str}. It features side-by-side visual comparisons of cutouts before and after preprocessing, "
            f"demonstrating the effects of the applied techniques. A representative sample from each species is included, "
            f"carefully chosen based on metadata distribution—for example, if bbox ranges from 0 to 600, "
            f"cutouts with bbox 100,200,300,..,600 are selected to reflect this spread."
        )
        return description, species_list

    def grab_data_and_build_body_of_report(self, species_list: str, report: PDFDrafter):

        species_processes_dictionary = invert_and_check_species_preprocess_dictionary(self.preprocess_cutouts, self.common_names)

        data_for_body_of_pdf = {}
        

        for species in species_list:  
            species = species.upper()
            formatted_preprocess = [
                f"{preprocess.title().replace('_', ' ')} at level {params}" if preprocess.title() == "Remove_Soil"
                else f"{preprocess.title().replace('_', ' ')} with params {params}"
                for preprocess, params in species_processes_dictionary[species]
            ]
            preprocess_string = add_grammar_and_capitlization_to_list(formatted_preprocess)

            species_heading = f"{species.title()} had the following preprocess performed {preprocess_string}. The results are shown below."

            # Find the samples of cutouts based on metadata
            if species not in self.cutouts_indexed_by_species.keys():
                log.warning("Requested analysis for a species that wasnt preprocessed. Skipping.")
                continue

            preprocessed_cutouts = self.cutouts_indexed_by_species[species]
            
            meta_data = 'blur_effect'
            list_of_cutout_metadata_for_species = self.pick_cutouts_based_on_metadata(preprocessed_cutouts, meta_data)
            list_of_cutouts_for_species = []
            for cutout in list_of_cutout_metadata_for_species:
                list_of_cutouts_for_species.append(cutout["cutout_id"])

            data_for_body_of_pdf[species] = (species_heading, list_of_cutouts_for_species)
        
        self.download_orignal_cutouts(data_for_body_of_pdf)

        report.compare_cutouts(data_for_body_of_pdf)

        clear_directory(f"{self.cfg.paths.cutoutdir}/tmp")

    def download_orignal_cutouts(self, cutouts_to_download: dict[tuple[str, list]]) -> None:

        list_of_cutouts_to_download = []
        for species in cutouts_to_download:
            _,list_of_cutouts_for_species = cutouts_to_download[species]
            list_of_cutouts_to_download.extend(list_of_cutouts_for_species)

        downloader = ModifiedCutoutDownloader(self.cfg)
        os.makedirs(f"{self.cfg.paths.cutoutdir}/tmp", exist_ok=True)
        downloader.process_cutouts_sequentially(list_of_cutouts_to_download)

    def pick_cutouts_based_on_metadata(self, preprocessed_cutouts, metadata):
        # Sort them by bounding box area
        sorted_cutouts = sorted(
            preprocessed_cutouts,
            key=lambda x: x['cutout_props'][metadata]
        )

        # Pick 6 equally spaced cutouts
        num_to_select = 6
        total = len(sorted_cutouts)

        if total <= num_to_select:
            selected_cutouts = sorted_cutouts  # Not enough, return all
        else:
            step = total / num_to_select
            selected_cutouts = [sorted_cutouts[int(i * step)] for i in range(num_to_select)]
        
        return selected_cutouts


class ModifiedCutoutDownloader(CutoutDownloader):
    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.local_download_folder = f"{cfg.paths.cutoutdir}/tmp"

    def process_cutouts_sequentially(self, allowed_cutout_ids: list) -> None:

        synthetic_images = self.load_json(self.json_file_path)

        unique_cutouts = self.get_unique_cutouts(synthetic_images)

        for cutout_id, batch_id in unique_cutouts.items():
            if cutout_id in allowed_cutout_ids:
                self.download_image(cutout_id, batch_id)

        log.info("Download process completed in serial mode.")

def main(cfg: DictConfig) -> None:    

    log.info("Reached analyze_preprocessed_cutouts subtask of analysis")


    # Create PDF from graphs
    report = PDFDrafter(cfg)
    analyzer = PreprocessAnalyzer(cfg)

    '''
        Heading
    '''
    heading = "Analysis of Preprocessing"

    '''
        Description
    '''
    description, species_list = analyzer.build_description(report)
    report.initialize_heading_and_description(heading, description)

    '''
        Grab data to pass into PDF.py to build body of report
    '''
    analyzer.grab_data_and_build_body_of_report(species_list, report)

    '''
        Save PDF
    '''
    report.save_pdf()