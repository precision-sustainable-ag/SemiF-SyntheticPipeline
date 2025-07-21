import os
import cv2
import random
import logging
import numpy as np
from omegaconf import DictConfig

from utils.pdf import PDFDrafter
from move_cutouts import CutoutDownloader
from preprocess_cutouts import remove_soil, invert_and_check_species_preprocess_dictionary
from utils.utils import clear_directory, index_cutouts_by_species, add_grammar_and_capitlization_to_list, image_comp_grid

log = logging.getLogger(__name__)

class PreprocessAnalyzer():
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg

        recipe_file = f"{cfg.paths.recipesdir}/{cfg.project_name}_{cfg.sub_name}.json"
        self.cutouts_indexed_by_species = index_cutouts_by_species(recipe_file)

        self.preprocess_cutouts = cfg.preprocess_cutouts
        self.preprocesses_list = list(set(preprocess.lower().replace('_', ' ') for preprocess in cfg.preprocess_cutouts.keys()))
        self.species_list = self.list_species_for_preprocessing()

        self.num_cutouts_per_species = 15

        # EXG CONFIGS
        self.exg_tests = [0, 20, 40, 60, 80, 100]

        self.test_dictionary = {}
        self.test_dictionary['remove_soil'] = self.exg_tests

        self.row_label_dictionary = {}
        self.row_label_dictionary['remove_soil'] = ['no exg'] + [f'exg={x}' for x in self.test_dictionary['remove_soil']]

        self.row_spacing_dictionary = {}
        self.row_spacing_dictionary['remove_soil'] = .16

        self.analysisdir = cfg.paths.cutoutanalysisdir
    
    def build_description(self, report: PDFDrafter) -> str:

        # Make list of preprocesses a string
        preprocesses_str = add_grammar_and_capitlization_to_list(self.preprocesses_list)

        species_str = add_grammar_and_capitlization_to_list(self.species_list)

        # Make description string
        description = (
            f"This section provides a report on the use of {preprocesses_str} for the species {species_str}. Please note: "
            f"the results shown do not reflect the combined impact of multiple preprocessing steps on a single cutout. "
            f"Applying more than one process to the same group of cutouts may yield unpredictable outcomes."
        )
        return description

    def build_body_of_report(self, report: PDFDrafter):
        """
            This function works to build the body of the report. 
            It handles grabbing the images building the plots 
            and formulating the headings
        """

        # pass in species list to ensure we analyze all cutouts
        species_processes_dictionary = invert_and_check_species_preprocess_dictionary(self.preprocess_cutouts, self.species_list) 
        
        # remove empty params
        species_processes_dictionary = {
            species: [process for (process, _) in processes]
            for species, processes in species_processes_dictionary.items()
        }

        for species in species_processes_dictionary:

            if species not in self.cutouts_indexed_by_species:
                log.error(f"Requested analysis for a species: {species} not in recipe (not requested in category.common_name). Skipping")
                continue

            # Get list of cutouts to add to report
            meta_data = 'bbox_area_cm2'
            list_of_cutout_metadata_for_species = self.pick_cutouts_based_on_metadata(self.cutouts_indexed_by_species[species], meta_data)
            list_of_cutouts_for_species = []
            for cutout in list_of_cutout_metadata_for_species:
                list_of_cutouts_for_species.append(cutout["cutout_id"])

            # Loop over preprocesses, perform them, create plot, then delete downloaded cutouts
            for preprocess in species_processes_dictionary[species]:
                preprocess=preprocess.lower()

                # Download cutouts
                download_directory=f"{self.analysisdir}/{species}/{preprocess}"
                os.makedirs(download_directory, exist_ok=True)
                self.download_cutouts(species, list_of_cutouts_for_species, download_directory)

                # Apply preprocessing
                for cutout in list_of_cutouts_for_species:
                    img = cv2.imread(f"{download_directory}/{cutout}.png", cv2.IMREAD_UNCHANGED) 

                    process_function = PROCESSING_METHODS.get(preprocess)
                    for idx, param in enumerate(self.test_dictionary[preprocess]):
                        processed_image = process_function(img,"",param)

                        # save file add underscore to keep track on where to place in plot
                        underscores = '_' * (idx + 1)
                        filename = f"{download_directory}/{cutout}{underscores}.png"
                        cv2.imwrite(filename, processed_image)
            
                # Create grid from downloaded/preprocessed cutouts
                num_cutouts = len(list_of_cutout_metadata_for_species)
                num_params = len(self.test_dictionary[preprocess])+1 # add one to include original image
                row_labels = self.row_label_dictionary[preprocess]
                col_labels = sorted(list_of_cutouts_for_species)
                row_spacing = self.row_spacing_dictionary[preprocess]
                image_grid_path = image_comp_grid(base_dir=download_directory, row_labels=row_labels, col_labels=col_labels, num_rows=num_params, num_cols=num_cutouts, row_spacing=row_spacing)

                # Add grid to report with caption
                caption = f"Below is {species.title()} with varying {preprocess.replace('_', ' ').title()} applied"
                report.add_one_image_with_caption(caption, image_grid_path)

                # Delete grid and the images it used
                # clear_directory(self.analysisdir)

    def download_cutouts(self, species: str, list_of_cutouts: list[str], download_directory: str) -> None:
        downloader = CutoutDownloader(self.cfg)

        # Download cutouts
        downloader.local_download_folder = download_directory
        downloader.process_cutouts_sequentially(list_of_cutouts)

    def pick_cutouts_based_on_metadata(self, preprocessed_cutouts: list, metadata: str):
        # Sort by brown colors
        def is_brown(x):
            r, g, b = x['cutout_props']['cropout_rgb_mean']
            return r - min(g, b)

        sorted_cutouts = sorted(
            preprocessed_cutouts,
            key=is_brown,
            reverse=True 
        )

        total = len(sorted_cutouts)

        if total <= self.num_cutouts_per_species:
            selected_cutouts = sorted_cutouts
        else:
            step = total / self.num_cutouts_per_species
            selected_cutouts = [sorted_cutouts[int(i * step)] for i in range(self.num_cutouts_per_species)]
        
        return selected_cutouts

    def list_species_for_preprocessing(self) -> list[str]:
        in_list = set()
        species_list = []

        if self.preprocess_cutouts:
            for preprocess_key, species_dict in self.preprocess_cutouts.items():
                if preprocess_key == "num_workers":
                    continue  

                if not isinstance(species_dict, dict):
                    continue

                species_iterable = species_dict.keys()
                if not species_iterable:
                    continue

                for species in species_iterable:
                    species = species.upper()
                    if species not in in_list:
                        in_list.add(species)
                        species_list.append(species)

        return species_list

PROCESSING_METHODS = {
    "remove_soil": remove_soil,
}

def main(cfg: DictConfig, report: PDFDrafter) -> None:    

    log.info("Reached analyze_preprocessed_cutouts subtask of analysis")

    if not cfg.preprocess_cutouts:
        log.error("Left analyze_preprocessed_cutouts dictionary empty. Skipping this part of analysis.")
        return  

    # Create PDF from graphs
    analyzer = PreprocessAnalyzer(cfg)

    '''
        Heading
    '''
    heading = "Analysis of Preprocessing"

    '''
        Description
    '''
    description = analyzer.build_description(report)
    report.initialize_heading_and_description(heading, description)

    '''
        Grab data to pass into PDF.py to build body of report
    '''
    analyzer.build_body_of_report(report)