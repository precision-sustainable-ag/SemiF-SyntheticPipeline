import os
import cv2
import json
import random
import sqlite3
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from utils.pdf import PDFDrafter
from utils.utils import query_for_cutout_metadata

class PreprocessAnalyzer():
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.db_path = str(cfg.paths.sql_database)
        self.original_cutout_path = cfg.paths.cutoutdir
        self.preproccessed_cutout_path = cfg.paths.preprocessed_cutoutdir

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        self.cursor = conn.cursor()

        self.species_cutout_dict = {}
    
    def compute_iou(self, cutout_id: str) -> float:

        img1 = cv2.imread(str(f"{self.original_cutout_path}/{cutout_id}.png"), cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(str(f"{self.preproccessed_cutout_path}/{cutout_id}.png"), cv2.IMREAD_UNCHANGED)

        assert img1.shape == img2.shape, "Images must be the same shape"
        
        intersection = np.logical_and(img1, img2).sum()
        union = np.logical_or(img1, img2).sum()
        
        iou = intersection / union if union != 0 else 0
        return iou

    def build_description(self, report: PDFDrafter) -> str:

        # Grab list of preprocesses
        preprocesses_list = list(set(preprocess.lower().replace('_', ' ') for preprocess in self.cfg.preprocess_cutouts.keys()))
        preprocesses_str = report.add_grammar_and_capitlization_to_list(preprocesses_list)
        
        # Grab list of species
        seen = set()
        species_list = []
        for preprocess in self.cfg.preprocess_cutouts:
            for species in self.cfg.preprocess_cutouts[preprocess]:
                species = species.upper()
                if species not in seen:
                    seen.add(species)
                    species_list.append(species)
        species_str = report.add_grammar_and_capitlization_to_list(species_list)

        # Make description string
        technique_or_techniques = "techniques" if (len(preprocesses_list) > 1) else "technique"
        description = (
            f"The following is a report of the {preprocesses_str} preprocessing {technique_or_techniques}, "
            f"exacted on the species {species_str}. The aim is to display the cutouts before and after preprocessing, "
            f"allowing the user to see the effects the preprocessing has on the cutouts. A small sample is shown from "
            f"each of the species. This sample was chosen based on the IoU between the original and preprocessed cutout. "
        )
        return description, species_list

    def grab_data_and_build_body_of_report(self, species_list: str, report: PDFDrafter):

        data_for_body_of_pdf = {}

        species_processes_dictionary = {}
        for preprocess in self.cfg.preprocess_cutouts.keys():
            species_list = list(self.cfg.preprocess_cutouts[preprocess])
            for species in species_list:
                params = self.cfg.preprocess_cutouts[preprocess][species]
                species = species.upper()
                if species not in species_processes_dictionary:
                    species_processes_dictionary[species] = []

                species_processes_dictionary[species].append((preprocess,params)) 

        for species in species_list:  
            species = species.upper()
            formatted_preprocess = [
                f"{preprocess.title().replace('_', ' ')} at {params}%" if preprocess.title() == "Remove_Soil"
                else f"{preprocess.title().replace('_', ' ')} with params {params}"
                for preprocess, params in species_processes_dictionary[species]
            ]

            species_heading = f"{species.title()} had the following preprocess performed {formatted_preprocess}. The results are shown below."

            # Find the cutouts with the smallest IoU, get a random 40 to generate pdf faster
            cutouts_and_their_ious = {}
            preprocessed_cutouts_for_species = self.species_cutout_dict[species]
            selected_cutouts = random.sample(preprocessed_cutouts_for_species, min(40, len(preprocessed_cutouts_for_species)))

            for cutout in selected_cutouts:
                cutouts_and_their_ious[cutout] = self.compute_iou(cutout)

            list_of_cutouts_for_species = self.get_strings_for_smallest_six_floats(cutouts_and_their_ious)

            data_for_body_of_pdf[species] = (species_heading, list_of_cutouts_for_species)

        report.compare_cutouts(data_for_body_of_pdf)

    def create_species_cutout_id_dict(self, species_list):

        # Grab list of all cutouts that were preprocessed, remove .png
        preprocessed_cutouts = [filename[:-4] for filename in os.listdir(self.cfg.paths.preprocessed_cutoutdir)]

        # Query to for the cutout to attach it to the dict with the appropriate species key
        for cutout_id in preprocessed_cutouts:
            cutouts_species = query_for_cutout_metadata(cutout_id, self.cursor)

            if cutouts_species not in species_list:
                continue

            if cutouts_species not in self.species_cutout_dict:
                self.species_cutout_dict[cutouts_species] = []
            self.species_cutout_dict[cutouts_species].append(cutout_id)

    def get_strings_for_smallest_six_floats(self, data: dict[str, float]) -> list[str]:
        # Sort by value (the floats), take the first 6, return the keys
        return [k for k, _ in sorted(data.items(), key=lambda item: item[1])[:6]]

def main(cfg: DictConfig) -> None:    

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

    analyzer.create_species_cutout_id_dict(species_list)
    report.initialize_heading_and_description(heading, description)

    '''
        Grab data to pass into PDF.py to build body of report
    '''
    analyzer.create_species_cutout_id_dict(species_list)
    analyzer.grab_data_and_build_body_of_report(species_list, report)

    '''
        Save PDF
    '''
    report.save_pdf()