import os
import json
import sqlite3
import logging
import numpy as np
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from utils.pdf import PDFDrafter

class PreprocessAnalyzer():
    def __init__(self, cfg: DictConfig) -> None:

        self.cfg = cfg
        self.db_path = str(cfg.paths.sql_database)

        # Connect to database (READ ONLY)
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        cursor = conn.cursor()
    
    def compute_iou(self, img1: np.ndarray, img2: np.ndarray) -> float:
        assert img1.shape == img2.shape, "Images must be the same shape"
        
        intersection = np.logical_and(img1, img2).sum()
        union = np.logical_or(img1, img2).sum()
        
        iou = intersection / union if union != 0 else 0
        return iou

    def build_description(self, report: PDFDrafter) -> str:

        # Grab list of preprocesses
        preprocesses_list = list(set(preprocess.lower().replace('_', ' ') for preprocess in self.cfg.preprocess_cutouts.keys()))
        preprocesses_str = report.add_gramar_and_capitlization_to_string(preprocesses_list)
        
        # Grab list of species
        seen = set()
        species_list = []
        for preprocess in self.cfg.preprocess_cutouts:
            for species in self.cfg.preprocess_cutouts[preprocess]:
                if species not in seen:
                    seen.add(species)
                    species_list.append(species)
        species_str = report.add_gramar_and_capitlization_to_string(species_list)

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
        for preprocess in self.preprocess_cutouts.keys():
            species_list = list(self.preprocess_cutouts[preprocess])
            for species in species_list:
                params = self.preprocess_cutouts[preprocess][species]
                species = species.upper()
                if species not in species_processes_dictionary:
                    species_processes_dictionary[species] = []

                species_processes_dictionary[species].append((preprocess,params)) 

        for species in species_list:  
            formatted_preprocess = [
                f"{preprocess.title()} with params {params}"
                for preprocess, params in species_processes_dictionary[species]
            ]
            species_heading = f"{species.title()} had the following preprocess performed {formatted_preprocess}"



            data_for_body_of_pdf[species] = (species_heading, [list of cutout_ids])

        report.compare_cutouts()

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

    report.initialize_heading_and_description(heading, description)

    '''
        Grab data to pass into PDF.py to build body of report
    '''
    analyzer.grab_data_to_build_body_of_report(species_list, report)

    '''
        Save PDF
    '''
    report.save_pdf()