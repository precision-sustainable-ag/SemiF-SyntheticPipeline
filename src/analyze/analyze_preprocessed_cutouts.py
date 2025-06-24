import os
import json
import sqlite3
import logging
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

from utils.pdf import PDFDrafter



class PreprocessAnalyzer():
    def __init__() -> None:
        pass


def main(cfg: DictConfig) -> None:

    # # Total num of cutouts
    # num_cutouts = all_cutouts.num_cutouts, specified_cutouts.num_cutouts

    # Create PDF from graphs
    report = PDFDrafter(cfg, (0,0))

    # Heading
    heading = "Analysis of Preprocessing"

    # Description
    preprocesses_list = list(set(preprocess.lower().replace('_', ' ') for preprocess in cfg.preprocess_cutouts.keys()))
    sorted_preprocesses = sorted(preprocesses_list, key=lambda s: s.lower())
    if len(sorted_preprocesses) > 1:
        titled = [s.title() for s in sorted_preprocesses]
        species_str = ', '.join(titled[:-1]) + f", and {titled[-1]}"
    else:
        species_str = sorted_preprocesses[0].title()
        
    technique_or_techniques = "techniques" if (len(sorted_preprocesses) > 1) else "technique"
    description = f"The following is a report of the {species_str} preprocessing {technique_or_techniques}. The aim is to display the cutouts before and after preprocessing, allowing the user to see the effects the preprocessing has on the cuttous."

    report.initialize_heading_and_description(heading, description)
    # report.add_graphs_to_pdf()
    report.save_pdf()


    report = PDFDrafter(cfg, )