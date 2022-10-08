import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    if cfg.cutouts.mode == "balanced":
        print("balanced")
    path = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts/MD_2022-07-05/MD_Row-10_1657032952_6.json"
    cutoutdir = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts"
    batch_id = "MD_2022-07-06"
    csv_path = "MD_2022-07-06.csv"
    # cutoutmeta2csv(cutoutdir, batch_id, csv_path)
    pass
