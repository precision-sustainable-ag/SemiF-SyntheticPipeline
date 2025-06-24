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
    print("hello world")

    report = PDFDrafter(cfg, )