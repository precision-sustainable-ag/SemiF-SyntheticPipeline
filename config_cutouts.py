from pathlib import Path

import pandas as pd
from omegaconf import DictConfig

from utils.config_cutouts_utils import ConfigCutouts


def main(cfg: DictConfig) -> None:
    """ Creates csv file with all configs to pul from based on cutout config yaml."""
    # Using species proportions
    cc = ConfigCutouts(cfg)
    df = cc.df.reset_index(drop=True)

    if cfg.cutouts.save_csv:
        Path(Path(cfg.data.csvpath).parent).mkdir(parents=True, exist_ok=True)
        df.to_csv(cfg.data.csvpath, index=False)
