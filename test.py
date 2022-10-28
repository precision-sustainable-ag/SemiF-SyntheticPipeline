import json
import os

os.chdir("/home/weedsci/matt/SemiF-SyntheticPipeline")
import numpy as np
from omegaconf import DictConfig

from synth_utils.config_utils import read_cutouts, sort_cutouts
from synth_utils.datasets import SynthData
from synth_utils.synth_utils import SynthPipeline


def main(cfg: DictConfig) -> None:
    alldf = read_cutouts(cfg.data.cutoutdir)
    sort_cutouts(alldf, cfg, save_csv=True)
    with open(cfg.data.speciesinfo) as f:
        spec_info = json.load(f)
        spec_info = spec_info["species"]
    # Create synth data container
    synthdata = SynthData(synthdir=cfg.data.synthdir,
                          cfg=cfg,
                          background_dir=cfg.data.backgrounddir,
                          pot_dir=cfg.data.potdir,
                          color_map=spec_info)

    # print(synthdata)
