import json
import logging
import random
from multiprocessing import Pool, cpu_count

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm, trange

from synth_utils.config_utils import read_cutouts, sort_cutouts
from synth_utils.datasets import SynthData
from synth_utils.synth_utils import SynthPipeline

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    alldf = read_cutouts(cfg.data.cutoutdir)
    sort_cutouts(alldf, cfg, save_csv=True)
    log.info("Cutouts sorted.")
    with open(cfg.data.speciesinfo) as f:
        spec_info = json.load(f)
        spec_info = spec_info["species"]
    # Create synth data container
    log.info("Creating SynthData container")
    synthdata = SynthData(synthdir=cfg.data.synthdir,
                          cfg=cfg,
                          background_dir=cfg.data.backgrounddir,
                          pot_dir=cfg.data.potdir,
                          color_map=spec_info)

    # cutout_groups = synthdata.read_cutout_dcs()
    log.info(f"SynthData container created.")

    syn = SynthPipeline(synthdata, cfg)

    if cfg.synth.multiprocess:
        procs = cpu_count() - 3

        with Pool(processes=procs) as pool:
            with tqdm(total=syn.count) as pbar:
                for _ in pool.imap_unordered(syn.pipeline):
                    pbar.update()
            pool.close()
            pool.join()
    else:
        for i in range(0, cfg.synth.count):
            syn.pipeline()
