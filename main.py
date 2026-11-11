import logging
import os
import sys

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

sys.path.append("src")


log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_SYNTH(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    source = cfg.get("cutout_source", "legacy")
    if source not in ("legacy", "seg_to_cut", "agir_db"):
        raise ValueError(f"Unknown cutout_source: {source}")
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    for tsk in cfg.tasks.keys():
        if source in ("seg_to_cut", "agir_db") and tsk in ("analysis", "move_cutouts", "preprocess_cutouts"):
            log.info("Skipping %s: %s supplies metadata and ready-to-use cutout files", tsk, source)
            continue
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            if source in ("seg_to_cut", "agir_db"):
                raise
            return


if __name__ == "__main__":
    run_SYNTH()
