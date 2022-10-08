import logging
import os

import hydra
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

import config_cutouts
import synthesize  # Do not remove

log = logging.getLogger(__name__)


@hydra.main(version_base="1.2", config_path="conf", config_name="config")
def run_SYNTH(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)
    log.info(f"Starting task {cfg.general.task}")
    task = get_method(f"{cfg.general.task}.main")
    task(cfg)


if __name__ == "__main__":
    run_SYNTH()
