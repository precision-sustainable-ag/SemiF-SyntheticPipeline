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
    log.info(f"Starting task {','.join(cfg.tasks)}")
    
    for tsk in cfg.tasks:
        try:
            task = get_method(f"{tsk}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            return


if __name__ == "__main__":
    run_SYNTH()
