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

    resize_area = None

    try:
        if "create_recipes" in cfg.tasks:
            create_fn = get_method("create_recipes.main")
            resize_area = create_fn(cfg)

        if "move_cutouts" in cfg.tasks:
            move_fn = get_method("move_cutouts.main")
            move_fn(cfg)

        if "synthesize" in cfg.tasks:
            synth_fn = get_method("synthesize.main")
            synth_fn(cfg, resize_area=resize_area)

    except Exception as e:
        log.exception("Failed")
        return


if __name__ == "__main__":
    run_SYNTH()
