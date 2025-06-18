import logging
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    cfg = OmegaConf.create(cfg)

    analysis_subtasks = cfg.tasks.analysis

    log.info("Reached analysis.py")
    log.info("Going through subtasks under analysis")

    for sub_task_name in analysis_subtasks:
        try:
            task = get_method(f"{sub_task_name}.main")
            task(cfg)

        except Exception as e:
            log.exception("Failed")
            return
