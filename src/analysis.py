import hydra
import logging
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Import the task functions
from src.analyze_cutouts import main as analyze_cutouts

# Define a registry of tasks
TASK_REGISTRY = {
    "analyze_cutouts": analyze_cutouts,
}

@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    
    analysis_subtasks = cfg.tasks.analysis

    log.info("Reached analysis.py")
    log.info("Going through subtasks under analysis")

    for sub_task_name in analysis_subtasks:

        if sub_task_name in TASK_REGISTRY:
            log.info(f"Running task {sub_task_name}")
            try:
                TASK_REGISTRY[sub_task_name](cfg)
            except Exception as e:
                log.error(f"Error running task {sub_task_name}: {e}")
                raise

        else:
            log.error(f"Task {sub_task_name} not found in analysis task registry")
            raise ValueError(f"Task {sub_task_name} not found in analysis task registry")
    
    log.info("Analysis completed.")
    return

if __name__ == "__main__":
    main()