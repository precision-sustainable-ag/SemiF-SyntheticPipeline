import os
import hydra
import logging
from utils.pdf import PDFDrafter
from hydra.utils import get_method
from omegaconf import DictConfig, OmegaConf

log = logging.getLogger(__name__)

# Import the task functions
from src.analyze.analyze_cutouts import main as analyze_cutouts
from src.analyze.analyze_preprocessed_cutouts import main as analyze_preprocessed_cutouts

# Define a registry of tasks
TASK_REGISTRY = {
    "analyze_cutouts": analyze_cutouts,
    "analyze_preprocessed_cutouts": analyze_preprocessed_cutouts
}

@hydra.main(version_base="1.2", config_path="../conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """ Main entry point for the application """
    
    analysis_subtasks = cfg.tasks.analysis

    log.info("Reached analysis.py")

    file_path = cfg.paths.analysisdir
    os.makedirs(str(file_path), exist_ok=True)

    if cfg.tasks.analysis : 

        report = PDFDrafter(cfg)
        title = "Pre-Synthesis Analysis"
        author = "Maintainer: PSA CV Team"
        report.add_title_author_date(title,author)

        for sub_task_name in analysis_subtasks:

            if sub_task_name in TASK_REGISTRY:
                log.info(f"Running task {sub_task_name}")
                try:
                    TASK_REGISTRY[sub_task_name](cfg, report)
                except Exception as e:
                    log.error(f"Error running task {sub_task_name}: {e}")
                    raise

                # If we have more than one analysis task and we are not on the last one
                # add a new page
                if not sub_task_name == analysis_subtasks[-1]:
                    report.add_new_page()

            else:
                log.error(f"Task {sub_task_name} not found in analysis task registry")
                raise ValueError(f"Task {sub_task_name} not found in analysis task registry")
    
        report.save_pdf()
        log.info("Analysis completed.")
    else : 
        # Description
        task_list = list(set(name.lower() for name in TASK_REGISTRY.keys()))
        if len(task_list) > 1:
            titled = [s.title() for s in task_list]
            task_str = ', '.join(titled[:-1]) + f", and {titled[-1]}"
        else:
            task_str = task_list[0].title()
        log.info(f"No Analysis tasks requested.. set a task [{task_str}] in config.yaml")
    return

if __name__ == "__main__":
    main()