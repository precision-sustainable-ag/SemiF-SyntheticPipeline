import sys
import hydra
from from_root import from_root
from omegaconf import DictConfig, OmegaConf

# repo imports
from analyze_images.pdf import generate_pdf
from analyze_images.analyze_cutouts import CutoutAnalyzer

@hydra.main(version_base="1.2", config_path=str(from_root("conf")), config_name="config")
def main(cfg: DictConfig):
    cfg = OmegaConf.create(cfg)

    # Graph all species specified in config
    all_cutouts = CutoutAnalyzer("all", cfg.cutout_filters.category.common_name, [])

    # Graph cutouts from local folder
    CutoutAnalyzer("downloaded", str(from_root("data/cutouts")), all_cutouts.states)

    generate_pdf()

if __name__ == "__main__":
    main()