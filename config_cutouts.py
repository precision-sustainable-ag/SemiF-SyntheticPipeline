from omegaconf import DictConfig


def main(cfg: DictConfig) -> None:
    if cfg.cutouts.mode == "balanced":
        # TODO configure "species" and "balanced" to work
        print("balanced")

    # cutoutmeta2csv(cutoutdir, batch_id, csv_path)
    pass
