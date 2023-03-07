import json
import logging
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from omegaconf import DictConfig
from tqdm import tqdm

from utils.datasets import Cutout

log = logging.getLogger(__name__)


class ConfigCutouts:

    def __init__(self, cfg):
        self.cfg = cfg
        self.cutoutdir = cfg.data.cutoutdir
        self.species = cfg.cutouts.species

        # Main filters
        self.green_sum = cfg.cutouts.green_sum
        self.area = cfg.cutouts.area
        self.extends_border = cfg.cutouts.extends_border
        self.is_primary = cfg.cutouts.is_primary

        self.df = self.sort_cutouts()

    def get_cutout_meta(self, path):
        with open(path) as f:
            j = json.load(f)
            cutout = Cutout(**j)
        return cutout

    def read_cutouts(self):
        batch_pref = ("MD", "TX", "NC")
        cutout_batchs = [
            x for x in Path(self.cutoutdir).glob("*")
            if x.name.startswith(batch_pref)
        ]
        cutout_csvs = [x for y in cutout_batchs for x in y.glob("*.csv")]
        cutoutsdf = pd.concat([
            pd.read_csv(x, low_memory=False) for x in cutout_csvs
        ]).reset_index(drop=True)
        return cutoutsdf

    def sort_cutouts(self):
        df = self.read_cutouts()
        # Filter by species
        if self.species:
            usda_symbol = self.species
            df = df[df.USDA_symbol.isin(usda_symbol)]
        log.info(f"{len(df)} cutouts after filter by species ({self.species})")

        # Filter using green sum
        if self.green_sum:
            gsmax = self.green_sum.max
            gsmin = self.green_sum.min
            df = df.loc[(df.green_sum <= gsmax) & (df.green_sum >= gsmin)]
        log.info(
            f"{len(df)} cutouts after filter by green_sum ({self.green_sum})")

        # Filter using cutout area
        if self.area:
            area = self.area
            desc = df["area"].describe()
            if area.min == "mean":
                df = df[df["area"] > desc.iloc[1]]
            if area.max == "mean":
                df = df[df["area"] < desc.iloc[1]]
            if area.min == 25:
                df = df[df["area"] > desc.iloc[4]]
            if area.max == 25:
                df = df[df["area"] < desc.iloc[4]]
            if area.min == 50:
                df = df[df["area"] > desc.iloc[5]]
            if area.max == 50:
                df = df[df["area"] < desc.iloc[5]]
            if area.min == 75:
                df = df[df["area"] > desc.iloc[6]]
            if area.max == 75:
                df = df[df["area"] < desc.iloc[6]]
        log.info(f"{len(df)} cutouts after filter by area ({self.area})")

        if len(df) == 0:
            log.error("No cutouts. Exiting.")
            exit(1)

        # Filter by border
        if self.extends_border is not None:
            df = df.loc[df.extends_border == self.extends_border]
            log.info(
                f"{len(df)} cutouts after filter by extend border ({self.extends_border})"
            )

        if self.is_primary is not None:
            # Replace with all including non_primary
            df = df[df["is_primary"] == self.is_primary]
            log.info(
                f"{len(df)} cutouts after filter by is_primary ({self.is_primary})"
            )

        return df

    # def save_csv(self):
    #     # Save csv
    #     csv_path = f"{cfg.workdir}/filtered_cutouts.csv"
    #     describe_csvpath = f"{job_dir}/description.csv"
    #     species_count_csvpath = f"{job_dir}/species_count.csv"

    #     # Save results, description, and count by species
    #     df.to_csv(csv_path, index=False)
    #     df.describe(include="all").to_csv(describe_csvpath)
    #     df.groupby(["common_name"]).count().sort_values(
    #         "blob_home",
    #         ascending=False)["blob_home"].to_csv(species_count_csvpath)
    #     print(len(df))
    # return df
