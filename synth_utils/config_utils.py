import json
import logging
import os
import sys

from tqdm import tqdm

sys.path.append("/home/weedsci/matt/SemiF-AnnotationPipeline")
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from semif_utils.datasets import Cutout

log = logging.getLogger(__name__)


def get_cutout_meta(path):
    with open(path) as f:
        j = json.load(f)
        cutout = Cutout(**j)
    return cutout


def read_cutouts(cutoutdir):
    batch_pref = ("MD", "TX", "NC")
    cutout_batchs = [
        x for x in Path(cutoutdir).glob("*") if x.name.startswith(batch_pref)
    ]
    cutout_csvs = [x for y in cutout_batchs for x in y.glob("*.csv")]
    cutoutsdf = pd.concat([pd.read_csv(x) for x in cutout_csvs
                          ]).reset_index(drop=True)
    return cutoutsdf


def sort_cutouts(df, cfg, save_csv=False):
    log.info(f"Starting cutout number:\t\t\t {len(df)}")
    ogdf = df.copy()
    scfg = cfg.cutouts.species
    present_spec = df.common_name.unique()

    # Check to make sure only choosing species that have data in cutouts config
    spec = [
        x for x in scfg.keys()
        if (scfg[x] != None) & (scfg[x] != False) & (scfg[x] != 0)
    ]
    assert cfg.cutouts.balanced or spec or (
        cfg.cutouts.sample_size
        == "all"), "You must provide species proportions if 'balanced' is False"
    assert not (
        cfg.cutouts.balanced and spec
    ), "You must provide species proportions or 'balanced' must be True\nbut not both."
    if cfg.cutouts.sample_size == "all":
        assert not spec, "If sample size is 'all', you cannot provide species proportions."

    if spec:
        spec_props = {spe: scfg[spe] for spe in spec}
        # Filter by species
        df = df.loc[df.USDA_symbol.isin(spec)]
        log.info(f"Filter by species:\t\t\t {len(df)}")

    # Filter is_primary
    df = df.loc[df.is_primary == cfg.cutouts.features.is_primary]
    log.info(
        f"Filter by is_primary ({cfg.cutouts.features.is_primary}):\t\t\t {len(df)}"
    )

    # Filter exprint(f"Filtering by species... \n{spec}")tends_border if possible
    df = df.loc[df.extends_border == cfg.cutouts.features.extends_border]
    log.info(
        f"Filter by extends_border ({cfg.cutouts.features.extends_border}):\t\t {len(df)}"
    )

    # Filter using green sum
    if cfg.cutouts.features.green_sum:
        gsmax = cfg.cutouts.features.green_sum.max
        gsmin = cfg.cutouts.features.green_sum.min
        df = df.loc[(df.green_sum <= gsmax) & (df.green_sum >= gsmin)]
        log.info(
            f"Filter by green_sum ({cfg.cutouts.features.green_sum.min, cfg.cutouts.features.green_sum.max}):\t {len(df)}"
        )
    # Filter sing cutout area
    if cfg.cutouts.features.area:
        # print(df["area"].describe())
        area = cfg.cutouts.features.area
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
        # print(df["area"].describe())
        log.info(f"Filter by area:\t\t\t\t {len(df)}")

    # Filter using DAP
    if cfg.cutouts.features.dap:
        dapmax = cfg.cutouts.features.dap.max
        dapmin = cfg.cutouts.features.dap.min
        df = df.loc[(df.dap <= dapmax) & (df.dap >= dapmin)]
        log.info(
            f"Filter by dap ({cfg.cutouts.features.dap.min, cfg.cutouts.features.dap.max}):\t\t\t {len(df)}"
        )

    # Sample with replacement based on config species proportions and sample size
    if spec and (type(cfg.cutouts.sample_size) is int):
        sample_size = len(ogdf)
        df['weights'] = df.USDA_symbol.map(spec_props)
        df = df.sample(sample_size,
                       weights='weights',
                       ignore_index=True,
                       replace=True,
                       random_state=42).reset_index(drop=True)
        log.info(
            f"Cutouts after species proportions and sample_size ({cfg.cutouts.sample_size}): {len(df)}"
        )

    # Sample balanced
    elif cfg.cutouts.balanced and (type(cfg.cutouts.sample_size) is int):
        sample_size = cfg.cutouts.sample_size
        df = df.groupby('USDA_symbol').apply(lambda x: x.sample(
            sample_size, replace=True)).reset_index(drop=True)
        log.info(
            f"Cutouts after balanced sampling and sample_size ({cfg.cutouts.sample_size}): {len(df)}"
        )

    if save_csv:
        csv_path = f"{cfg.job.jobdir}/filtered_cutouts.csv"
        describe_csvpath = f"{cfg.job.jobdir}/description.csv"
        species_count_csvpath = f"{cfg.job.jobdir}/species_count.csv"

        # Save results, description, and count by species
        df.to_csv(csv_path, index=False)
        df.describe(include="all").to_csv(describe_csvpath)
        df.groupby(["common_name"]).count().sort_values(
            "blob_home",
            ascending=False)["blob_home"].to_csv(species_count_csvpath)
    log.info(f"Final number of cutouts:\t\t\t {len(df)}")
    print("\nDone filtering cutouts...")
    return df


def cutoutmeta2csv(cutoutdir, batch_id, csv_savepath, save_df=True):
    # Get all json files
    metas = [x for x in Path(cutoutdir, batch_id).glob("*.json")]
    cutouts = []
    for meta in tqdm(metas):
        # Get dictionaries
        cutout = asdict(get_cutout_meta(meta))

        row = cutout["cutout_props"]

        cls = cutout["cls"]
        colors = cutout["cls"]["rgb"]
        # Extend nested dicts to single column header
        for ro in row:
            rec = {ro: row[ro]}
            cutout.update(rec)
            for cl in cls:
                spec = {cl: cls[cl]}
                if cl == "rgb":
                    r = {"r": str(spec["rgb"][0])}
                    g = {"g": str(spec["rgb"][1])}
                    b = {"b": str(spec["rgb"][2])}
                    cutout.update(r)
                    cutout.update(g)
                    cutout.update(b)

                cutout.update(spec)

        # Remove duplicate nested dicts
        cutout.pop("cutout_props")
        cutout.pop("cls")
        cutout.pop("rgb")
        cutout.pop("local_contours")
        # Create and append df
        cutdf = pd.DataFrame(cutout, index=[0])
        cutouts.append(cutdf)
    # Concat and reset index of main df
    cutouts_df = pd.concat(cutouts)
    cutouts_df = cutouts_df.reset_index()
    cutouts_df.drop(columns="index", inplace=True)
    cutouts_df.sort_values(by=['image_id', 'cutout_num'],
                           axis=0,
                           ascending=[True, True],
                           inplace=True,
                           kind='quicksort',
                           na_position='first',
                           ignore_index=True,
                           key=None)
    # Save dataframe
    if save_df:
        cutouts_df.to_csv(csv_savepath, index=False)
    return cutouts_df


# Test
# if __name__ == "__main__":
#     path = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts/MD_2022-07-05/MD_Row-10_1657032952_6.json"
#     cutoutdir = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts"
#     batch_id = "MD_2022-07-06"
#     csv_path = "MD_2022-07-06.csv"
#     cutoutmeta2csv(cutoutdir, batch_id, csv_path)
