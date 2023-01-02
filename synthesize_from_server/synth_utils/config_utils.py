import json
import logging
from pprint import pprint
from tqdm import tqdm

# sys.path.append("/home/weedsci/matt/SemiF-AnnotationPipeline")
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
    cutoutsdf = pd.concat([
        pd.read_csv(x, low_memory=False) for x in cutout_csvs
    ]).reset_index(drop=True)
    return cutoutsdf


def sort_cutouts(df, cfg, save_csv=False):
    # Filter by species
    if cfg.cutouts.species:
        usda_symbol = cfg.cutouts.species
        df = df[df.USDA_symbol.isin(usda_symbol)]
    log.info(
        f"{len(df)} cutouts after filter by species ({cfg.cutouts.species})")

    # Filter using green sum
    if cfg.cutouts.features.green_sum:
        gsmax = cfg.cutouts.features.green_sum.max
        gsmin = cfg.cutouts.features.green_sum.min
        df = df.loc[(df.green_sum <= gsmax) & (df.green_sum >= gsmin)]
    log.info(
        f"{len(df)} cutouts after filter by green_sum ({cfg.cutouts.features.green_sum})"
    )

    # Filter using cutout area
    if cfg.cutouts.features.area:
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
    log.info(
        f"{len(df)} cutouts after filter by area ({cfg.cutouts.features.area})"
    )

    if len(df) == 0:
        log.error("No cutouts. Exiting.")
        exit(1)

    # Filter by border
    if cfg.cutouts.features.extends_border is not None:
        df = df.loc[df.extends_border == cfg.cutouts.features.extends_border]
        log.info(
            f"{len(df)} cutouts after filter by extend border ({cfg.cutouts.features.extends_border})"
        )

    if cfg.cutouts.features.is_primary is not None:
        # Replace with all including non_primary
        df = df[df["is_primary"] == cfg.cutouts.features.is_primary]
        log.info(
            f"{len(df)} cutouts after filter by is_primary ({cfg.cutouts.features.is_primary})"
        )

    # if cfg.cutouts.replace:
    #     samp_size = cfg.cutouts.sample_size
    #     ndf = df.copy()
    #     # Filter by border
    #     df = df.loc[df.extends_border == cfg.cutouts.features.extends_border]
    #     # Replace with all including non_primary
    #     df = df[df["is_primary"] == cfg.cutouts.features.is_primary]
    #     prim = len(df)
    #     # Sample size
    #     pdf = ndf.sample(n=samp_size - prim,
    #                      replace=True,
    #                      random_state=cfg.cutouts.seed)
    #     df = pd.concat([pdf, df])
    #     log.info(f"{len(df)} cutouts after filter by replace")
    # else:
    #     # Filter by border
    #     df = df.loc[df.extends_border == cfg.cutouts.features.extends_border]
    #     # Filter is_primary
    #     df = df[df["is_primary"] == cfg.cutouts.features.is_primary]
    #     log.info(f"{len(df)} cutouts after filter by not replace")

    # Save csv
    if save_csv:
        job_dir = cfg.job.jobdir
        Path(job_dir).mkdir(exist_ok=True, parents=True)
        csv_path = f"{job_dir}/filtered_cutouts.csv"
        describe_csvpath = f"{job_dir}/description.csv"
        species_count_csvpath = f"{job_dir}/species_count.csv"

        # Save results, description, and count by species
        df.to_csv(csv_path, index=False)
        df.describe(include="all").to_csv(describe_csvpath)
        df.groupby(["common_name"]).count().sort_values(
            "blob_home",
            ascending=False)["blob_home"].to_csv(species_count_csvpath)
    print(len(df))
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

        # Color distribution data
        cd = cutout["cutout_props"]["color_distribution"]
        nd = dict()
        for idx, c in enumerate(cd):
            nd["hex_" + str(idx)] = c["hex"]
            nd["rgb_" + str(idx)] = c["rgb"]
            nd["occurences_" + str(idx)] = int(c["occurence"])
        # exit(1)
        cutout.update(nd)

        # Descriptive stats
        ds = cutout["cutout_props"]["descriptive_stats"]
        nd = dict()
        for d in ds:
            chan_suff = d.split("_")[-1]
            for chan in ds[d]:

                nd[chan_suff + "_" + chan] = ds[d][chan]
        cutout.update(nd)

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
        cutout.pop("descriptive_stats")
        cutout.pop("color_distribution")
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
    if save_df:
        cutouts_df.to_csv(csv_savepath, index=False)
    return cutouts_df