import json
import os
import sys

from tqdm import tqdm

sys.path.append("/home/weedsci/matt/SemiF-AnnotationPipeline")
from dataclasses import asdict
from pathlib import Path

import pandas as pd
from semif_utils.datasets import Cutout


def get_cutout_meta(path):
    with open(path) as f:
        j = json.load(f)
        cutout = Cutout(**j)
    return cutout


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
                # for idx, color in enumerate(colors):
                #     colo = {color: colors[idx]}
                #     cutout.update(colo)

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
    # Save dataframe
    if save_df:
        cutouts_df.sort_values(by=['image_id', 'cutout_num'],
                               axis=0,
                               ascending=[True, True],
                               inplace=True,
                               kind='quicksort',
                               na_position='first',
                               ignore_index=True,
                               key=None)
        cutouts_df.to_csv(csv_savepath, index=False)

    return cutouts_df


# Test
# if __name__ == "__main__":
#     path = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts/MD_2022-07-05/MD_Row-10_1657032952_6.json"
#     cutoutdir = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts"
#     batch_id = "MD_2022-07-06"
#     csv_path = "MD_2022-07-06.csv"
#     cutoutmeta2csv(cutoutdir, batch_id, csv_path)
