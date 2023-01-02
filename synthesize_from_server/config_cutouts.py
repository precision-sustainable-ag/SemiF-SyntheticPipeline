import shutil
from pathlib import Path

from omegaconf import DictConfig
import json
from synth_utils.config_utils import read_cutouts, sort_cutouts
from pprint import pprint


def main(cfg: DictConfig) -> None:
    """ Creates csv file with all configs to pul from based on cutout config yaml."""
    # Using species proportions
    alldf = read_cutouts(cfg.data.cutoutdir)
    df = sort_cutouts(alldf, cfg, save_csv=cfg.cutouts.save_csv)
    # print(cfg.job.jobdir)
    # srcdir = "/home/weedsci/matt/SemiF-AnnotationPipeline/data/semifield-cutouts/"
    # dstdir = Path(cfg.job.jobdir, "cutouts")
    # try:
    #     dstdir.mkdir()
    # except Exception as e:
    #     print(e)

    # srccutouts = srcdir + df["cutout_path"]

    # for src in srccutouts:
    #     src_parent = Path(src).parent
    #     stem = Path(src).stem
    #     srcname = Path(src).name

    #     dstpath = Path(dstdir, srcname)

    #     jsonsrcpath = Path(src_parent, stem + ".json")
    #     jsondstpath = Path(dstdir, stem + ".json")
    #     masksrcpath = Path(src_parent, stem + "_mask.png")
    #     maskdstpath = Path(dstdir, stem + "_mask.png")
    #     imgsrcpath = Path(src_parent, stem + ".jpg")
    #     imgdstpath = Path(dstdir, stem + ".jpg")

    #     # print(jsonsrcpath.is_file())
    #     # # print(jsondstpath.is_file())
    #     # print(masksrcpath.is_file())
    #     # # print(maskdstpath.is_file())
    #     # print(imgsrcpath.is_file())
    #     # # print(imgdstpath.is_file())

    #     shutil.copy2(src, dstpath)
    #     shutil.copy2(jsonsrcpath, jsondstpath)
    #     shutil.copy2(masksrcpath, maskdstpath)
    #     shutil.copy2(imgsrcpath, imgdstpath)
