import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from utils.datasets import SynthData, SynthImage
from utils.overlay_utils import (create_yolo_labels, overlay_cutout2pot,
                                 overlay_multicutouts2background,
                                 overlay_multicutouts2pot,
                                 prep_cutout2pot_contents,
                                 prep_multicutout2pot_contents,
                                 process_contents)
from utils.synth_utils import SynthPipeline
from utils.utils import clean_data, rand_positions, save_dataclass_json

log = logging.getLogger(__name__)


def synth_image(cuts, pots, back, cfg):
    """ Operates to produce single image

    Args:
        cuts (_type_): _description_
        pots (_type_): _description_
        back (_type_): _description_
        cfg (_type_): _description_
    """
    # Instantiate pipeline tools
    syn = SynthPipeline(cfg)

    used_backgrounds = []
    back_arr = back.array.copy()
    back_arr_mask = np.zeros_like(back_arr)
    back_hw = back_arr.shape[:2]

    mode = cfg.synth.mode
    if mode == "symmetric":
        pot_positions = syn.get_pot_positions()
        if cfg.synth.multicutouts:
            min_multicut_dist = cfg.synth.min_multicutout_dist
            used_cutouts, used_pots, potted, mask = overlay_multicutouts2pot(
                cuts, pots, syn, back_arr, back_arr_mask, back_hw,
                pot_positions, min_multicut_dist)
        else:
            used_cutouts, used_pots, potted, mask = overlay_cutout2pot(
                cuts, pots, syn, back_arr, back_arr_mask, back_hw,
                pot_positions)

    elif mode == "random":
        pot_positions = rand_positions(0, back_hw[1], 0, back_hw[0], len(pots),
                                       cfg.synth.min_pot_dist)
        if cfg.synth.multicutouts:
            min_multicut_dist = cfg.synth.min_multicutout_dist
            used_cutouts, used_pots, potted, mask = overlay_multicutouts2pot(
                cuts, pots, syn, back_arr, back_arr_mask, back_hw,
                pot_positions, min_multicut_dist)

        else:
            used_cutouts, used_pots, potted, mask = overlay_cutout2pot(
                cuts, pots, syn, back_arr, back_arr_mask, back_hw,
                pot_positions)

    elif mode == "no pots":
        num_total_cutouts = sum([len(x) for x in cuts])
        multicut_pos = rand_positions(0, back_hw[1], 0, back_hw[0],
                                      num_total_cutouts,
                                      cfg.synth.no_pots.min_cut_dist)
        used_cutouts, used_pots, potted, mask = overlay_multicutouts2background(
            cuts, syn, back_arr, back_arr_mask, back_hw, multicut_pos)

    used_backgrounds.append(back)

    # Save synth image and mask
    savepath, savemask = syn.save_synth(potted, mask[:, :, :3])

    # # Path info to save
    savestem = savepath.stem
    savepath = "/".join(savepath.parts[-2:])
    savemask = "/".join(savemask.parts[-2:])
    data_root = Path(cfg.synth.synthdir).name
    # To SynthImage dataclass
    synimage = SynthImage(
        data_root=data_root,
        synth_path=savepath,
        synth_maskpath=savemask,
        pots=used_pots,
        background=used_backgrounds,
        # synth_cut_norm_xy=used_cutout_positions,
        synthimg_pix_hwc=potted.shape,
        cutouts=used_cutouts)
    # Clean dataclass
    data_dict = asdict(synimage)
    synimage = clean_data(data_dict)

    # Save to json metadata
    jsonpath = Path(syn.json_dir, savestem + ".json")
    save_dataclass_json(synimage, jsonpath)


def main(cfg: DictConfig) -> None:

    # Create synth data container
    log.info("Creating synthetic dataclasses.")
    data = SynthData(synthdir=cfg.synth.synthdir,
                     background_dir=cfg.synth.backgrounddir,
                     pot_dir=cfg.synth.potdir,
                     cutout_dir=cfg.data.cutoutdir,
                     cutout_csv=cfg.data.csvpath)

    # Data prep
    if cfg.synth.multicutouts:
        log.info(
            "Organizing multiple cutouts per pot, pots, and backgrounds for each image."
        )
        synth_contents = prep_multicutout2pot_contents(cfg, data)
    else:
        log.info("Organizing cutouts, pots, and backgrounds for each image.")
        synth_contents = prep_cutout2pot_contents(cfg, data)

    # Multi- or single-processing
    log.info("Starting synthetic image processing.")
    cfg = process_contents(cfg, synth_contents, synth_image)

    # YOLO labels
    if cfg.synth.export_yolo_labels:
        log.info("Creating Yolo formatted labels.")
        create_yolo_labels(cfg)