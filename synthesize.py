import json
import logging
from dataclasses import asdict
from multiprocessing import Pool, Process, cpu_count, freeze_support
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm, trange

from utils.datasets import SynthData, SynthImage
from utils.synth_utils import (SynthPipeline, clean_data, img2RGBA,
                               save_dataclass_json, transform_position)

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

    used_pot_positions = []
    used_cutouts = []
    used_pots = []
    used_backgrounds = []
    used_cutout_positions = []
    back_arr = back.array.copy()
    back_arr_mask = np.zeros_like(back_arr)

    pot_positions = syn.get_pot_positions()
    for pot, pot_position, cutout in zip(pots, pot_positions,cuts):
        # Get pot and info
        pot_arr = pot.array
        pot_arr = syn.transform(pot_arr)
        potshape = pot_arr.shape[:2]
        potshape = pot_arr.shape[:2]
        # Cutout data
        cutout_arr = cutout.array
        cutout_arr = syn.transform(cutout_arr)
        cutoutshape = cutout_arr.shape[:2]
        
        # Check coordinates for pot in top left corner
        topl_y, topl_x = transform_position(pot_position, potshape)  # to top left corner
        #Overlay pot on background
        syn.fore_str = "Overlay pot"
        
        potted, poty, potx = syn.overlay(topl_y, topl_x, pot_arr,
                                                 back_arr)
        
        # Get cutout position from pot position
        cutx, cuty = syn.center_on_background(poty, potx, potshape,
                                                cutoutshape)
        syn.fore_str = "Overlay cutout"
        potted, mask, _, _ = syn.overlay(cuty,
                                            cutx,
                                            cutout_arr,
                                            back_arr,
                                            mask=back_arr_mask)
        used_pot_positions.append([topl_y, topl_x, potshape])
        used_cutout_positions.append([cutx, cuty, cutoutshape])
        used_cutouts.append(cutout)
        used_pots.append(pot)
    used_backgrounds.append(back)
    # Save synth image and mask
    savepath, savemask = syn.save_synth(potted, mask[:, :, 0])

    # # Path info to save
    savestem = savepath.stem
    savepath = "/".join(savepath.parts[-2:])
    savemask = "/".join(savemask.parts[-2:])
    data_root = Path(cfg.synth.synthdir).name
    # To SynthImage dataclass
    synimage = SynthImage(data_root=data_root,
                            synth_path=savepath,
                            synth_maskpath=savemask,
                            pots=used_pots,
                            background=used_backgrounds,
                            cutouts=used_cutouts)
    # Clean dataclass
    data_dict = asdict(synimage)
    synimage = clean_data(data_dict)

    # To json
    jsonpath = Path(syn.json_dir, savestem + ".json")
    save_dataclass_json(synimage, jsonpath)


def main(cfg: DictConfig) -> None:

    # def pipeline():
    # Create synth data container
    data = SynthData(synthdir=cfg.synth.synthdir,
                     background_dir=cfg.synth.backgrounddir,
                     pot_dir=cfg.synth.potdir,
                     cutout_dir=cfg.data.cutoutdir,
                     cutout_csv=cfg.data.csvpath)

    num_images = cfg.synth.count

    all_args = []
    for num in range(num_images):
        num_cuts = np.random.randint(1, 10, size=1)
        cuts = np.random.choice(data.cutouts, num_cuts, replace=False)
        pots = np.random.choice(data.pots, num_cuts, replace=True)
        back = np.random.choice(data.backgrounds, 1)[0]
        
        all_args.append((cuts, pots, back, cfg))

    if cfg.synth.multiprocess:
        procs = cpu_count() - 5
        with Pool(procs) as pool:
            pool.starmap(synth_image, all_args)
    else:
        for cuts, pots, back, cfg in tqdm(all_args):
            synth_image(cuts, pots, back, cfg)
