import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import trange
import logging

from semif_utils.datasets import SynthData, SynthImage
from semif_utils.synth_utils import (SynthPipeline, img2RGBA, clean_data,
                                     save_dataclass_json, transform_position)

log = logging.getLogger(__name__)


def main(cfg: DictConfig) -> None:
    # Create synth data container
    data = SynthData(synthdir=cfg.synth.synthdir,
                     background_dir=cfg.synth.backgrounddir,
                     pot_dir=cfg.synth.potdir,
                     cutout_dir=cfg.synth.cutout_batchdir)
    # Call pipeline
    syn = SynthPipeline(data, cfg)
    used_backgrounds = []
    for cnt in trange(cfg.synth.count):

        # Get single background
        syn.get_back()
        syn.get_pot_positions()
        back_arr = syn.back.array.copy()
        back_arr_mask = np.zeros_like(back_arr)

        # Iterate using number of pots
        pot_positions = []
        used_cutouts = []
        used_pots = []

        for potidx in range(len(syn.pot_positions)):
            # Get pot and info
            pot_position = syn.pot_positions[potidx]  # center (y,x)
            # Get single pot
            syn.get_pot()
            pot_arr = syn.pot.array
            syn.prep_cutout()
            cutout_arr = syn.cutout.array

            cutoutshape = syn.cutout.array.shape[:2]
            potshape = syn.pot.pot_ht, syn.pot.pot_wdt

            # Check coordinates for pot in top left corner
            topl_y, topl_x = transform_position(pot_position,
                                                potshape)  # to top left corner
            # topl_y, topl_x = syn.check_overlap(topl_y, topl_x, potshape,
            #    pot_positions)
            #Overlay pot on background
            syn.fore_str = "Overlay pot"
            potted, poty, potx = syn.overlay(topl_y, topl_x, pot_arr, back_arr)
            # Get cutout position from pot position
            cutx, cuty = syn.center_on_background(poty, potx, potshape,
                                                  cutoutshape)
            syn.fore_str = "Overlay cutout"
            potted, mask, _, _ = syn.overlay(cuty,
                                             cutx,
                                             cutout_arr,
                                             back_arr,
                                             mask=back_arr_mask)

            pot_positions.append([topl_y, topl_x, potshape])
            used_cutouts.append(syn.cutout)
            used_pots.append(syn.pot)
        used_backgrounds.append(syn.back)

        # Save synth image and mask
        savepath, savemask = syn.save_synth(potted, mask[:, :, 0])

        # Path info to save
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
