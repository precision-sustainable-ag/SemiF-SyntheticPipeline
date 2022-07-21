import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from semif_utils.datasets import SynthDataContainer, SynthImage
from semif_utils.synth_utils import (SynthPipeline, center_on_background,
                                     clean_data, get_cutout_dir, img2RGBA,
                                     rand_pot_grid, save_dataclass_json,
                                     transform_position)


def main(cfg: DictConfig) -> None:
    cutout_dir = get_cutout_dir(cfg.data.batchdir, cfg.data.cutoutdir)

    # Create synth data container
    data = SynthDataContainer(synthdir=cfg.synth.synthdir,
                              batch_id=cfg.general.batch_id,
                              background_dir=cfg.synth.backgrounddir,
                              pot_dir=cfg.synth.potdir,
                              cutout_dir=cfg.synth.cutout_batchdir)
    # Call pipeline
    syn = SynthPipeline(data, cfg)

    # jitter_tuple = list(cfg.synth.pot_jitter)
    # pot_jitter = random.randint(jitter_tuple[0], jitter_tuple[1])
    used_backgrounds = []
    for cnt in trange(cfg.synth.count):
        # Config pot placement info
        potmaps = rand_pot_grid((6368, 9560))

        # Get synth data samples
        back = syn.get_back()
        # back = cv2.resize(back.array, (9592, 6368))
        back_arr = cv2.cvtColor(back.array, cv2.COLOR_BGR2BGRA).copy()
        back_h, back_w = back_arr.shape[:2]

        # Get pots
        pots = syn.get_pots(len(potmaps))

        # Get cutouts and mask
        cutouts = syn.get_cutouts(len(potmaps))
        cutout_zero_mask = np.zeros_like(back_arr)

        # Iterate using number of pots
        pot_positions = []
        used_cutouts = []

        used_pots = []
        for potidx in range(len(potmaps)):

            # Get pot and info
            pot_position = potmaps[potidx]  # center (y,x)
            print("Pot map ", potmaps)
            print("Index: ", potidx)
            print("\nPot postiion 1", pot_position)
            pot = random.choice(pots)
            pot_array = pot.array
            pot_array = syn.transform(pot_array)
            potshape = pot_array.shape

            # Check coordinates for pot in top left corner
            topl_y, topl_x = transform_position(pot_position, potshape,
                                                0)  # to top left corner
            print("Top left pot position 2 ", (topl_y, topl_x))
            topl_y, topl_x = syn.check_overlap(topl_y, topl_x, potshape,
                                               pot_positions)
            print("Pot position after overlap check (3) ", (topl_y, topl_x))
            if topl_y > back_h or topl_x > back_w:
                continue
            #Overlay pot on background
            potted, poty, potx = syn.overlay(topl_y, topl_x, pot_array,
                                             back_arr)
            if poty > back_h or potx > back_w:
                continue
            print("Pot position after overlay (4) ", (poty, potx))
            pot_positions.append([topl_y, topl_x, potshape])
            print("Pot overlay successful\n")
            print("Starting cutout processing...\n")
            # Convert to RGBA cutout
            cutout = random.choice(cutouts)
            cutout_arr = img2RGBA(cutout.array)
            cutout_arr = syn.transform(cutout_arr)
            cutoutshape = cutout_arr.shape

            # Get cutout position from pot position
            cutx, cuty = center_on_background(poty, potx, potshape,
                                              cutoutshape)
            if cuty > back_h or cutx > back_w:
                continue
            # Overlay cutout on pot
            potted, mask, _, _ = syn.overlay(cuty,
                                             cutx,
                                             cutout_arr,
                                             potted,
                                             mask=cutout_zero_mask)
            used_cutouts.append(cutout)
            used_pots.append(pot)
        used_backgrounds.append(back)

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
