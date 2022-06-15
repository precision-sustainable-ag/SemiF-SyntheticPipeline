import random
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from tqdm import trange

from semif_utils.datasets import POTMAPS, SynthDataContainer, SynthImage
from semif_utils.synth_utils import (SynthPipeline, center_on_background,
                                     clean_data, get_cutout_dir, img2RGBA,
                                     save_dataclass_json, transform_position)


def main(cfg: DictConfig) -> None:
    cutout_dir = get_cutout_dir(cfg.data.batchdir, cfg.data.cutoutdir)

    # Create synth data container
    data = SynthDataContainer(synthdir=cfg.data.synthdir,
                              db_db=cfg.general.db,
                              from_db=cfg.general.from_database,
                              cutout_dir=cutout_dir,
                              pot_dir=cfg.synth.potdir,
                              background_dir=cfg.synth.backgrounddir)

    # Call pipeline
    syn = SynthPipeline(data, cfg)

    # Config pot placement info
    potmaps = random.choice(POTMAPS)
    jitter_tuple = list(cfg.synth.pot_jitter)
    pot_jitter = random.randint(jitter_tuple[0], jitter_tuple[1])

    for cnt in trange(cfg.synth.count):

        # Get synth data samples
        back = syn.get_back()
        back = cv2.resize(back.array, (9592, 6368))
        back = cv2.cvtColor(back, cv2.COLOR_BGR2RGBA).copy()

        # Get pots
        pots = syn.get_pots(len(potmaps))

        # Get cutouts and mask
        cutouts = syn.get_cutouts(len(potmaps))
        cutout_zero_mask = np.zeros_like(back)

        # Iterate using number of pots
        pot_positions = []
        for potidx in range(len(potmaps)):

            # Get pot and info
            pot_position = potmaps[potidx]  # center x,y
            pot = random.choice(pots)
            pot_array = pot.array
            potshape = pot_array.shape

            # Check coordinates for pot
            x, y = transform_position(pot_position, potshape,
                                      pot_jitter)  # to top left corner
            x, y = syn.check_overlap(x, y, potshape, pot_positions)

            #Overlay pot on background
            potted, poty, potx = syn.overlay(y, x, pot_array, back)
            pot_positions.append([y, x, potshape])

            # Convert to RGBA cutout
            cutout = img2RGBA(random.choice(cutouts).array)
            cutoutshape = cutout.shape

            # Get cutout position from pot position
            cutx, cuty = center_on_background(poty, potx, potshape,
                                              cutoutshape)
            # Overlay cutout on pot
            potted, mask, _, _ = syn.overlay(cuty,
                                             cutx,
                                             cutout,
                                             potted,
                                             mask=cutout_zero_mask)

        # Save synth image and mask
        savepath, savemask = syn.save_synth(potted, mask[:, :, 0])

        # Path info to save
        savestem = savepath.stem
        savepath = "/".join(savepath.parts[-2:])
        savemask = "/".join(savemask.parts[-2:])
        data_root = Path(cfg.data.synthdir).name

        # To SynthImage dataclass
        synimage = SynthImage(data_root=data_root,
                              synth_path=savepath,
                              synth_maskpath=savemask,
                              pots=syn.pots,
                              background=syn.back,
                              cutouts=syn.cutouts)
        # Clean dataclass
        data_dict = asdict(synimage)
        synimage = clean_data(data_dict)

        # DC to database
        if cfg.general.save_to_database:
            connect = Connect.get_connection()
            db = getattr(connect, cfg.general.db)
            to_db(db, str(data_root), synimage)

        # To json
        jsonpath = Path(syn.json_dir, savestem + ".json")
        save_dataclass_json(synimage, jsonpath)
