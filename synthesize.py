import logging
from dataclasses import asdict
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from omegaconf import DictConfig
from tqdm import tqdm

from utils.datasets import SynthData, SynthImage
from utils.synth_utils import SynthPipeline
from utils.utils import (clean_data, meta2yolo_prep, metadata2yolo_labels,
                         rand_positions, save_dataclass_json,
                         transform_position)

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

    used_cutouts = []
    used_pots = []
    used_backgrounds = []

    back_arr = back.array.copy()
    back_arr_mask = np.zeros_like(back_arr)
    back_hw = back_arr.shape[:2]
    mode = cfg.synth.mode
    if mode == "symmetric":
        pot_positions = syn.get_pot_positions()
    elif mode == "random":
        pot_positions = rand_positions(0, back_hw[1], 0, back_hw[0], len(pots), cfg.synth.min_pot_dist)
    
    for pot, pot_position, cutout in zip(pots, pot_positions,cuts):
        cutout.local_contours = None
        rgb = cutout.cls["rgb"]
        bgra = [rgb[2], rgb[1], rgb[0]]
        
        # Get pot and info
        pot_arr = pot.array
        pot_arr = syn.transform(pot_arr)
        potshape = pot_arr.shape[:2]
        potshape = pot_arr.shape[:2]
    
        # Check coordinates for pot in top left corner
        topl_y, topl_x = transform_position(pot_position, potshape)  # to top left corner
    
        #Overlay pot on background
        syn.fore_str = "Overlay pot"
        potted, poty, potx = syn.overlay(topl_y, topl_x, pot_arr,
                                                back_arr)
        
        # Cutout data
        cutout_arr = cutout.array
        cutout_arr = syn.transform(cutout_arr)
        cutoutshape = cutout_arr.shape[:2]
        cutout.synth_hwc = cutout_arr.shape
        
        # Check coordinates for pot in top left corner
        # cut_topl_y, cut_topl_x = transform_position(pot_position, cutoutshape)  # to top left corner
        # Get cutout position from pot position
        cutx, cuty = syn.center_on_background(topl_y, topl_x, potshape,
                                                cutoutshape)
        # Overlay cutout on pot and background
        syn.fore_str = "Overlay cutout"
        potted, mask, cut_tly, cut_tlx = syn.overlay(cuty, 
                                            cutx,
                                            cutout_arr,
                                            back_arr,
                                            mask=back_arr_mask,
                                            bgra=bgra)
        
        # Place color in used color list (palette)
        syn.bgra_palette.append(bgra)
        
        # Add yolov labels
        xnorm, ynorm = float(cut_tlx/ back_hw[1]), float(cut_tly/back_hw[0])
        w, h = float(cutout_arr.shape[1]/ back_hw[1]), float(cutout_arr.shape[0]/ back_hw[0])
        # Write bbox to cutout dataclass
        cutout.synth_norm_xywh = xnorm, ynorm, w, h
        used_cutouts.append(cutout)
        used_pots.append(pot)
    used_backgrounds.append(back)

    # Save synth image and mask
    savepath, savemask = syn.save_synth(potted, mask[:,:,:3])

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
    log.info("Organizing cutouts, pots, and backgrounds for each image.")
    synth_contents = prep_synth_contents(cfg, data)
    
    # Multi- or single-processing
    log.info("Starting synthetic image processing.")
    cfg = process_contents(cfg, synth_contents)

    # YOLO labels
    if cfg.synth.export_yolo_labels:
        log.info("Creating Yolo formatted labels.")
        create_yolo_labels(cfg)









def create_yolo_labels(cfg):
    jsonpaths = sorted(list(Path(cfg.synth.savedir, "metadata").glob("*.json")))
    imgpaths =  sorted(list(Path(cfg.synth.savedir, "images").glob("*.png")))
    data = []
    for imgp, jsonp in zip(imgpaths, jsonpaths):
        if imgp.stem != jsonp.stem:
            log.error("Metadata (json) and image (png) file stems do not match. Yolo label processing failed.")
        data.append(meta2yolo_prep(jsonp, imgp))
    
    metadata2yolo_labels(cfg.synth.savedir, data)



def process_contents(cfg, synth_contents):
    if cfg.synth.multiprocess:
        log.info("Starting multiprocessing.")
        procs = cpu_count() - 5
        with Pool(procs) as pool:
            pool.starmap(synth_image, synth_contents)
    else:
        log.info("Starting single image processing.")

        for cuts, pots, back, cfg in tqdm(synth_contents):
            synth_image(cuts, pots, back, cfg)
    return cfg

def prep_synth_contents(cfg, data):
    num_images = cfg.synth.count
    contents = []
    for num in range(num_images):
        # Randomized
        back = np.random.choice(data.backgrounds, 1)[0]

        min_cutouts = cfg.synth.num_cutouts.min
        max_cutouts = cfg.synth.num_cutouts.max
        num_cuts = np.random.randint(min_cutouts, max_cutouts, size=1)
        cuts = np.random.choice(data.cutouts, num_cuts, replace=True)
        
        if cfg.synth.mode == "symmetric":
            num_pots = num_cuts
        elif cfg.synth.mode == "random":
            num_pots = np.random.randint(cfg.synth.min_pots,cfg.synth.max_pots , size=1)
        pots = np.random.choice(data.pots,num_pots, replace=True) if cfg.synth.use_pots else []
        
        contents.append((cuts, pots, back, cfg))
    return contents