import logging
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from tqdm import tqdm

from utils.utils import (meta2yolo_prep, metadata2yolo_labels,
                         rand_multicut_positions, transform_position)

log = logging.getLogger(__name__)


def overlay_cutout2pot(cuts, pots, syn, back_arr, back_arr_mask, back_hw,
                       pot_positions):

    used_cutouts = []
    used_pots = []

    for pot, pot_position, cutout in zip(pots, pot_positions, cuts):
        cutout.local_contours = None
        rgb = cutout.cls["rgb"]
        bgra = [rgb[2], rgb[1], rgb[0]]

        # Get pot and info
        pot_arr = pot.array
        pot_arr = syn.transform(pot_arr)
        potshape = pot_arr.shape[:2]
        potshape = pot_arr.shape[:2]

        # Check coordinates for pot in top left corner
        topl_y, topl_x = transform_position(pot_position,
                                            potshape)  # to top left corner

        #Overlay pot on background
        syn.fore_str = "Overlay pot"
        potted, poty, potx = syn.overlay(topl_y, topl_x, pot_arr, back_arr)

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
        xnorm, ynorm = float(cut_tlx / back_hw[1]), float(cut_tly / back_hw[0])
        w, h = float(cutout_arr.shape[1] / back_hw[1]), float(
            cutout_arr.shape[0] / back_hw[0])
        # Write bbox to cutout dataclass
        cutout.synth_norm_xywh = xnorm, ynorm, w, h
        used_cutouts.append(cutout)
        used_pots.append(pot)
    return used_cutouts, used_pots, potted, mask


def overlay_multicutouts2background(multicuts, syn, back_arr, back_arr_mask,
                                    back_hw, multicut_pos):

    used_cutouts = []
    used_pots = []
    multicuts = [num for sublist in multicuts for num in sublist]
    # Overlaying multiple cutouts on a single pot
    for idx, cutout in enumerate(multicuts):

        cutout.local_contours = None
        rgb = cutout.cls["rgb"]
        bgra = [rgb[2], rgb[1], rgb[0]]
        # Cutout data
        cutout_arr = cutout.array
        cutout_arr = syn.transform(cutout_arr)
        cutoutshape = cutout_arr.shape[:2]
        cutout.synth_hwc = cutout_arr.shape

        # Get cutout position from multi pot position
        # multi_topl_x = multicut_pos[idx][0]
        # multi_topl_y = multicut_pos[idx][1]
        cut_position = [multicut_pos[idx][0], multicut_pos[idx][1]]
        # Check coordinates for pot in top left corner
        multi_topl_y, multi_topl_x = transform_position(cut_position, back_hw)

        # Get cutout center position
        cutx, cuty = syn.center_on_background(multi_topl_y, multi_topl_x,
                                              back_hw, cutoutshape)

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
        xnorm, ynorm = float(cut_tlx / back_hw[1]), float(cut_tly / back_hw[0])
        w, h = float(cutout_arr.shape[1] / back_hw[1]), float(
            cutout_arr.shape[0] / back_hw[0])
        # Write bbox to cutout dataclass
        cutout.synth_norm_xywh = xnorm, ynorm, w, h
        used_cutouts.append(cutout)

    return used_cutouts, used_pots, potted, mask


def overlay_multicutouts2pot(multicuts, pots, syn, back_arr, back_arr_mask,
                             back_hw, pot_positions, min_multicut_dist):
    used_cutouts = []
    used_pots = []

    for pot, pot_position, multicutout in zip(pots, pot_positions, multicuts):
        # Get pot and info
        pot_arr = pot.array
        pot_arr = syn.transform(pot_arr)
        potshape = pot_arr.shape[:2]
        potshape = pot_arr.shape[:2]
        # Check coordinates for pot in top left corner
        topl_y, topl_x = transform_position(pot_position,
                                            potshape)  # to top left corner

        # Get a random set of cutout positions for a single pot
        pot_center = [topl_x, topl_y]
        maxdist = int(potshape[1] / 2)
        num_points = len(multicutout)
        multicut_pos = rand_multicut_positions(pot_center, num_points,
                                               min_multicut_dist, maxdist)

        #Overlay pot on background
        syn.fore_str = "Overlay pot"
        potted, _, _ = syn.overlay(topl_y, topl_x, pot_arr, back_arr)

        # Overlaying multiple cutouts on a single pot
        for idx, cutout in enumerate(multicutout):
            cutout.local_contours = None
            rgb = cutout.cls["rgb"]
            bgra = [rgb[2], rgb[1], rgb[0]]
            # Cutout data
            cutout_arr = cutout.array
            cutout_arr = syn.transform(cutout_arr)
            cutoutshape = cutout_arr.shape[:2]
            cutout.synth_hwc = cutout_arr.shape

            # Get cutout position from multi pot position
            multi_topl_x = multicut_pos[idx][0]
            multi_topl_y = multicut_pos[idx][1]

            # Get cutout center position
            cutx, cuty = syn.center_on_background(multi_topl_y, multi_topl_x,
                                                  potshape, cutoutshape)

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
            xnorm, ynorm = float(cut_tlx / back_hw[1]), float(cut_tly /
                                                              back_hw[0])
            w, h = float(cutout_arr.shape[1] / back_hw[1]), float(
                cutout_arr.shape[0] / back_hw[0])
            # Write bbox to cutout dataclass
            cutout.synth_norm_xywh = xnorm, ynorm, w, h
            used_cutouts.append(cutout)

        used_pots.append(pot)
    return used_cutouts, used_pots, potted, mask


def create_yolo_labels(cfg):
    jsonpaths = sorted(list(
        Path(cfg.synth.savedir, "metadata").glob("*.json")))
    imgpaths = sorted(list(Path(cfg.synth.savedir, "images").glob("*.jpg")))
    data = []
    for imgp, jsonp in zip(imgpaths, jsonpaths):
        if imgp.stem != jsonp.stem:
            log.error(
                "Metadata (json) and image (png) file stems do not match. Yolo label processing failed."
            )
        data.append(meta2yolo_prep(jsonp, imgp))

    metadata2yolo_labels(cfg.synth.savedir, data)


def process_contents(cfg, synth_contents, synth_image):
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


def prep_multicutout2pot_contents(cfg, data):
    num_images = cfg.synth.count
    contents = []

    for num in range(num_images):
        # Back
        back = np.random.choice(data.backgrounds, 1)[0]
        # Pots
        num_pots = np.random.randint(cfg.synth.min_pots,
                                     cfg.synth.max_pots,
                                     size=1)
        pots = np.random.choice(data.pots, num_pots,
                                replace=True) if cfg.synth.use_pots else []
        # Multicutouts per pot
        multicuts = []
        for i in range(num_pots[0]):
            min_cutouts = cfg.synth.cutouts_per_pot.min
            max_cutouts = cfg.synth.cutouts_per_pot.max
            num_cuts = np.random.randint(min_cutouts, max_cutouts, size=1)
            cuts = np.random.choice(data.cutouts, num_cuts, replace=True)
            multicuts.append(cuts)
        # Combine
        contents.append((multicuts, pots, back, cfg))
    return contents


def prep_cutout2pot_contents(cfg, data):
    num_images = cfg.synth.count
    contents = []
    for num in range(num_images):
        # Background
        back = np.random.choice(data.backgrounds, 1)[0]
        # Pots
        num_pots = np.random.randint(cfg.synth.min_pots,
                                     cfg.synth.max_pots,
                                     size=1)
        pots = np.random.choice(data.pots, num_pots,
                                replace=True) if cfg.synth.use_pots else []
        # Cutouts
        min_cutouts = cfg.synth.cutouts_per_pot.min
        max_cutouts = cfg.synth.cutouts_per_pot.max
        num_cuts = np.random.randint(min_cutouts, max_cutouts, size=1)
        cuts = np.random.choice(data.cutouts, num_cuts, replace=True)
        # Combine
        contents.append((cuts, pots, back, cfg))
    return contents