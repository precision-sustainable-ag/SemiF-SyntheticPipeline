import itertools
import json
import math
import random
import shutil
import uuid
from dataclasses import asdict
from multiprocessing import Value
from pathlib import Path
from statistics import mean, median

import cv2
import numpy as np
import pandas as pd
import torch
import torchvision.ops.boxes as bops
from omegaconf import DictConfig
from PIL import Image, ImageEnhance
from tqdm import trange

from synth_utils.datasets import SynthImage
from synth_utils.utils import img2RGBA, read_json


class SynthPipeline:
    def __init__(self, data, cfg: DictConfig) -> None:
        self.cfg = cfg
        self.data = data
        self.jobdir = self.cfg.job.jobdir
        self.synthdir = Path(self.cfg.data.synthdir)
        self.count = self.cfg.synth.count
        self.back_dir = self.cfg.data.backgrounddir
        self.pot_dir = self.cfg.data.potdir
        self.cmap = self.data.color_map
        self.cls_id = None

        self.backgrounds = self.data.backgrounds
        self.back = None
        self.back_ht, self.back_wd = None, None

        self.pots = self.data.pots
        self.pot = None
        self.pot_positions = None
        self.pot_ht, self.pot_wd = None, None

        self.multi_coty_dist = cfg.cutouts.max_coty_dist.max
        self.plants_per_pot = cfg.cutouts.plants_per_pot
        self.cutouts = self.data.cutouts
        # self.used_cutouts = []
        self.cutout = None
        self.cut_ht, self.cut_wd = None, None

        self.bboxes = []
        if self.cfg.job.jobdir:
            self.imagedir = Path(self.cfg.job.jobdir, "images")
            self.imagedir.mkdir(parents=True, exist_ok=True)
            self.maskdir = Path(self.cfg.job.jobdir, "masks")
            self.maskdir.mkdir(parents=True, exist_ok=True)
            self.json_dir = Path(self.cfg.job.jobdir, "metadata")
            self.json_dir.mkdir(parents=True, exist_ok=True)
            self.bbox_dir = Path(self.cfg.job.jobdir, "labels")
            self.bbox_dir.mkdir(parents=True, exist_ok=True)

        # Make a copy of classes txt file for easy handling
        dst = Path(self.jobdir, "classes.txt")
        src = Path(cfg.data.datadir, "classes.txt")
        shutil.copy(src, dst)

        self.fore_str = None

#---------------------- Get images -------------------------------

    def replace_backgrounds(self):
        """Checks if count is larger than total number of backgrounds. Returns True for replacements. 
        """
        return True if self.count > lesn(self.backgrounds) else False

    def get_back(self, sortby=None):
        return random.choice(self.backgrounds)

    def get_pot_positions(self):
        pots = random.randint(1, self.plants_per_pot)
        pot_pos = rand_pot_grid(pots, (6368, 9560))
        return pot_pos

    def get_pot(self):
        self.pot = random.choice(self.pots)

    def get_cutouts(self, sortby=None):
        cutout = random.choice(self.cutouts)[0]
        return cutout

#------------------- Overlap checks --------------------------------

    def check_overlap(self, y, x, potshape, pot_positions):    # x = w ; h = y
        """Check overlap from list of previous bbox coordinates"""
        if not None in pot_positions:
            pot_h, pot_w = potshape

            # adds pot dimensions to y,x position information
            x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
            # Something for overlap checking
            r0 = Rect(Point(x0, y0), Point(x1, y1))

            for y_old, x_old, oldpotshape in pot_positions:
                old_h, old_w = oldpotshape
                x00, x11, y00, y11 = x_old, x_old + old_w, y_old, y_old + old_h
                r1 = Rect(Point(x00, y00), Point(x11, y11))
                ovrlap_var = overlap(r0, r1)
                while ovrlap_var:
                    x, y = x + random.randint(-150, 150), y + random.randint(
                        -50, 50)
                    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
                    r0 = Rect(Point(x0, y0), Point(x1, y1))
                    if y > self.back.back_ht or x > self.back.back_wdt:
                        ovrlap_var = True
        return y, x

    def check_negative_positions(self, topl_y, topl_x, fore):
        """ Crops pot image if position coordinates are negative. 
            Crop amount is absolute value of negative position value. """
        if topl_x < 0:
            fore = fore[:, abs(topl_x):]
            # topl_x = 0
            topl_x = abs(topl_x)

        if topl_y < 0:
            fore = fore[abs(topl_y):, :]
            # topl_y = 0
            topl_y = abs(topl_y)

        return topl_y, topl_x, fore

    def check_positive_position(self, topl_y, topl_x, potshape, backshape,
                                pot_arr):
        """ Crops foreground image (pot or plant) if position coordinates extend beyond background frame in positive direction.
        """
        pot_h, pot_w, _ = potshape
        back_h, back_w, _ = backshape

        if topl_x + pot_w > back_w:
            back_w_edge = topl_x + pot_w - back_w
            pot_arr = pot_arr[:, :-back_w_edge]

        if topl_y + pot_h > back_h:
            back_h_edge = topl_y + pot_h - back_h
            pot_arr = pot_arr[:-back_h_edge, :]

        if topl_x > back_w:
            topl_x = topl_x - back_w

        if topl_y > back_h:
            topl_y = topl_y - back_h

        return pot_arr, topl_x, topl_y

    def center_on_background(self, y, x, back_shape, fore_shape):
        # pot positions and shape top left corner
        back_h, back_w = back_shape
        fore_h, fore_w = fore_shape
        newx = int(((back_w - fore_w) / 2) + x)
        newy = int(((back_h - fore_h) / 2) + y)
        # assert (newy > self.back.back_ht) and (newx > self.back.back_wdt), "Centering on background failed."
        return newx, newy

#-------------------------- Transform pots or plants --------------------------------------

    def transform_cutout(self, fore):
        # get the alpha channel
        if fore.shape[2] == 3:
            fore = img2RGBA(fore)

        pil_fore = Image.fromarray(fore)

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        pil_fore = pil_fore.rotate(angle_degrees,
                                   resample=Image.BICUBIC,
                                   expand=True)

        # Scale the foreground
        scale = random.uniform(0.8, 1.1)
        new_size = (int(pil_fore.size[0] * scale),
                    int(pil_fore.size[1] * scale))
        pil_fore = pil_fore.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random(
        ) * .4 + .7    # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(pil_fore)
        pil_fore = enhancer.enhance(brightness_factor)

        # Add any other transformations here...
        fore = np.array(pil_fore)
        return fore

    def transform_pot(self, fore):
        # get the alpha channel
        if fore.shape[2] == 3:
            fore = img2RGBA(fore)

        pil_fore = Image.fromarray(fore)

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        pil_fore = pil_fore.rotate(angle_degrees,
                                   resample=Image.BICUBIC,
                                   expand=True)

        # Scale the foreground
        scale = random.uniform(0.7, 1.1)

        new_size = (int(pil_fore.size[0] * scale),
                    int(pil_fore.size[1] * scale))
        pil_fore = pil_fore.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random(
        ) * .4 + .7    # Pick something between .7 and 1.1
        enhancer = ImageEnhance.Brightness(pil_fore)
        pil_fore = enhancer.enhance(brightness_factor)

        # Add any other transformations here...
        fore = np.array(pil_fore)
        return fore


#-------------------------- Overlay and blend --------------------------------------

    def blend_pot(self, y, x, pot, back):
        # image info
        back_h, back_w, _ = back.shape
        pot_h, pot_w, _ = pot.shape

        y2 = y + pot_h
        x2 = x + pot_w
        if y2 > back_h:
            y2 = back_h
        if x2 > back_w:
            x2 = back_w

        # masks
        pot_mask = pot[..., 3:] / 255
        alpha_l = 1.0 - pot_mask
        back[y:y2, x:x2] = alpha_l * back[y:y2, x:x2] + pot_mask * pot
        return back, y, x

    def blend_cutout(self, y, x, cutout, back, mask):
        # image info
        back_h, back_w, _ = back.shape
        cutout_h, cutout_w, _ = cutout.shape

        y2 = y + cutout_h
        x2 = x + cutout_w
        if y2 > back_h:
            y2 = back_h
        if x2 > back_w:
            x2 = back_w

        # bbox normalized xywh, xy as centerpoints of bbox
        norm_center_x = (float(x) + cutout_w / 2) / back_w
        norm_center_y = (float(y) + cutout_h / 2) / back_h
        norm_w = cutout_w / back_w
        norm_h = cutout_h / back_h

        bbox = norm_center_x, norm_center_y, norm_w, norm_h

        # masks
        cutout_mask = cutout[..., 3:] / 255

        alpha_l = 1.0 - cutout_mask
        back[y:y2, x:x2] = alpha_l * back[y:y2, x:x2] + cutout_mask * cutout

        mask[y:y2, x:x2] = (255) * cutout_mask + mask[y:y2, x:x2] * alpha_l
        mask[y:y2, x:x2][mask[y:y2, x:x2] != 0] = self.cls_id

        return back, mask, y, x, bbox

    def overlay(self, topl_y, topl_x, fore_arr, back_arr, mask=None):

        if "pot" in self.fore_str.lower():
            # check positions
            topl_y, topl_x, fore_arr = self.check_negative_positions(
                topl_y, topl_x, fore_arr)

            fore_arr, topl_x, topl_y = self.check_positive_position(
                topl_y, topl_x, fore_arr.shape, back_arr.shape, fore_arr)

            back_arr, arr_y, arr_x = self.blend_pot(topl_y, topl_x, fore_arr,
                                                    back_arr)
            return back_arr, arr_y, arr_x

        elif "cutout" in self.fore_str.lower():
            # check positions
            topl_y, topl_x, fore_arr = self.check_negative_positions(
                topl_y, topl_x, fore_arr)

            fore_arr, topl_x, topl_y = self.check_positive_position(
                topl_y, topl_x, fore_arr.shape, back_arr.shape, fore_arr)

            back_arr, mask, arr_y, arr_x, bbox = self.blend_cutout(topl_y,
                                                                   topl_x,
                                                                   fore_arr,
                                                                   back_arr,
                                                                   mask=mask)

            return back_arr, mask, arr_y, arr_x, bbox

    #---------------------- Save to directory and DB --------------------------------

    def save_synth(self, res, mask):
        fname = uuid.uuid4().hex + ".png"
        savepath = Path(self.imagedir, fname)
        savemask = Path(self.maskdir, fname)
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(savepath), res)
        pilmask = Image.fromarray(mask)
        pilmask = pilmask.save(savemask)

        return Path(savepath), Path(savemask)

    def pipeline(self):
        back = self.get_back()
        pot_positions = self.get_pot_positions()
        back_arr = back.array.copy()
        back_arr_mask = np.zeros_like(back_arr)
        used_pot_positions = []
        used_cutouts = []
        unique_cutouts = []
        used_pots = []
        bboxes = []
        for potidx in range(len(pot_positions)):
            # Get pot and info
            pot_position = pot_positions[potidx]    # center (y,x)
            # Get single pot
            self.get_pot()
            pot_arr = self.pot.array
            pot_arr = self.transform_pot(pot_arr)
            potshape = self.pot.pot_ht, self.pot.pot_wdt
            # Check coordinates for pot in top left corner
            topl_y, topl_x = transform_position(
                pot_position, potshape)    # to top left corner
            #Overlay pot on background
            self.fore_str = "Overlay pot"
            potted, poty, potx = self.overlay(topl_y, topl_x, pot_arr,
                                              back_arr)
            # Overlay cutouts
            if self.plants_per_pot > 1:
                new_pot_coords = get_random_point(
                    (potx, poty), self.multi_coty_dist, self.plants_per_pot)
                # self.cutout_group = random.choices(cutout_groups,
                #                                    k=len(new_pot_coords))
            else:
                new_pot_coords = [[potx, poty]]
            for pot_coord in new_pot_coords:
                potx, poty = pot_coord
                self.cutout = self.get_cutouts()
                cutout_arr = self.cutout.array
                cutoutshape = self.cutout.array.shape[:2]
                cutout_arr = self.transform_cutout(cutout_arr)
                self.fore_str = "Overlay cutout"
                cls = self.cutout.cls
                if cls["USDA_symbol"] == "unknown":
                    self.cls_id = self.cmap["plant"]["class_id"]
                else:
                    self.cls_id = self.cmap[cls["USDA_symbol"]]["class_id"]

                cutx, cuty = self.center_on_background(poty, potx, potshape,
                                                       cutoutshape)

                potted, mask, _, _, bbox = self.overlay(cuty,
                                                        cutx,
                                                        cutout_arr,
                                                        back_arr,
                                                        mask=back_arr_mask)
                norm_center_x, norm_center_y, norm_w, norm_h = bbox

                # bboxes.append(self.cutout.cutout_id, self.cls_id, norm_center_x,
                #               norm_center_y, norm_w, norm_h)
                used_pot_positions.append([topl_y, topl_x, potshape])
                self.cutout.xywh = self.cutout.cutout_id, self.cls_id, norm_center_x, norm_center_y, norm_w, norm_h
                txt_bbox = f"{self.cls_id} {norm_center_x} {norm_center_y} {norm_w} {norm_h}"
                bboxes.append(txt_bbox)
                cutout = asdict(self.cutout)
                unique_cutouts.append(cutout)
                used_pots.append(self.pot)

        # Save synth image and mask
        savepath, savemask = self.save_synth(potted, mask[:, :, 0])

        # Path info to save
        savestem = savepath.stem
        savepath = "/".join(savepath.parts[-2:])
        savemask = "/".join(savemask.parts[-2:])
        data_root = Path(self.cfg.data.synthdir).name

        # To SynthImage dataclass
        synimage = SynthImage(data_root=data_root,
                              synth_path=savepath,
                              synth_maskpath=savemask,
                              pots=[asdict(pot) for pot in used_pots],
                              background=asdict(back),
                              cutouts=unique_cutouts)
        # Clean dataclass
        # data_dict = asdict(synimage)
        # synimage = clean_data(data_dict)
        # To json
        jsonpath = Path(self.json_dir, savestem + ".json")
        bboxlabel_path = Path(self.bbox_dir, savestem + ".txt")

        with open(bboxlabel_path, "a") as f:
            for box in bboxes:
                f.write(box)
                f.write("\n")
        # synimage.save_config(jsonpath)

        save_dataclass_json(asdict(synimage), jsonpath)
        return used_cutouts


class FilterCutouts:
    def __init__(self, cutout_jsons, cfg: DictConfig) -> None:

        self.cutout_jsons = cutout_jsons
        self.cutoutdir = cfg.data.cutoutdir
        self.batch_id = cfg.general.batch_id
        self.species = cfg.cutouts.species
        self.is_green = cfg.cutouts.is_green
        self.is_primary = cfg.cutouts.is_primary
        self.extends_border = cfg.cutouts.extends_border
        self.green_sum_max = cfg.cutouts.green_sum_max
        self.green_sum_min = cfg.cutouts.green_sum_min
        self.save_csv = cfg.cutouts.save_csv
        self.filtered_jsons = self.get_cutout_jsons()

    def get_cutout_jsons(self):
        df = self.cutoutjson2csv()
        df = self.prep_clean(df)
        df = self.set_and_sort(df)
        return df["path"]

    def read_cutout_json(self, path):
        with open(path) as f:
            cutout = json.load(f)
        return cutout

    def cutoutjson2csv(self):
        # Get all json files
        cutouts = []
        for cutout in self.cutout_jsons:
            # Get dictionaries
            cutout = self.read_cutout_json(cutout)
            row = cutout["cutout_props"]
            cls = cutout["cls"]
            # Extend nested dicts to single column header
            for ro in row:
                rec = {ro: row[ro]}
                cutout.update(rec)
                for cl in cls:
                    spec = {cl: cls[cl]}
                    cutout.update(spec)
            # Remove duplicate nested dicts
            cutout.pop("cutout_props")
            cutout.pop("cls")
            # Create and append df
            cutouts.append(pd.DataFrame(cutout, index=[0]))
        # Concat and reset index of main df
        df = pd.concat(cutouts).reset_index()
        # Save dataframe
        if self.save_csv:
            csv_path = f"{self.cutoutdir}/{self.batch_id}.csv"
            if not csv_path.exists():
                print(f"Creating cutout metadata csv...\nSaving at {csv_path}")
                df.to_csv(csv_path, index=False)
            else:
                print("Metadata CSV already exists")
        return df

    def get_path(self, df):
        """ 
        """
        df["path"] = self.cutoutdir + "/" + df["cutout_path"].str.replace(
            ".png", ".json", regex=False)
        return df

    def calc_thresh(self, df):
        """ Calculates threshold values

        NOT USED """
        features = ["green_sum", "solidity", "area", "perimeter"]
        stats = {"max": max, "min": min, "median": median, "mean": mean}

        for feat in features:
            for stat in stats:
                val = df[feat]
                new_val = getattr(val, stat)()
                print(f"{feat} {stat}", new_val)


########################################################################
########################################################################
#------------------- Helper functions --------------------------------


def get_random_point(coord, max_dist, max_cotys):
    """ For adding cotyledons. Takes in coords and returns another coordinate randomely scattered.
        Define max distnace from og coord."""

    coord = np.array(coord, dtype=np.float64)
    coords = []
    for coty in range(max_cotys):
        coord_copy = coord.copy()
        noise = np.random.uniform(low=-max_dist,
                                  high=max_dist,
                                  size=len(coord_copy))
        coord_copy += noise
        coords.append(coord_copy.astype(np.uint16))
    return coords


def rand_pot_grid(num_pots, img_shape, max_cotys=None, max_coty_dist=None):
    """Creates a set of grid-like coordinates based on image size.
       The number of coordinates (pots) is based on randomely choosing
       the number rows and columns. Coordinates are evenly spaced both 
       vertically and horizontally based on image shape and number of 
       rows and columns. Zero coordinates are removed as are the maximum
       extent of the image (img.shape).

    Args:
        img_shape (tuple): shape of image

    Returns:
        coords: list of tuples, evenly spaced coordinates
    """

    imght, imgwid = img_shape[:2]

    rand_ht, rand_wid = pot_rows_cols(num_pots)

    if rand_wid == 1:
        wid = [int(imgwid / 2)]
    else:
        # Create width locations
        wid = np.linspace(0, imgwid, rand_wid, dtype=int)
        wid = wid[wid != 0]
        wid_diff = np.diff(wid)

        if len(wid_diff) >= 2:
            wid_diff = wid_diff[0]
        wid = [(x -
                math.ceil(wid_diff / 2)) if wid_diff != 0 else math.ceil(x / 2)
               for x in wid]    # Accounts for 0 diff

    if rand_ht == 1:
        ht = [int(imght / 2)]
    else:
        # Create height locations
        ht = np.linspace(0, imght, rand_ht, dtype=int)
        ht_diff = np.diff(ht)[0]
        ht = ht[ht != imght]
        ht = [(x + int(ht_diff / 2)) for x in ht]
    # Combine height and width to make coordinates
    coords = list(itertools.product(wid, ht))

    rand_x = 50    # if rand_wid >= 4 else 600
    rand_y = 50    # if (rand_ht == 3) and (rand_wid >= 4) else 700

    if rand_wid == 1:
        rand_x = 3000
    if rand_ht == 1:
        rand_y = 1000

    coords = [(x + random.randint(-rand_x, rand_x),
               y + random.randint(-rand_y, rand_y)) for x, y in coords]
    if max_coty_dist:
        rand_coords = []
        for coord in coords:
            cos = get_random_point(coord, max_coty_dist, max_cotys)
            for co in cos:
                rand_coords.append(co)
        coords = rand_coords

    return coords


def pot_rows_cols(num_pots):

    rows_cols = {
        1: [[1, 1]],
        2: [[1, 2], [2, 1]],
        3: [[1, 3], [3, 1], [2, 2]],
        4: [[1, 4], [2, 2]],
        5: [[1, 5], [2, 3], [2, 4], [2, 5], [3, 2], [3, 3]],
        6: [[2, 3], [2, 4], [2, 5], [3, 2], [3, 3]],
        7: [[2, 4], [2, 5], [3, 3]],
        8: [[2, 4], [2, 5], [3, 3], [3, 4]],
        9: [[2, 5], [3, 3], [3, 4]],
        10: [[2, 5], [3, 3], [3, 4]],
        11: [[2, 6], [3, 3], [3, 4], [3, 5]],
        12: [[2, 6], [3, 4], [3, 5]],
        13: [[3, 5]],
        14: [[3, 5]],
        15: [[3, 5]],
    }

    return random.choice(rows_cols[num_pots])


# def bbox_iou(box1, box2):
#     box1 = torch.tensor([box1], dtype=torch.float)
#     box2 = torch.tensor([box2], dtype=torch.float)
#     iou = bops.box_iou(box1, box2)
#     return iou


def get_img_bbox(x, y, imgshape):
    pot_h, pot_w, _ = imgshape
    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
    bbox = [x0, y0, x1, y1]    # top right corner, bottom left corner
    return bbox


def center2topleft(y, x, background_imgshape):
    """ Gets top left coordinates of an image from center point coordinate
    """
    back_h, back_w = background_imgshape
    tpl_y = y - int(back_h / 2)
    tpl_x = x - int(back_w / 2)
    return tpl_y, tpl_x


def transform_position(pot_position, imgshape):
    """ Applies random jitter factor to points and transforms them to top left image coordinates. 
    """
    x, y = pot_position
    tpl_y, tpl_x = center2topleft(y, x, imgshape)

    return tpl_y, tpl_x


class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y


class Rect(object):
    def __init__(self, p1, p2):
        '''Store the top, bottom, left and right values for points 
               p1 and p2 are the (corners) in either order
        '''
        self.left = min(p1.x, p2.x)
        self.right = max(p1.x, p2.x)
        self.bottom = min(p1.y, p2.y)
        self.top = max(p1.y, p2.y)


def overlap(r1, r2):
    '''Overlapping rectangles overlap both horizontally & vertically
    '''
    return range_overlap(r1.left, r1.right,
                         r2.left, r2.right) and range_overlap(
                             r1.bottom, r1.top, r2.bottom, r2.top)


def range_overlap(a_min, a_max, b_min, b_max):
    '''Neither range is completely greater than the other
    '''
    return (a_min <= b_max) and (b_min <= a_max)


def dict_to_json(dic, path):
    json_path = Path(path)
    with open(json_path, 'w') as j:
        json.dump(dic, j, indent=4, default=str)


def clean_data(data):
    """ Convert absolute pot and background paths to relative.
        Takes the last two components of a path object for each.
        
        Takes in and returns a dictionary of dataclass to be 
        stored in json and db. 
    """
    data["background"]["background_path"] = "/".join(
        Path(data["background"]["background_path"]).parts[-2:])

    pots = data["pots"]
    for pot in pots:
        pot["pot_path"] = "/".join(Path(pot["pot_path"]).parts[-2:])

    for cutout in data["cutouts"]:
        cutout["cutout_path"] = "/".join(
            Path(cutout["cutout_path"]).parts[-2:])

    return data


def save_dataclass_json(data_dict, path):
    json_path = Path(path)
    with open(json_path, 'w') as j:
        json.dump(data_dict, j, indent=4, default=str)
