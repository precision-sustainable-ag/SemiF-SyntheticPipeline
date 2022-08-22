import itertools
import json
import math
import random
import uuid
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops.boxes as bops
from omegaconf import DictConfig
from PIL import Image, ImageEnhance
from semif_utils.utils import img2RGBA


class SynthPipeline:
    def __init__(self, data, cfg: DictConfig) -> None:
        self.synthdir = Path(cfg.synth.synthdir)
        self.count = cfg.synth.count
        self.back_dir = cfg.synth.backgrounddir
        self.pot_dir = cfg.synth.potdir

        self.backgrounds = data.backgrounds
        self.back = None
        self.back_ht, self.back_wd = None, None

        self.pots = data.pots
        self.pot = None
        self.pot_positions = None
        self.pot_ht, self.pot_wd = None, None

        self.cutouts = data.cutouts
        self.cutout = None
        self.cut_ht, self.cut_wd = None, None

        self.imagedir = Path(self.synthdir, "images")
        self.imagedir.mkdir(parents=True, exist_ok=True)
        self.maskdir = Path(self.synthdir, "masks")
        self.maskdir.mkdir(parents=True, exist_ok=True)
        self.json_dir = Path(self.synthdir, "metadata")
        self.json_dir.mkdir(parents=True, exist_ok=True)

        self.fore_str = None

#---------------------- Get images -------------------------------

    def replace_backgrounds(self):
        """Checks if count is larger than total number of backgrounds. Returns True for replacements. 
        """
        return True if self.count > len(self.backgrounds) else False

    def get_back(self, sortby=None):
        self.back = random.choice(self.backgrounds)

    def get_pot_positions(self):
        self.pot_positions = rand_pot_grid((6368, 9560))

    def get_pot(self):
        self.pot = random.choice(self.pots)

    def get_cutouts(self, sortby=None):
        return self.cutouts

    def prep_cutout(self):
        self.cutout = random.choice(self.cutouts)

#------------------- Overlap checks --------------------------------

    def check_overlap(self, y, x, potshape, pot_positions):  # x = w ; h = y
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
            topl_x = 0

        if topl_y < 0:
            fore = fore[abs(topl_y):, :]
            topl_y = 0
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
            print("topl_y + pot_h > back_h")
            print("topl_y ", topl_y)
            print("pot_h ", pot_h)
            back_h_edge = topl_y + pot_h - back_h
            print("back_h changed ", back_h)
            pot_arr = pot_arr[:-back_h_edge, :]

        return pot_arr

    def center_on_background(self, y, x, back_shape, fore_shape):
        # pot positions and shape top left corner
        back_h, back_w = back_shape
        fore_h, fore_w = fore_shape
        newx = int(((back_w - fore_w) / 2) + x)
        newy = int(((back_h - fore_h) / 2) + y)
        # assert (newy > self.back.back_ht) and (newx > self.back.back_wdt), "Centering on background failed."
        return newx, newy

#-------------------------- Transform pots or plants --------------------------------------

    def transform(self, fore):
        # get the alpha channel
        pil_fore = Image.fromarray(fore)
        fg_alpha = fore[:, :, 3]
        assert np.any(
            fg_alpha ==
            0), f'Foreground must have a 4th alpha layer for transparency.'

        # ** Apply Transformations **
        # Rotate the foreground
        angle_degrees = random.randint(0, 359)
        pil_fore = pil_fore.rotate(angle_degrees,
                                   resample=Image.BICUBIC,
                                   expand=True)

        # Scale the foreground
        scale = random.random() * .5 + .5  # Pick something between .5 and 1
        new_size = (int(pil_fore.size[0] * scale),
                    int(pil_fore.size[1] * scale))
        pil_fore = pil_fore.resize(new_size, resample=Image.BICUBIC)

        # Adjust foreground brightness
        brightness_factor = random.random(
        ) * .4 + .7  # Pick something between .7 and 1.1
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

    def blend_cutout(self, y, x, cutout, pot, mask):
        # image info
        pot_h, pot_w, _ = pot.shape
        cutout_h, cutout_w, _ = cutout.shape

        y2 = y + cutout_h
        x2 = x + cutout_w
        if y2 > pot_h:
            y2 = pot_h
        if x2 > pot_w:
            x2 = pot_w

        # masks
        cutout_mask = cutout[..., 3:] / 255
        alpha_l = 1.0 - cutout_mask
        pot[y:y2, x:x2] = alpha_l * pot[y:y2, x:x2] + cutout_mask * cutout
        mask[y:y2, x:x2] = (255) * cutout_mask + mask[y:y2, x:x2] * alpha_l
        return pot, mask, y, x

    def overlay(self, topl_y, topl_x, fore_arr, back_arr, mask=None):

        if "pot" in self.fore_str.lower():
            # check positions
            topl_y, topl_x, fore_arr = self.check_negative_positions(
                topl_y, topl_x, fore_arr)

            fore_arr = self.check_positive_position(topl_y, topl_x,
                                                    fore_arr.shape,
                                                    back_arr.shape, fore_arr)

            fore_arr, arr_y, arr_x = self.blend_pot(topl_y, topl_x, fore_arr,
                                                    back_arr)
            return fore_arr, arr_y, arr_x

        elif "cutout" in self.fore_str.lower():
            fore_arr = img2RGBA(fore_arr)
            # check positions
            topl_y, topl_x, fore_arr = self.check_negative_positions(
                topl_y, topl_x, fore_arr)

            fore_arr = self.check_positive_position(topl_y, topl_x,
                                                    fore_arr.shape,
                                                    back_arr.shape, fore_arr)

            fore_arr, mask, arr_y, arr_x = self.blend_cutout(topl_y,
                                                             topl_x,
                                                             fore_arr,
                                                             back_arr,
                                                             mask=mask)

            return fore_arr, mask, arr_y, arr_x

    #---------------------- Save to directory and DB --------------------------------

    def save_synth(self, res, mask):
        fname = uuid.uuid4().hex + ".png"
        savepath = Path(self.imagedir, fname)
        savemask = Path(self.maskdir, fname)
        res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
        mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(savepath), res)
        cv2.imwrite(str(savemask), mask)
        return Path(savepath), Path(savemask)


########################################################################
########################################################################
#------------------- Helper functions --------------------------------
def rand_pot_grid(img_shape):
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

    rand_wid = random.randint(2, 5)
    rand_ht = random.choice([2, 3])

    # Create width locations
    wid = np.linspace(0, imgwid, rand_wid, dtype=int)
    wid = wid[wid != 0]
    wid_diff = np.diff(wid)

    if len(wid_diff) >= 2:
        wid_diff = wid_diff[0]
    wid = [(x - math.ceil(wid_diff / 2)) if wid_diff != 0 else math.ceil(x / 2)
           for x in wid]  # Accounts for 0 diff

    # Create height locations
    ht = np.linspace(0, imght, rand_ht, dtype=int)
    ht_diff = np.diff(ht)[0]
    ht = ht[ht != imght]
    ht = [(x + int(ht_diff / 2)) for x in ht]
    # Combine height and width to make coordinates
    coords = list(itertools.product(wid, ht))
    rand_x = 100 if rand_wid >= 4 else 600
    rand_y = 100 if (rand_ht == 3) and (rand_wid >= 4) else 700
    coords = [(x + random.randint(-rand_x, rand_x),
               y + random.randint(-rand_y, rand_y)) for x, y in coords]
    return coords


def bbox_iou(box1, box2):
    box1 = torch.tensor([box1], dtype=torch.float)
    box2 = torch.tensor([box2], dtype=torch.float)
    iou = bops.box_iou(box1, box2)
    return iou


def get_img_bbox(x, y, imgshape):
    pot_h, pot_w, _ = imgshape
    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
    bbox = [x0, y0, x1, y1]  # top right corner, bottom left corner
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
    print("\nClean data: ")
    print(data["background"][0])
    data["background"][0]["background_path"] = "/".join(
        Path(data["background"][0]["background_path"]).parts[-2:])

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
