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


class SynthPipeline:

    def __init__(self, datacontainer, cfg: DictConfig) -> None:
        self.synthdir = Path(cfg.synth.synthdir)
        self.count = cfg.synth.count
        self.back_dir = cfg.synth.backgrounddir
        self.pot_dir = cfg.synth.potdir

        self.backgrounds = datacontainer.backgrounds
        self.used_backs = []

        self.pots = datacontainer.pots
        self.used_pots = []

        self.cutouts = datacontainer.cutouts
        self.used_cutouts = []

        self.replace_backs = self.replace_backgrounds()

        self.imagedir = Path(self.synthdir, "images")
        self.imagedir.mkdir(parents=True, exist_ok=True)
        self.maskdir = Path(self.synthdir, "masks")
        self.maskdir.mkdir(parents=True, exist_ok=True)
        self.json_dir = Path(self.synthdir, "metadata")
        self.json_dir.mkdir(parents=True, exist_ok=True)

#---------------------- Get images -------------------------------

    def replace_backgrounds(self):
        """Checks if count is larger than total number of backgrounds. Returns True for replacements. 
        """
        return True if self.count > len(self.backgrounds) else False

    def get_back(self, sortby=None):
        # if self.replace_backs:
        #     self.back = random.choice(self.backgrounds)
        # else:
        #     self.backgrounds = [
        #         x for x in self.backgrounds if x not in self.used_backs
        #     ]
        #     self.back = random.choice(self.backgrounds)
        # self.used_backs.append(self.back)

        # return self.back
        back = random.choice(self.backgrounds)
        return back

    def get_pots(self, num_pots, sortby=None):

        # self.pots = [x for x in self.pots if x not in self.used_pots]
        # if num_pots > self.count or len(self.pots) == 1:
        #     self.pots = self.pots
        # else:
        #     self.pots = random.sample(self.pots, num_pots)
        # self.used_pots.append(self.pots)
        return self.pots

    def get_cutouts(self, num_cutouts, sortby=None):
        # self.cutouts = [x for x in self.cutouts if x not in self.used_cutouts]

        # self.cutouts = random.sample(self.cutouts,
        #  min(num_cutouts, len(self.cutouts)))
        self.cutouts = random.sample(self.cutouts[:5000], 2000)
        # self.used_cutouts.append(self.cutouts)
        return self.cutouts

#------------------- Overlap checks --------------------------------

    def check_overlap(self, y, x, potshape, pot_positions):  # x = w ; h = y
        """Check overlap from list of previous bbox coordinates"""
        if not None in pot_positions:
            pot_h, pot_w, _ = potshape

            # adds pot dimensions to y,x position information
            x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
            # Something for overlap checking
            r0 = Rect(Point(x0, y0), Point(x1, y1))

            for y_old, x_old, oldpotshape in pot_positions:
                old_h, old_w, _ = oldpotshape
                x00, x11, y00, y11 = x_old, x_old + old_w, y_old, y_old + old_h
                r1 = Rect(Point(x00, y00), Point(x11, y11))
                while overlap(r0, r1):
                    x, y = x + random.randint(-2500, 2500), y + random.randint(
                        -2000, 2000)
                    x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
                    r0 = Rect(Point(x0, y0), Point(x1, y1))
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
                                fore):
        """ Crops foreground image (pot or plant) if position coordinates extend beyond background frame in positive direction.
        """
        pot_h, pot_w, _ = potshape
        back_h, back_w, _ = backshape

        if topl_x + pot_w > back_w:
            back_w = back_w - topl_x
            fore = fore[:, :back_w]

        if topl_y + pot_h > back_h:
            back_h = back_h - topl_y
            fore = fore[:back_h, :]
        return fore

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

    def blend(self, y, x, fore, back, mask=None):
        # image info
        back_h, back_w, _ = back.shape
        fore_h, fore_w, _ = fore.shape

        y2 = y + fore_h
        x2 = x + fore_w
        if y2 > back_h:
            y2 = back_h
        if x2 > back_w:
            x2 = back_w

        # masks
        fore_mask = fore[..., 3:] / 255
        alpha_l = 1.0 - fore_mask
        # blend
        print("\nback shape", back.shape)
        print("y", y)
        print("y2", y2)
        print("x", x)
        print("x2", x2)
        print("alpha_l ", alpha_l.shape)
        print("fore_mask ", fore_mask.shape)
        print("fore ", fore.shape)
        print("Sliced background shape ", back[y:y2, x:x2].shape)
        back[y:y2, x:x2] = alpha_l * back[y:y2, x:x2] + fore_mask * fore
        if mask is None:
            return back, y, x
        else:
            mask[y:y2, x:x2] = (255) * fore_mask + mask[y:y2, x:x2] * alpha_l
            return back, mask, y, x

    def overlay(self, topl_y, topl_x, fore, back, mask=None):
        # check positions
        print("\nBefore check negative", (topl_y, topl_x))
        topl_y, topl_x, fore = self.check_negative_positions(
            topl_y, topl_x, fore)
        print("After check negative", (topl_y, topl_x))
        print()
        fore = self.check_positive_position(topl_y, topl_x, fore.shape,
                                            back.shape, fore)
        return self.blend(topl_y, topl_x, fore, back, mask=mask)

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
    x_wid = np.linspace(0, imgwid, rand_wid, dtype=int)
    x_wid = x_wid[x_wid != 0]
    x_diff = np.diff(x_wid)
    x_diff = x_diff[0] if len(x_diff) > 1 else 0
    xs = [(x - math.ceil(x_diff / 2)) if x != 0 else math.ceil(x / 2)
          for x in x_wid]  # Accounts for 0 diff

    # Create height locations
    y_ht = np.linspace(0, imght, rand_ht, dtype=int)
    y_diff = np.diff(y_ht)[0]
    y_ht = y_ht[y_ht != imght]
    ys = [(y + int(y_diff / 2)) for y in y_ht]

    # Combine height and width to make coordinates
    coords = list(itertools.product(ys, xs))

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
    back_h, back_w, _ = background_imgshape
    tpl_y = y - int(back_h / 2)
    tpl_x = x - int(back_w / 2)
    return tpl_y, tpl_x


def transform_position(pot_position, imgshape, spread_factor):
    """ Applies random jitter factor to points and transforms them to top left image coordinates. 
    """
    y, x = pot_position

    # x = x + random.randint(-spread_factor, spread_factor)
    # y = y + random.randint(-int(spread_factor / 3), int(spread_factor / 3))

    tpl_y, tpl_x = center2topleft(y, x, imgshape)

    return tpl_y, tpl_x


def center_on_background(y, x, back_shape, fore_shape):
    # pot positions and shape top left corner
    back_h, back_w, _ = back_shape
    fore_h, fore_w, _ = fore_shape
    newx = int(((back_w - fore_w) / 2) + x)
    newy = int(((back_h - fore_h) / 2) + y)
    return newx, newy


def img2RGBA(img):
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))
    return img


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


def get_cutout_dir(batch_dir, cutout_dir):
    batch = Path(batch_dir).name
    cutout_dir = Path(cutout_dir, batch)
    return cutout_dir
