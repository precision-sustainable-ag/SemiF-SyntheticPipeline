import json
import random
import uuid
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.ops.boxes as bops
from omegaconf import DictConfig


class SynthPipeline:

    def __init__(self, datacontainer, cfg: DictConfig) -> None:
        self.synth_config = cfg.synth
        self.synthdir = Path(cfg.data.synthdir)
        self.count = cfg.synth.count
        self.back_config = cfg.synth.backgrounds
        self.pot_config = cfg.synth.pots

        self.backgrounds = datacontainer.backgrounds
        self.used_backs = []

        self.pots = datacontainer.pots
        self.used_pots = []

        self.cutouts = datacontainer.cutouts
        self.used_cutouts = []

        self.replace_backs = self.replace_backgrounds()

        self.imagedir = Path(self.synthdir, "images")
        self.maskdir = Path(self.synthdir, "masks")
        self.json_dir = Path(self.synthdir, "metadata")

#---------------------- Get images -------------------------------

    def replace_backgrounds(self):
        """Checks if count is larger than total number of backgrounds. Returns True for replacements. 
        """
        return True if self.count > len(self.backgrounds) else False

    def get_back(self, sortby=None):
        if self.replace_backs:
            self.back = random.choice(self.backgrounds)
        else:
            self.backgrounds = [
                x for x in self.backgrounds if x not in self.used_backs
            ]
            self.back = random.choice(self.backgrounds)
        self.used_backs.append(self.back)

        return self.back

    def get_pots(self, num_pots, sortby=None):

        self.pots = [x for x in self.pots if x not in self.used_pots]
        if num_pots > self.count or len(self.pots) == 1:
            self.pots = self.pots
        else:
            self.pots = random.sample(self.pots, num_pots)
        self.used_pots.append(self.pots)
        return self.pots

    def get_cutouts(self, num_cutouts, sortby=None):
        self.cutouts = [x for x in self.cutouts if x not in self.used_cutouts]

        self.cutouts = random.sample(self.cutouts,
                                     min(num_cutouts, len(self.cutouts)))

        self.used_cutouts.append(self.cutouts)
        return self.cutouts

#------------------- Overlap checks --------------------------------

    def check_overlap(self, x, y, potshape, pot_positions):  # x = w ; h = y
        """Check overlap from list of previous bbox coordinates"""
        if not None in pot_positions:
            pot_h, pot_w, _ = potshape
            x0, x1, y0, y1 = x, x + pot_w, y, y + pot_h
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
        return x, y

    def check_negative_positions(self, h0, w0, pot):
        """ Crops pot image if position coordinates are negative. 
            Crop amount is absolute value of negative position value. """
        if w0 < 0:
            pot = pot[:, abs(w0):]
            w0 = 0

        if h0 < 0:
            pot = pot[abs(h0):, :]
            h0 = 0
        return h0, w0, pot

    def check_positive_position(self, h0, w0, potshape, backshape, pot):
        """ Crops pot image if position coordinates extend beyond background frame in positive direction.
        """
        pot_h, pot_w, _ = potshape
        back_h, back_w, _ = backshape

        if w0 + pot_w > back_w:
            back_w = back_w - w0
            pot = pot[:, :back_w]

        if h0 + pot_h > back_h:
            back_h = back_h - h0
            pot = pot[:back_h, :]
        return pot

#-------------------------- Overlay and blend --------------------------------------

    def blend(self, h0, w0, fore, back, mask=None):
        # image info
        fore_h, fore_w, _ = fore.shape
        h1 = h0 + fore_h
        w1 = w0 + fore_w
        # masks
        fore_mask = fore[..., 3:] / 255
        alpha_l = 1.0 - fore_mask
        # blend
        back[h0:h1, w0:w1] = alpha_l * back[h0:h1, w0:w1] + fore_mask * fore
        if mask is None:
            return back, h0, w0
        else:
            mask[h0:h1,
                 w0:w1] = (255) * fore_mask + mask[h0:h1, w0:w1] * alpha_l
            return back, mask, h0, w0

    def overlay(self, y0, x0, fore, back, mask=None):
        # check positions
        y0, x0, fore = self.check_negative_positions(y0, x0, fore)
        fore = self.check_positive_position(y0, x0, fore.shape, back.shape,
                                            fore)
        return self.blend(y0, x0, fore, back, mask=mask)

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


def center2topleft(x, y, background_imgshape):
    """ Gets top left coordinates of an image from center point coordinate
    """
    back_h, back_w, _ = background_imgshape
    y = y - int(back_h / 2)
    x = x - int(back_w / 2)
    return x, y


def transform_position(points, imgshape, spread_factor):
    """ Applies random jitter factor to points and transforms them to top left image coordinates. 
    """
    y, x = points

    x, y = x + random.randint(
        -spread_factor, spread_factor), y + random.randint(
            -int(spread_factor / 3), int(spread_factor / 3))

    x, y = center2topleft(x, y, imgshape)

    return x, y


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


def get_cutout_dir(batch_dir, cutout_dir):
    batch = Path(batch_dir).name
    cutout_dir = Path(cutout_dir, batch)
    return cutout_dir
