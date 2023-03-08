import random
import uuid
from pathlib import Path

import cv2
import numpy as np
from omegaconf import DictConfig
from PIL import Image, ImageEnhance

from utils.utils import Point, Rect, img2RGBA, overlap, rand_pot_grid


class SynthPipeline:

    def __init__(self, cfg: DictConfig) -> None:
        self.synthdir = Path(cfg.synth.synthdir)
        self.bgra_palette = [[0, 0, 0]]
        self.pot_positions = None
        self.cutout = None
        self.cut_ht, self.cut_wd = None, None
        # Create folders
        if cfg.synth.savedir:
            self.imagedir = Path(cfg.synth.savedir, "images")
            self.imagedir.mkdir(parents=True, exist_ok=True)
            self.maskdir = Path(cfg.synth.savedir, "masks")
            self.maskdir.mkdir(parents=True, exist_ok=True)
            self.json_dir = Path(cfg.synth.savedir, "metadata")
            self.json_dir.mkdir(parents=True, exist_ok=True)

        else:
            self.imagedir = Path(self.synthdir, "images")
            self.imagedir.mkdir(parents=True, exist_ok=True)
            self.maskdir = Path(self.synthdir, "masks")
            self.maskdir.mkdir(parents=True, exist_ok=True)
            self.json_dir = Path(self.synthdir, "metadata")
            self.json_dir.mkdir(parents=True, exist_ok=True)

        self.fore_str = None

#---------------------- Get images -------------------------------

# def get_back(self, sortby=None):
#     self.back = random.choice(self.backgrounds)

    def get_pot_positions(self):
        self.pot_positions = rand_pot_grid((6368, 9560))
        return self.pot_positions

    # def get_pot(self):
    #     self.pot = random.choice(self.pots)

    # def get_cutouts(self, sortby=None):
    #     return self.cutouts

    # def prep_cutout(self):
    #     self.cutout = random.choice(self.cutouts)

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
            back_h_edge = topl_y + pot_h - back_h
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
        if fore.shape[2] == 3:
            fore = img2RGBA(fore)

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
        # Crop to extent
        pil_fore_cropped = pil_fore.crop(pil_fore.getbbox())
        fore = np.array(pil_fore_cropped)
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

    def blend_cutout(self, y, x, cutout, pot, mask, bgra):
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
        # TODO add color to the masks depending on unique rgb values
        cutout_mask = cutout[..., 3:] / 255
        alpha_l = 1.0 - cutout_mask
        pot[y:y2, x:x2] = alpha_l * pot[y:y2, x:x2] + cutout_mask * cutout
        mask[y:y2, x:x2] = (255) * cutout_mask + mask[y:y2, x:x2] * alpha_l
        mask = mask[:, :, :3]  # this is dumb but whatever

        # Already used colors
        palette = np.array(self.bgra_palette).transpose()

        # all(2) force all channels to be equal
        # any(-1) matches any color
        temp_mask = (mask[y:y2, x:x2][:, :, :, None] == palette).all(2).any(-1)

        # target color
        bgra = np.array(bgra)

        # np.where to remap mask while keeping palette colors:
        mask[y:y2, x:x2] = np.where(temp_mask[:, :, None], mask[y:y2, x:x2],
                                    bgra[None, None, :])

        return pot, mask, y, x

    def overlay(self,
                topl_y,
                topl_x,
                fore_arr,
                back_arr,
                bgra=None,
                mask=None):

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
            # check positions
            topl_y, topl_x, fore_arr = self.check_negative_positions(
                topl_y, topl_x, fore_arr)

            fore_arr = self.check_positive_position(topl_y, topl_x,
                                                    fore_arr.shape,
                                                    back_arr.shape, fore_arr)

            fore_arr, mask, arr_y, arr_x = self.blend_cutout(
                topl_y, topl_x, fore_arr, back_arr, mask, bgra)

            return fore_arr, mask, arr_y, arr_x

    #---------------------- Save to directory and DB --------------------------------

    def save_synth(self, res, mask):
        fstem = uuid.uuid4().hex

        savepath = Path(self.imagedir, fstem + ".jpg")
        savemask = Path(self.maskdir, fstem + ".png")
        # res = cv2.cvtColor(res, cv2.COLOR_RGBA2BGRA)
        # mask = cv2.cvtColor(mask, cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(savepath), res[:, :, :3])
        cv2.imwrite(str(savemask), mask)
        return Path(savepath), Path(savemask)
