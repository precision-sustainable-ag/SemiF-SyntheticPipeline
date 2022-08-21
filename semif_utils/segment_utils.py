import json
from pathlib import Path

import cv2
import numpy as np
from dacite import Config, from_dict
from omegaconf import DictConfig
from scipy import ndimage as ndi
from skimage import measure
from skimage.color import label2rgb
from skimage.exposure import rescale_intensity
from skimage.filters import rank
from skimage.measure import label
from skimage.morphology import disk
from skimage.segmentation import watershed
from tqdm import tqdm

from semif_utils.datasets import CutoutProps, ImageData
from semif_utils.utils import (make_exg, make_exg_minus_exr, make_exr,
                               make_kmeans, make_ndi, otsu_thresh, parse_dict,
                               reduce_holes, rescale_bbox)

################################################################
########################## SETUP ###############################
################################################################


def get_siteinfo(imagedir):
    """Uses image directory to gather site specific information.
            Agnostic to what relative path structure is used. As in it does
            not matter whether parent directory of images is sitedir or "developed". 

        Returns:
            sitedir: developed image parent directory name
            site_id: state id takend from sitedir
        """
    imagedir = Path(imagedir)
    states = ['TX', 'NC', 'MD']
    sitedir = [p for st in states for p in imagedir.parts if st in p][0]
    site_id = [st for st in states if st in sitedir][0]
    return sitedir, site_id


def get_image_meta(path):
    with open(path) as f:
        j = json.load(f)
        imgdata = from_dict(data_class=ImageData,
                            data=j,
                            config=Config(check_types=False))
    return imgdata


def save_cutout_json(cutout, cutoutpath):
    cutout_json_path = Path(
        Path(cutoutpath).parent,
        Path(cutoutpath).stem + ".json")
    with open(cutout_json_path, 'w') as j:
        json.dump(cutout, j, indent=4, default=str)


def get_species_info(path, cls, default_species="grass"):
    with open(path) as f:
        spec_info = json.load(f)
        spec_info = spec_info["species"][cls] if cls in spec_info[
            "species"].keys() else default_species
    return spec_info


################################################################
######################## PROCESSING ############################
################################################################


class GenCutoutProps:

    def __init__(self, mask):
        """ Generate cutout properties and returns them as a dataclass.
        """
        self.mask = mask

    def from_regprops_table(self, connectivity=2):
        """Generates list of region properties for each cutout mask
        """
        labels = measure.label(self.mask, connectivity=connectivity)
        props = [measure.regionprops_table(labels, properties=CUTOUT_PROPS)]
        # Parse regionprops_table
        nprops = [parse_dict(d) for d in props][0]
        return nprops

    def to_dataclass(self):
        table = self.from_regprops_table()
        cutout_props = from_dict(data_class=CutoutProps, data=table)
        return cutout_props


class ClassifyMask:

    def otsu(self, vi):
        # Otsu's thresh
        vi_mask = otsu_thresh(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255
        return reduce_holes_mask

    def kmeans(self, vi):
        vi_mask = make_kmeans(vi)
        reduce_holes_mask = reduce_holes(vi_mask * 255) * 255

        return reduce_holes_mask


class VegetationIndex:

    def exg(self, img):
        exg_vi = make_exg(img, thresh=True)
        return exg_vi

    def exr(self, img):
        exr_vi = make_exr(img)
        return exr_vi

    def exg_minus_exr(self, img):
        gmr_vi = make_exg_minus_exr(img)
        return gmr_vi

    def ndi(self, img):
        ndi_vi = make_ndi(img)
        return ndi_vi


def get_image_meta(path):
    with open(path) as f:
        j = json.load(f)
        imgdata = from_dict(data_class=ImageData,
                            data=j,
                            config=Config(check_types=False))
    return imgdata


def thresh_vi(vi, low=20, upper=100, sigma=2):
    """
    Args:
        vi (np.ndarray): vegetation index single channel
        low (int, optional): lower end of vi threshold. Defaults to 20.
        upper (int, optional): upper end of vi threshold. Defaults to 100.
        sigma (int, optional): multiplication factor applied to range within
                                "low" and "upper". Defaults to 2.
    """
    thresh_vi = np.where(vi <= 0, 0, vi)
    thresh_vi = np.where((thresh_vi > low) & (thresh_vi < upper),
                         thresh_vi * sigma, thresh_vi)
    return thresh_vi


def seperate_components(mask):
    """ Seperates multiple unconnected components in a mask
        for seperate processing. 
    """
    # Store individual plant components in a list
    mask = mask.astype(np.uint8)
    nb_components, output, _, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8)
    # Remove background component
    nb_components = nb_components - 1
    list_filtered_masks = []
    for i in range(0, nb_components):
        filtered_mask = np.zeros((output.shape))
        filtered_mask[output == i + 1] = 255
        list_filtered_masks.append(filtered_mask)
    return list_filtered_masks


def prep_bbox(box, scale):
    box = rescale_bbox(box, scale)
    x1, y1 = box.local_coordinates["top_left"]
    x2, y2 = box.local_coordinates["bottom_right"]
    x1, y1 = int(x1), int(y1)
    x2, y2 = int(x2), int(y2)
    return box, x1, y1, x2, y2


def get_watershed(mask, disk1=1, grad1_thresh=12, disk2=10, lbl_fact=2.5):
    # process the watershed
    markers = rank.gradient(mask, disk(disk1)) < grad1_thresh
    markers = ndi.label(markers)[0]
    gradient = rank.gradient(mask, disk(disk2))
    labels = watershed(gradient, markers)
    seg1 = label(labels <= 0)
    lbls = label2rgb(seg1, image=mask, bg_label=0) * lbl_fact
    wtrshed_lbls = rescale_intensity(lbls, in_range=(0, 1), out_range=(0, 1))
    return wtrshed_lbls
