import json
import os
import platform
from datetime import datetime
from difflib import get_close_matches
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage
from skimage import morphology, segmentation
from sklearn.cluster import KMeans

######################################################
################### GENERAL ##########################
######################################################


def read_json(path):
    # Opening JSON file
    with open(path) as json_file:
        data = json.load(json_file)
    return data


def write_json(path, data):
    # data = json.dumps(data, indent=4)
    with open(path, "w") as outfile:
        json.dump(data, outfile)


######################################################
################## GET METADATA ######################
######################################################


def get_bbox_info(csv_path):

    df = pd.read_csv(csv_path).drop(columns=['Unnamed: 0'])
    bbox_dict = df.groupby(
        by='imgname', sort=True).apply(lambda x: x.to_dict(orient='records'))
    img_list = list(bbox_dict.keys())
    return bbox_dict, img_list


def get_site_id(imagedir):
    # Must be in TX_2022-12-31 format
    imgstem = Path(imagedir).stem
    siteid = imgstem.split("_")[0]
    return siteid


def creation_date(path_to_file):
    """
    Try to get the date that a file was created, falling back to when it was
    last modified if that isn't possible.
    See http://stackoverflow.com/a/39501288/1709587 for explanation.
    """
    if platform.system() == 'Windows':
        return os.path.getctime(path_to_file)
    else:
        stat = os.stat(path_to_file)
        try:
            return stat.st_birthtime
        except AttributeError:
            # We're probably on Linux. No easy way to get creation dates here,
            # so we'll settle for when its content was last modified.
            return stat.st_mtime


def get_upload_datetime(imagedir):

    creation_dt = creation_date(imagedir)
    creation_dt = datetime.fromtimestamp(creation_dt).strftime(
        '%Y-%m-%d_%H:%M:%S')
    return creation_dt


def parse_dict(props_tabl):
    """Used to parse regionprops table dictionary"""
    ndict = {}
    for key, val in props_tabl.items():
        key = key.replace("-", "") if "-" in key else key
        new_val_entry = []
        if isinstance(val, np.ndarray) and val.shape[0] > 1:
            for i, v in enumerate(val):
                new_val_entry.append({f'{key}_{i+1}': float(v)})
            ndict[key] = new_val_entry
        else:
            ndict[key] = float(val)
    return ndict


def img2RGBA(img):
    alpha = np.sum(img, axis=-1) > 0
    alpha = np.uint8(alpha * 255)
    img = np.dstack((img, alpha))
    return img


######################################################
############### VEGETATION INDICES ###################
######################################################


def make_exg(img, normalize=False, thresh=0):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXG = 2 * G - R - B
    img = img.astype(float)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    if normalize:
        total = r + g + b
        exg = 2 * (g / total) - (r / total) - (b / total)
    else:
        exg = 2 * g - r - b
    if thresh is not None and normalize == False:
        exg = np.where(exg < thresh, 0, exg)
        return exg.astype("uint8")


def make_exr(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # EXR = 1.4 * R - G
    img = rgb_img.astype(float)

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]

    exr = 1.4 * red - green
    exr = np.where(exr < 0, 0,
                   exr)    # Thresholding removes low negative values
    return exr.astype("uint8")


def make_exg_minus_exr(img):
    img = img.astype(float)    # Rgb image
    exg = make_exg(img)
    exr = make_exr(img)
    exgr = exg - exr
    exgr = np.where(exgr < 25, 0, exgr)
    return exgr.astype("uint8")


def make_ndi(rgb_img):
    # rgb_img: np array in [RGB] channel order
    # exr: single band vegetation index as np array
    # NDI = 128 * (((G - R) / (G + R)) + 1)
    img = rgb_img.astype(float)

    blue = img[:, :, 2]
    green = img[:, :, 1]
    red = img[:, :, 0]
    gminr = green - red
    gplusr = green + red
    gdivr = np.true_divide(gminr,
                           gplusr,
                           out=np.zeros_like(gminr),
                           where=gplusr != 0)
    ndi = 128 * (gdivr + 1)
    # print("Max ndi: ", ndi.max())
    # print("Min ndi: ", ndi.min())

    return ndi


######################################################
###################### BBOX ##########################
######################################################


def rescale_bbox(box, imgshape):
    # TODO change
    """Rescales local bbox coordinates, that were first scaled to "downscaled_photo" size (height=3184, width=4796),
       to original image size (height=6368, width=9592). Takes in and returns "Box" dataclass.

    Args:
        box (dataclass): box metedata from bboxes from image metadata
        imgshape: np.ndarray: dimensions of the image to be scaled to (widt, height)
    
    Returns:
        box (dataclass): box metadata with scaled/updated bbox
    """
    scale = imgshape
    box.local_coordinates["top_left"] = [
        c * s for c, s in zip(box.local_coordinates["top_left"], scale)
    ]
    box.local_coordinates["top_right"] = [
        c * s for c, s in zip(box.local_coordinates["top_right"], scale)
    ]
    box.local_coordinates["bottom_left"] = [
        c * s for c, s in zip(box.local_coordinates["bottom_left"], scale)
    ]
    box.local_coordinates["bottom_right"] = [
        c * s for c, s in zip(box.local_coordinates["bottom_right"], scale)
    ]
    return box


######################################################
################# MORPHOLOGICAL ######################
######################################################


def clean_mask(mask, kernel_size=3, iterations=1, dilation=True):
    if int(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    mask = morphology.opening(mask, morphology.disk(3))
    mask = mask.astype("float32")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)
    if dilation:
        mask = cv2.dilate(mask, kernel, iterations=iterations)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, (5, 5))
    mask = cv2.erode(mask, (5, 5), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, (7, 7))
    return mask


def dilate_erode(mask,
                 kernel_size=3,
                 dil_iters=5,
                 eros_iters=3,
                 hole_fill=True):
    mask = mask.astype(np.float32)

    if int(kernel_size):
        kernel_size = (kernel_size, kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, kernel_size)

    mask = cv2.dilate(mask, kernel, iterations=dil_iters)
    if hole_fill:
        mask = ndimage.binary_fill_holes(mask.astype(np.int32))
    mask = mask.astype("float")
    mask = cv2.erode(mask, kernel, iterations=eros_iters)

    cleaned_mask = clean_mask(mask)
    return cleaned_mask


def clear_border(mask):
    mask = segmentation.clear_border(mask)
    return mask


def reduce_holes(mask, min_object_size=1000, min_hole_size=1000):

    mask = mask.astype(np.bool8)
    # mask = measure.label(mask, connectivity=2)
    mask = morphology.remove_small_holes(
        morphology.remove_small_objects(mask, min_hole_size), min_object_size)
    # mask = morphology.opening(mask, morphology.disk(3))
    return mask


######################################################
########### CLASSIFIERS AND THRESHOLDING #############
######################################################


def check_kmeans(mask):
    max_sum = mask.shape[0] * mask.shape[1]
    ones_sum = np.sum(mask)
    if ones_sum > max_sum / 2:
        mask = np.where(mask == 1, 0, 1)
    return mask


def make_kmeans(exg_mask):
    rows, cols = exg_mask.shape
    n_classes = 2
    exg = exg_mask.reshape(rows * cols, 1)
    kmeans = KMeans(n_clusters=n_classes, random_state=3).fit(exg)
    mask = kmeans.labels_.reshape(rows, cols)
    mask = check_kmeans(mask)
    return mask.astype("uint64")


def otsu_thresh(mask, kernel_size=(3, 3)):
    mask_blur = cv2.GaussianBlur(mask, kernel_size, 0).astype("uint16")
    ret3, mask_th3 = cv2.threshold(mask_blur, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask_th3


######################################################
##################### MASKING ########################
######################################################


def apply_mask(img, mask, mask_color):
    """Apply white image mask to image, with bitwise AND operator bitwise NOT operator and ADD operator.
    Inputs:
    img        = RGB image data
    mask       = Binary mask image data
    mask_color = 'white' or 'black'
    Returns:
    masked_img = masked image data
    :param img: numpy.ndarray
    :param mask: numpy.ndarray
    :param mask_color: str
    :return masked_img: numpy.ndarray
    """
    if mask_color.upper() == "WHITE":
        color_val = 255
    elif mask_color.upper() == "BLACK":
        color_val = 0

    array_data = img.copy()

    # Mask the array
    array_data[np.where(mask == 0)] = color_val
    return array_data


######################################################
#################### CUTOUTS #########################
######################################################


def trans_cutout(imgpath):
    """ Get transparent cutout from cutout image with black background"""
    img = cv2.cvtColor(cv2.imread(imgpath), cv2.COLOR_BGR2RGB)
    # threshold on black to make a mask
    color = (0, 0, 0)
    mask = np.where((img == color).all(axis=2), 0, 255).astype(np.uint8)

    # put mask into alpha channel
    result = img.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    result[:, :, 3] = mask
    return result


def crop_cutouts(img, add_padding=False):
    foreground = Image.fromarray(img)
    pil_crop_frground = foreground.crop(foreground.getbbox())
    array = np.array(pil_crop_frground)
    if add_padding:
        pil_crop_frground = foreground.crop((
            foreground.getbbox()[0] - 2,
            foreground.getbbox()[1] - 2,
            foreground.getbbox()[2] + 2,
            foreground.getbbox()[3] + 2,
        ))
    return array


def get_species_info(path, cls, default_species="plant"):
    with open(path) as f:
        spec_info = json.load(f)
        spec_info = (spec_info["species"][cls]
                     if cls in spec_info["species"].keys() else
                     spec_info["species"][default_species])
    return spec_info
