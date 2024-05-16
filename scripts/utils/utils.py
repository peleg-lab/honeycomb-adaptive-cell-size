import math
import os
import sys
from pathlib import Path

import numpy as np
import cv2
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.signal import gaussian
from scipy.signal import argrelextrema
from PIL import Image
from PIL.TiffTags import TAGS

def my_imshow(rows, cols, images, cmap='gray'):
    """
    This function displays images in a (rows x cols) grid.

    Parameters:
    rows (int): The number of rows in the grid of subplots.
    cols (int): The number of columns in the grid of subplots.
    images (list): A list of images to display row by row on the grid.
    cmap (string): Custom colur map for imshow(). Default: 'gray'

    Returns:
    No return value.
    """
    # Handle boundary conditions
    if len(images) > rows * cols:
        raise Exception('Number of images passed exceeds dimensions of grid')
    
    subfigs = []
    fig = plt.figure(figsize = (cols * 7, rows * 7))

    # Add subplots into the grid
    for i in range(rows):
        for j in range(cols):
            subfigs.append(fig.add_subplot(rows, cols, i * cols + j + 1))

    # Show images on the subplots created
    for i in range(len(images)):
        if cmap == 'gray':
            subfigs[i].imshow(images[i], cmap='gray', vmin=0, vmax=255)
        else:
            subfigs[i].imshow(images[i], cmap=cmap)

def write_tiff(tiff_template, dest_path, img):    
    """
    This function saves an input image as a TIFF image with metadata.

    Parameters:
    tiff_template (TIFF image): A sample TIFF image whose metadata is used as a template.
    dest_path (string): Output path at which the TIFF image is to be saved.
    img (np.ndarray): A histogram thresholded image from the dataset.

    Returns:
    No return value.
    """
    out = Image.fromarray(img)
    out.save(dest_path, tiffinfo = tiff_template.tag_v2)

def get_tiff_dict(path):
    """
    This function finds all sub-directories of path containing .tiff files

    Parameters:
    path (string): Parent directory to search for .tiff files.

    Returns:
    tiff_dirs (dict): A dictionary in which key: sub-directory, value: list of all .tiff files within the sub-directory
    """
    # Get the paths of all .tiff files in the sub-directories of path
    tiff_files = Path(path).rglob('*.tiff')
    tiff_dirs = {}

    # Form a list of tiff files within the same directory and add it
    # to the dict tiff_dirs with the directory path as key
    for tiff_file in tiff_files:
        tiff_file = str(tiff_file)
        if tiff_file.split('/')[-1][0] == '.':
            continue

        tiff_dir = '/'.join(tiff_file.split('/')[: -1])
        if tiff_dir in tiff_dirs.keys():
            tiff_dirs[tiff_dir].append(tiff_file)
        else:
            tiff_dirs[tiff_dir] = [tiff_file]
    
    return tiff_dirs

def normalize(img):
    """
    This function normalizes the input image using cv2.equalizeHist() to improve contrast

    Parameters:
    img (np.ndarray): The input image to normalize

    Returns:
    The normalized image
    """
    return cv2.equalizeHist(img)


# Define global variables used further in the notebook
dataset_root = '../input/'
datasets = ['s_o5', 's_o75', 's_1', 's_1o25', 's_1o5', 's_1o75', 's_2', 's_3']