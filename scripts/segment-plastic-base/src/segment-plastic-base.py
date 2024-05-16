#!/usr/bin/env python
# coding: utf-8

import math
import os
from pathlib import Path
import argparse

import numpy as np
import cv2
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from scipy.signal import gaussian
from scipy.signal import argrelextrema
from PIL import Image
from PIL.TiffTags import TAGS

from scripts.utils.utils import *

def get_plastic_mask_sliding_window(img, left, dilation_iters):
    """
    This function computes the plastic base mask for a single histogram-thresholded dataset image.

    Parameters:
    img (np.ndarray): A histogram thresholded image from the dataset.
    left (bool): Switch to indicate the relative position of the honeycomb to the plastic base.
    dilation_iters (int): A tunable parameter to control the degree of dilation while creating the mask.

    Returns:
    np.ndarray: Mask to use to filter out the plastic base.
    """
    mask = np.zeros(img.shape)
    plastic_rect = np.zeros(img.shape)

    kernel_rows = 300
    kernel_cols = 100
        
    # Create the kernel to identify the boundary of the plastic base and air
    kernel = np.ones((kernel_rows, kernel_cols))
    if left:
        kernel[:, 50:] = -1
        col_range = np.arange(img.shape[1] - kernel_cols, 0, -10)
    else:
        kernel[:, :50] = -1
        col_range = np.arange(0, img.shape[1] - kernel_cols, 10)

    # Convolve the image with the kernel by using it as a sliding window
    for i in range(0, img.shape[0] - kernel_rows, kernel_rows):
        highest = 0
        highest_index = -1
        for j in col_range:
            # Heuristic to mark the boundary of the plastic base and air
            if np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel) > highest:
                highest = np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel)
                highest_index = j

        if left:
            mask[i: i + kernel_rows, highest_index + 90: highest_index + 120] = 255
        else:
            mask[i: i + kernel_rows, highest_index: highest_index + 30] = 255

    # Convolve the last strip that gets excluded above
    i = img.shape[0] - kernel_rows
    highest = 0
    highest_index = -1
    for j in col_range:
        # Heuristic to mark the boundary of the plastic base and air
        if np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel) > highest:
            highest = np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel)
            highest_index = j
    
    if left:
        mask[i: i + kernel_rows, highest_index + 90: highest_index + 120] = 255
    else:
        mask[i: i + kernel_rows, highest_index: highest_index + 30] = 255

    # Create a contour of the identified boundary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_wo_dilation = max(contours, key = cv2.contourArea)

    # Find the bounding box for this contour
    rect = cv2.minAreaRect(max_contour_wo_dilation)
    box = cv2.boxPoints(rect)
    plastic_box_wo_dilation = np.intp(box)

    # Dilate to create the final mask
    cv2.drawContours(plastic_rect, [plastic_box_wo_dilation], -1, (255, 255, 255), -1)
        
    dilated_result = cv2.dilate(plastic_rect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)), iterations = dilation_iters)
    
    return dilated_result

def get_plastic_mask_sliding_window_o5x(img, left, dilation_iters):
    """
    This function computes the plastic base mask for a single histogram-thresholded o5x or o75x image.
    We have a separate function for o5x and o75x because it is not as clean as the other datasets and needs special handling.

    Parameters:
    img (np.ndarray): A histogram thresholded image from the dataset.
    
    Returns:
    np.ndarray: Mask to use to filter out the plastic base.
    """
    mask = np.zeros(img.shape)
    plastic_rect = np.zeros(img.shape)

    kernel_rows = 300
    kernel_cols = 100
        
    # Create the kernel to identify the boundary of the plastic base and air
    kernel = np.ones((kernel_rows, kernel_cols))
    # o5x data
    kernel[:, 35:65] = -1
    if left:
        col_range = np.arange(img.shape[1] - kernel_cols, 0, -10)
    else:
        col_range = np.arange(0, img.shape[1] - kernel_cols, 10)

    # Convolve the image with the kernel by using it as a sliding window
    for i in range(0, img.shape[0] - kernel_rows, kernel_rows):
        highest = 0
        highest_index = -1
        for j in col_range:
            # Heuristic to mark the boundary of the plastic base and air
            if np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel) > highest:
                highest = np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel)
                highest_index = j

        if left:
            mask[i: i + kernel_rows, highest_index + 90: highest_index + 120] = 255
        else:
            mask[i: i + kernel_rows, highest_index: highest_index + 30] = 255

    # Convolve the last strip that gets excluded above
    i = img.shape[0] - kernel_rows
    highest = 0
    highest_index = -1
    for j in col_range:
        # Heuristic to mark the boundary of the plastic base and air
        if np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel) > highest:
            highest = np.sum(img[i: i + kernel_rows, j: j + kernel_cols] * kernel)
            highest_index = j
    
    if left:
        mask[i: i + kernel_rows, highest_index + 90: highest_index + 120] = 255
    else:
        mask[i: i + kernel_rows, highest_index: highest_index + 30] = 255

    # Create a contour of the identified boundary
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_contour_wo_dilation = max(contours, key = cv2.contourArea)

    # Find the bounding box for this contour
    rect = cv2.minAreaRect(max_contour_wo_dilation)
    box = cv2.boxPoints(rect)
    plastic_box_wo_dilation = np.intp(box)

    # Dilate to create the final mask
    cv2.drawContours(plastic_rect, [plastic_box_wo_dilation], -1, (255, 255, 255), -1)
    dilated_result = cv2.dilate(plastic_rect.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25)), iterations = dilation_iters)
    
    return dilated_result

def filter_out_plastic_base(path, left, iters):
    """
    Driver function to compute the plastic base masks for the dataset images

    Parameters:
    path (string): Path to the top level directory of the dataset.
    left (bool): Switch to indicate the relative position of the honeycomb to the plastic base.
    dilation_iters (int): A tunable parameter to control the degree of dilation while creating the mask.

    Returns:
    No return value.
    """
    tiff_file_path =  path + '/hist_threshold/tiff'
    tiff_files = get_tiff_dict(tiff_file_path)

    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    plastic_base_masked_solution_path = curr_dir_path + f'../output/{path.split(os.path.sep)[-1]}/plastic_base_masked/'

    # Create the output directory if it does not already exist
    if not os.path.isdir(plastic_base_masked_solution_path):
        os.makedirs(plastic_base_masked_solution_path)

    if not os.path.isdir(f'{plastic_base_masked_solution_path}/tiff/'):
        os.makedirs(f'{plastic_base_masked_solution_path}/tiff/')

    tiff_template = Image.open(tiff_files[tiff_file_path][0])
    temp_img = cv2.imread(tiff_files[tiff_file_path][0], cv2.IMREAD_GRAYSCALE)
    plastic_base_masked_out = cv2.VideoWriter(plastic_base_masked_solution_path + path.split('/')[-1] + '.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (temp_img.shape[1], temp_img.shape[0]))

    print('Begin plastic base segmentation of ' + path)
    for file in tqdm(sorted(tiff_files[tiff_file_path])):
        img = cv2.imread(file, cv2.IMREAD_GRAYSCALE)
        tiff_image = cv2.imread(file, -1)
        
        # Get plastic base mask from get_plastic_mask_sliding_window()
        dataset = path.split(os.path.sep)[-1]
        if dataset == 's_o5' or dataset == 's_o75':
            plastic_mask = get_plastic_mask_sliding_window_o5x(img, left, iters)
        else:
            plastic_mask = get_plastic_mask_sliding_window(img, left, iters)

        plastic_base_masked_image = cv2.bitwise_and(img, 255 - plastic_mask)
        tiff_plastic_base_masked = cv2.bitwise_and(tiff_image, ((255 - plastic_mask) > 0).astype(np.uint16) * 0xffff)
        write_tiff(tiff_template, f'{plastic_base_masked_solution_path}/tiff/{file.split(os.path.sep)[-1]}', tiff_plastic_base_masked)
        plastic_base_masked_out.write(cv2.cvtColor(plastic_base_masked_image, cv2.COLOR_GRAY2BGR))
    print('Finish plastic base segmentation of ' + path)
    
    plastic_base_masked_out.release()

if __name__ == '__main__':
    curr_dir_path = os.path.dirname(os.path.abspath(__file__)) + '/'

    parser = argparse.ArgumentParser(description="Histogram thresholding on honeycomb data")
    parser.add_argument('--datasets', nargs='+', type=str, help='a list of datasets to process')
    args = parser.parse_args()

    if args.datasets != ['default']:
        datasets = args.datasets
    
    plastic_base_mask_parameters = {
        's_o5': [True, 9], # o5x
        's_o75': [False, 7], # o75x
        's_1': [True, 14], # 1x
        's_1o25': [True, 13], # 1o25x
        's_1o5': [True, 13], # 1o5x
        's_1o75': [False, 13], # 1o75x
        's_2': [True, 9], # 2x
        's_3': [True, 18], # 3x
    }
    
    # Mask out the plastic base for the required datasets
    for dataset in datasets:
        if os.path.isdir(curr_dir_path + dataset_root + dataset + '/hist_threshold'):
            filter_out_plastic_base(curr_dir_path + dataset_root + dataset, plastic_base_mask_parameters[dataset][0], plastic_base_mask_parameters[dataset][1])